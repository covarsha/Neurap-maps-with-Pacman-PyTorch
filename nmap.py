from __future__ import print_function

import cv2
import math

import logging
import numpy        as np
import numpy.random as npr

import gym
import gym_pacman

from utils import *

from baselines.common.distributions import make_pdtype

fc = tf.contrib.layers.fully_connected
conv2d = tf.contrib.layers.conv2d
arg_scope = tf.contrib.framework.arg_scope
flatten = tf.contrib.layers.flatten


class NeuralMap(object):
    def __init__(self, args, env, input_dims):
        self.args = args
        self.input_dims      = input_dims
        self.write_type      = args['nmapw_nl']
        self.memory_size     = args['memory_size']
        self.memory_channels = args['memory_channels']
        self.rescale_max     = args['rescale_max']
        self.egocentric      = args['egocentric']
        self.access_orient   = args['access_orient']
        self.erasure_on      = args['erasure_on']
        self.diffusion_on    = args['diffusion_on']

        self.use_position = args['use_position']
        self.use_orient   = args['use_orient']
        self.use_velocity = args['use_velocity']
        self.use_timestep = args['use_timestep']
        self.max_timestep = args['max_timestep']

        # Memory channels needs to be divisible by # of orientations if 'access_orient' is true
        assert((not self.access_orient) or (self.memory_channels%4 == 0))

        # Supported write types
        assert((self.write_type == 'gru') or (self.write_type == 'lstm'))

        input_size = self.input_dims[0] * self.input_dims[1] # *self.input_dims[2]

        ##############################################################
        # State model
        self.s_t, self.inputs = basenet(args, input_dims)

        # READ NETWORK
        self.r_t, self.memory = read_network(args)
        max_maze_size = max(env.MAX_MAZE_SIZE)
        # TODO: remember to increment timestep when training
        self.extras = {
            'pos': tf.placeholder(tf.float32, shape=[2], name='pos'),
            'p_pos': tf.placeholder(tf.float32, shape=[2], name='p_pos'),
            'orientation': tf.placeholder(tf.float32, shape=[1], name='orientation'),
            'p_orientation': tf.placeholder(tf.float32, shape=[1], name='p_orientation'),
            'timestep': tf.placeholder(tf.int64, shape=[1], name='timestep'),
            'max_maze_size': tf.constant(max_maze_size, dtype=tf.float32, name='max_maze_size'),
            'maze_size': tf.constant(env.maze_size, dtype=tf.float32, name='maze_size')
        }

        self.m0 = tf.Variable(0.01 * np.random.randn(1,args['memory_channels'], 1, 1))

        self.old_c_t = tf.placeholder(tf.float32, shape=[1,1, args['memory_channels']], name='old_c_t')

        self.ctx_state_input = tf.placeholder(tf.float32, [2, 1, args['memory_channels']])
        self.ctx_state_tuple = tf.nn.rnn_cell.LSTMStateTuple(self.ctx_state_input[0], self.ctx_state_input[1])
        print(self.ctx_state_tuple)
        self.c_t, self.ctx_hx, self.shift_memory, self.ctx_state_new = context_network(args,
            self.s_t,
            self.r_t,
            self.memory,
            self.old_c_t,
            self.extras,
            self.m0,
            self.ctx_state_tuple)

        self.w_t, self.updated_memory = write_network(
            args,
            self.s_t,
            self.r_t,
            self.c_t,
            self.shift_memory)

        self.feats = fc(
            tf.concat([tf.squeeze(self.ctx_hx, 0), self.c_t, tf.squeeze(self.w_t, 0)], 1),
            args['memory_channels'],
            activation_fn=tf.nn.elu) 

class NeuralMapPolicy(object):
    def __init__(self, args, env, input_dims):
        self.nmap = NeuralMap(args, env, input_dims)
        self.pi = fc(
            self.nmap.feats,
            env.action_space.n,
            activation_fn=None)
        self.vf = fc(
            self.nmap.feats,
            1,
            activation_fn=None)
        self.pdtype = make_pdtype(env.action_space)
        self.pd = self.pdtype.pdfromflat(self.pi)
        self.A = self.pd.sample()
        self.neglogp = self.pd.neglogp(self.A)



class NeuralMapModel(object):
    """
        https://github.com/openai/baselines/blob/master/baselines/ppo2/ppo2.py
    """
    def __init__(self, nmap_args, env, input_dims, nsteps=5, ent_coef=0.01,
            vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95):

        self.model = NeuralMapPolicy(nmap_args, env, input_dims)
        self.gamma = gamma
        self.lam = lam
        A = self.model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = self.model.pd.neglogp(A)
        entropy = tf.reduce_mean(self.model.pd.entropy())

        vpred = self.model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(self.model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        with tf.variable_scope('model'):
            params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)

        def train(lr, obs, returns, masks, actions, values, neglogpacs, states):
            cliprange = 0.2
            old_c_t, memory, ctx_cx = states
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {
                train_model.X: obs, 
                A: actions, 
                ADV: advs,
                R: returns, 
                LR: lr,
                CLIPRANGE: cliprange, 
                OLDNEGLOGPAC: neglogpacs, 
                OLDVPRED: values
            }
            # td_map[train_model.S] = states
            # td_map[train_model.M] = masks
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                td_map
            )[:-1]

        tf.global_variables_initializer().run(session=sess)

        self.train = train

        self.old_c_t = np.zeros((1,256))
        self.ctx_state = np.zeros((2,1,args['memory_channels']))
        self.memory = 0.01 * np.random.randn(
            1,
            args['memory_channels'],
            args['memory_size'],
            args['memory_size']
        )

        self.nsteps = nsteps

    def run(self, env, nepochs=100):
        for _ in range(nepochs):
            rollout = self.do_rollout(env)
            advantages = self.calculate_advantages(rollout)
            # self.do_train(advantages)

    def do_rollout(self, env):
        obs = []
        rewards = []
        dones = []
        values = []
        neglogpacs = []

        for x in range(self.nsteps):
            action, value, neglogp = sess.run([
                self.model.A,
                self.model.vf,
                self.model.neglogp])
            obs, reward, done, info = env.step()
            obs = np.expand_dims(s, 0)

            # append experience
            obs.append((obs, info))
            actions.append(action)
            dones.append(done)
            neglogpacs.append(neglogp)
            values.append(value)
            
            self.old_c_t = np.expand_dims(self.old_c_t, 0)
            self.memory, self.old_c_t, self.ctx_cx = sess.run([
                    nmap.memory, 
                    nmap.c_t, 
                    nmap.ctx_state_new], feed_dict={
                nmap.inputs: s,
                nmap.memory: memory,
                nmap.extras['pos']: info['curr_loc'],
                nmap.extras['p_pos']: info['past_loc'],
                nmap.extras['orientation']: [info['curr_orientation']],
                nmap.extras['p_orientation']: [info['past_orientation']],
                nmap.extras['timestep']: [x],
                nmap.old_c_t: old_c_t,
                nmap.ctx_state_input: ctx_state
            })
        nmap_state = (self.old_c_t, self.memory, self.ctx_state_new)
        return (obs, actions, rewards, dones, neglogpacs, values, nmap_state)

    

    # calculate advantages, etc.
    def calculate_advantages(self, rollout):
        obs, actions, rewards, dones, neglogpacs, values, nmap_state = rollout
        mb_obs = np.asarray(states, dtype=self.obs.dtype)
        mb_rewards = np.asarray(rewards, dtype=np.float32)
        mb_actions = np.asarray(actions)
        mb_values = np.asarray(values, dtype=np.float32)
        mb_neglogpacs = np.asarray(neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.obs, self.dones)
        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states)

    # # step through again and flow gradients
    # def do_train(self, data):
    #     (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs), mb_states = data
    #     for x in range(self.nsteps):
            

# helper
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


