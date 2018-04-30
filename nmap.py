from __future__ import print_function

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
    def __init__(self, args):
        self.args = args
        self.input_dims      = args['input_dims']
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
        self.s_t, self.inputs = basenet(args, self.input_dims)

        # READ NETWORK
        self.r_t, self.memory, self.before_flatten, self.after_flatten = read_network(args)
        self.max_maze_size = max(args['max_maze_size'])
        self.maze_size = max(args['maze_size'])
        self.map_scale = 2.0 if args['egocentric'] else 1.0
        # TODO: remember to increment timestep when training
        self.extras = {
            'pos': tf.placeholder(tf.float32, shape=[None, 2], name='pos'),
            'p_pos': tf.placeholder(tf.float32, shape=[None, 2], name='p_pos'),
            'timestep': tf.placeholder(tf.int64, shape=[None, 1], name='timestep'),
            'max_maze_size': tf.constant(self.max_maze_size, dtype=tf.float32, name='max_maze_size'),
            'maze_size': tf.constant(args['maze_size'], dtype=tf.float32, name='maze_size')
        }

        self.old_c_t = tf.placeholder(tf.float32, shape=[None,1, args['memory_channels']], name='old_c_t')

        self.ctx_state_input = tf.placeholder(tf.float32, [2, None, args['memory_channels']], name='ctx_state_input')
        self.ctx_state_tuple = tf.nn.rnn_cell.LSTMStateTuple(self.ctx_state_input[0], self.ctx_state_input[1])
        self.c_t, self.ctx_h, self.ctx_state_new = context_network(args,
            self.s_t,
            self.r_t,
            self.memory,
            self.old_c_t,
            self.extras,
            self.ctx_state_tuple)

        self.w_t = write_network(
            args,
            self.s_t,
            self.r_t,
            self.c_t,
            self.memory)

        self.feats = fc(
            tf.concat([tf.squeeze(self.ctx_h,1), self.c_t, tf.squeeze(self.w_t, 1)], 1),
            args['memory_channels'],
            activation_fn=tf.nn.elu)

    def shift_memory(self, memory, pos, lpos):
        pos, lpos = np.array(pos), np.array(lpos)
        velocity = pos - lpos
        rescale_func = None
        def rescale_max(p):
            return p/(self.max_maze_size * self.map_scale) * memory.shape[2]
        def rescale_maze(p):
            return p/(self.maze_size * self.map_scale) * memory.shape[2]
        if self.rescale_max:
            # Rescale the position data to the maximum map size
            rescale_func = rescale_max
        else:
            # Rescale the position data to the current map size
            rescale_func = rescale_maze

        scaled_pos = rescale_func(pos)
        scaled_p_pos = rescale_func(lpos)
        scaled_velocity = scaled_pos - scaled_p_pos

        shift_memory = 0.01 * np.random.randn(memory.shape[0], memory.shape[1], memory.shape[2], memory.shape[3])
        mem_sz = memory.shape[2]
        srcboundy = (
                    np.maximum(velocity[:,0], 0.0).astype(np.int32),
                    np.minimum(mem_sz + velocity[:,0], mem_sz).astype(np.int32)
            )
        srcboundx = (
                np.maximum(velocity[:,1], 0.0).astype(np.int32),
                np.minimum(mem_sz + velocity[:,1], mem_sz).astype(np.int32)
            )
        dstboundy = (
                np.maximum(-velocity[:,0], 0.0).astype(np.int32),
                np.minimum(mem_sz - velocity[:,0], mem_sz).astype(np.int32)
            )
        dstboundx = (
                np.maximum(-velocity[:,1], 0.0).astype(np.int32),
                np.minimum(mem_sz - velocity[:,1], mem_sz).astype(np.int32)
            )
        for ix in range(memory.shape[0]):
            shift_memory[ix,:,dstboundy[0][ix]:dstboundy[1][ix],dstboundx[0][ix]:dstboundx[1][ix]] = \
                memory[ix,:,srcboundy[0][ix]:srcboundy[1][ix],srcboundx[0][ix]:srcboundx[1][ix]]
        return shift_memory

    def write_to_memory(self, memory, w_t):
        new_memory = np.copy(memory)
        write_py, write_px = memory.shape[2] // 2, memory.shape[3] // 2
        new_memory[:,:,write_py,write_px] = w_t
        return new_memory

class NeuralMapPolicy(object):
    def __init__(self, sess, ob_space, ac_space, args, reuse=False):
        with tf.variable_scope('model', reuse=reuse):
            self.nmap = NeuralMap(args)
            self.pi = fc(
                self.nmap.feats,
                ac_space.n,
                activation_fn=None)
            self.vf = fc(
                self.nmap.feats,
                1,
                activation_fn=None)
            self.v0 = self.vf[:, 0]
            self.pdtype = make_pdtype(ac_space)
            self.pd = self.pdtype.pdfromflat(self.pi)
            self.A = self.pd.sample()
            self.neglogp = self.pd.neglogp(self.A)
        self.max_timestep = args['max_timestep']
        self.args=args


        def step(obs, state, done):
            obs_img, info = obs
            memory, old_c_t, ctx_state = state

            # shift memory first
            shift_memory = self.nmap.shift_memory(memory, info['curr_loc'], info['past_loc'])
            a, v0, w_t, c_t, c_new, neglogp = sess.run([
                        self.A,
                        self.v0,
                        self.nmap.w_t,
                        self.nmap.c_t,
                        self.nmap.ctx_state_new,
                        self.neglogp,
                    ], feed_dict={
                self.nmap.inputs: obs_img,
                self.nmap.memory: shift_memory,
                self.nmap.extras['pos']: info['curr_loc'],
                self.nmap.extras['p_pos']: info['past_loc'],
                self.nmap.extras['timestep']: [[t[0] % self.max_timestep] for t in info['step_counter']],
                self.nmap.old_c_t: old_c_t,
                self.nmap.ctx_state_input: ctx_state
            })

            w_t = np.squeeze(w_t, 1)
            # write to memory here
            new_memory = self.nmap.write_to_memory(shift_memory, w_t)
            return a, v0, new_memory, c_t, c_new, neglogp

        def value(obs, state, done):
            obs_img, info = obs
            memory, old_c_t, ctx_state = state
            return sess.run([
                        self.v0,
                    ], feed_dict= {
                self.nmap.inputs: obs_img,
                self.nmap.memory: memory,
                self.nmap.extras['pos']: info['curr_loc'],
                self.nmap.extras['p_pos']: info['past_loc'],
                self.nmap.extras['timestep']: [[t[0] % self.max_timestep] for t in info['step_counter']],
                self.nmap.old_c_t: old_c_t,
                self.nmap.ctx_state_input: ctx_state
            })

        self.step = step
        self.value = value

    def get_initial_state(self,nenv,past_states=None,dones=None):
        if past_states is not None and dones is not None:
            initial_ = 0.01 * np.random.randn(nenv, self.args['memory_channels'], self.args['memory_size'], self.args['memory_size'])
            initial_memory = np.transpose(np.multiply(np.transpose(past_states[0].copy(),(1,2,3,0)),1-dones),(3,0,1,2))
            initial_memory += np.transpose(np.multiply(np.transpose(initial_,(1,2,3,0)),dones),(3,0,1,2))
            
            initial_old_c_t = np.transpose(np.multiply(np.transpose(past_states[1].copy(),(1,2,0)),1-dones),(2,0,1))
            
            initial_ctx_state = (np.transpose(np.multiply(np.transpose(past_states[2][0].copy(),(1,0)),1-dones),(1,0)),np.transpose(np.multiply(np.transpose(past_states[2][1].copy(),(1,0)),1-dones),(1,0)))
            initial_state = (initial_memory, initial_old_c_t, initial_ctx_state)
            return initial_state
        else:
            initial_memory = 0.01 * np.random.randn(nenv, self.args['memory_channels'], self.args['memory_size'], self.args['memory_size'])
            initial_old_c_t = np.zeros((nenv, 1, self.args['memory_channels']))
            initial_ctx_state = (np.zeros((nenv, self.args['memory_channels'])), np.zeros((nenv, self.args['memory_channels'])))
            initial_state = (initial_memory, initial_old_c_t, initial_ctx_state)
            return initial_state
