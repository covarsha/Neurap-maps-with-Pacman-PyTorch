from __future__ import print_function

import cv2
import math

import logging
import numpy        as np
import numpy.random as npr

import gym
import gym_pacman

from utils import *


fc = tf.contrib.layers.fully_connected
conv2d = tf.contrib.layers.conv2d
arg_scope = tf.contrib.framework.arg_scope
flatten = tf.contrib.layers.flatten


class NeuralMap:
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

        self.feats = tf.contrib.layers.fully_connected(
            tf.concat([tf.squeeze(self.ctx_hx, 0), self.c_t, tf.squeeze(self.w_t, 0)], 1),
            args['memory_channels'],
            activation_fn=tf.nn.elu) 

        # TODO: linear actor critic layers
        # policy_fn
        # value_pred


class ActorCritic:
    def __init__(self, feats):
        self.critic = fc(feats, 1)
        self.actor = fc(feats, 4, activation_fn=tf.nn.softmax)





class NeuralMapPolicy(object):

    def __init__(self, args, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps

        self.nmap = NeuralMap(...)

        self.curr_memory = 0.01 * np.random.randn(1, args['memory_channels'], args['memory_size'], args['memory_size'])
        self.curr_c_t = np.expand_dims(old_c_t, 0)

        pi = fc(nmap.feats, env.action_space[0], activation_fn=tf.nn.softmax)

        self.pdtype = make_pdtype(env.action_space)
        self.pd = self.pdtype.pdfromflat(pi)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            v, self.curr_memory, self.curr_c_t = sess.run([
                v0, self.nmap.memory, self.nmap.ctx_state_new], feed_dict={
                    self.nmap.inputs: state,
                    self.nmap.memory: memory,
                    self.nmap.extras['pos']: info['curr_loc'],
                    self.nmap.extras['p_pos']: info['past_loc'],
                    self.nmap.extras['orientation']: [info['curr_orientation']],
                    self.nmap.extras['p_orientation']: [info['past_orientation']],
                    self.nmap.extras['timestep']: [x],
                    self.nmap.old_c_t: old_c_t,
                    self.nmap.ctx_state_input: ctx_state
            })
            self.curr_memory = memory_new
            return v

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
