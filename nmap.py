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
    def __init__(self, args, nbatch, nsteps, reuse=False):
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

        nenv = nbatch // nsteps
        self.memory = tf.placeholder(tf.float32,
            shape=[nenv, args['memory_channels'],
                args['memory_size'],
                args['memory_size']],
            name='memory'
        )
        self.old_c_t = tf.placeholder(tf.float32, shape=[nenv, args['memory_channels']], name='old_c_t')
        self.ctx_state_input = tf.placeholder(tf.float32, [2, nenv, args['memory_channels']], name='ctx_state_input')
        self.ctx_state_tuple = tf.nn.rnn_cell.LSTMStateTuple(self.ctx_state_input[0], self.ctx_state_input[1])
        self.inputs = tf.placeholder(tf.float32,
                    shape=[nbatch] + args['input_dims'],
                    name='inputs')
        self.pos = tf.placeholder(tf.float32, shape=[nbatch, 2], name='pos')
        self.p_pos = tf.placeholder(tf.float32, shape=[nbatch, 2], name='p_pos')
        self.timestep = tf.placeholder(tf.int64, shape=[nbatch, 1], name='timestep')
        self.masks = tf.placeholder(tf.float32, shape=[nbatch,], name='masks')


        self.memory_out, self.c_t_out, self.ctx_state_tuple_out, self.feats, self.ctx_prob = get_model(args, nbatch, nsteps,
            self.inputs, self.memory, self.old_c_t, self.ctx_state_tuple,
            self.pos, self.p_pos, self.timestep, self.masks, reuse=reuse)



class NeuralMapPolicy(object):
    def __init__(self, sess, ob_space, ac_space, args, nbatch, nsteps, reuse=False):
        with tf.variable_scope('model', reuse=reuse):
            self.nmap = NeuralMap(args, nbatch, nsteps)
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
            a, v0, new_memory, c_t, c_new, neglogp = sess.run([
                        self.A,
                        self.v0,
                        self.nmap.memory,
                        self.nmap.c_t_out,
                        self.nmap.ctx_state_tuple_out,
                        self.neglogp,
                    ], feed_dict={
                self.nmap.inputs: obs_img,
                self.nmap.memory: memory,
                self.nmap.pos: info['curr_loc'],
                self.nmap.p_pos: info['past_loc'],
                self.nmap.timestep: [[t[0] % self.max_timestep] for t in info['step_counter']],
                self.nmap.old_c_t: old_c_t,
                self.nmap.ctx_state_tuple: ctx_state,
                self.nmap.masks: done
            })
            return a, v0, (new_memory, c_t, c_new), neglogp


        def vizstep(obs, state, done):
            obs_img, info = obs
            memory, old_c_t, ctx_state = state

            # shift memory first
            a, new_memory, c_t, c_new, ctx_prob  = sess.run([
                        self.A,
                        self.nmap.memory,
                        self.nmap.c_t_out,
                        self.nmap.ctx_state_tuple_out,
                        self.nmap.ctx_prob
                    ], feed_dict={
                self.nmap.inputs: obs_img,
                self.nmap.memory: memory,
                self.nmap.pos: info['curr_loc'],
                self.nmap.p_pos: info['past_loc'],
                self.nmap.timestep: [[t[0] % self.max_timestep] for t in info['step_counter']],
                self.nmap.old_c_t: old_c_t,
                self.nmap.ctx_state_tuple: ctx_state,
                self.nmap.masks: done
            })
            return a, (new_memory, c_t, c_new), ctx_prob

        def value(obs, state, done):
            obs_img, info = obs
            memory, old_c_t, ctx_state = state
            return sess.run([
                        self.v0,
                    ], feed_dict= {
                self.nmap.inputs: obs_img,
                self.nmap.memory: memory,
                self.nmap.pos: info['curr_loc'],
                self.nmap.p_pos: info['past_loc'],
                self.nmap.timestep: [[t[0] % self.max_timestep] for t in info['step_counter']],
                self.nmap.old_c_t: old_c_t,
                self.nmap.ctx_state_tuple: ctx_state,
                self.nmap.masks: done
            })
        self.step = step
        self.vizstep = vizstep
        self.value = value

    def get_initial_state(self,nenv):
        initial_memory = np.zeros((nenv, self.args['memory_channels'], self.args['memory_size'], self.args['memory_size']))
        initial_old_c_t = np.zeros((nenv, self.args['memory_channels']))
        initial_ctx_state = (np.zeros((nenv, self.args['memory_channels'])), np.zeros((nenv, self.args['memory_channels'])))
        initial_state = (initial_memory, initial_old_c_t, initial_ctx_state)
        return initial_state
