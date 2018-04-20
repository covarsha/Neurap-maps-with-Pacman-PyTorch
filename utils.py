import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import argparse

import tensorflow as tf


fc = tf.contrib.layers.fully_connected
conv2d = tf.contrib.layers.conv2d
arg_scope = tf.contrib.framework.arg_scope
flatten = tf.contrib.layers.flatten

def string_to_nl(nlstr):
    # Get read network nonlinearity
    if nlstr == 'relu':
        return tf.nn.relu
    elif nlstr == 'elu':
        return tf.nn.elu
    elif nlstr == 'tanh':
        return tf.nn.tanh
    elif nlstr == 'sigmoid':
        return tf.nn.sigmoid

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1)[:,None].expand_as(out))
    return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


def basenet(args, input_dims):
    with tf.variable_scope('statenet'):
        inputs = tf.placeholder(tf.float32,
            shape=[1] + input_dims,
            name='inputs')
        last_layer = inputs
        for l in range(len(args['n_units'])):
            last_layer = conv2d(inputs=last_layer, 
                num_outputs=args['n_units'][l], 
                kernel_size=[args['filters'][l], args['filters'][l]], \
                stride=[args['strides'][l], args['strides'][l]],
                padding="same",
                activation=string_to_nl(args['nl'][l]))

        last_layer = flatten(last_layer, scope='flatten')

        for l in range(len(args['n_hid'])):
            last_layer = fc(last_layer, args['n_hid'][l], activation_fn=string_to_nl(args['nl'][l]))
        return last_layer, inputs


def read_network(args):
    with tf.variable_scope('readnet'):
        memory = tf.placeholder(tf.float32,
            shape=[1, args['memory_channels'],
                args['memory_size'],
                args['memory_size']],
            name='memory'
        )
        last_layer = memory
        for l in range(len(args['nmapr_n_units'])):
            last_layer = conv2d(inputs=last_layer, 
                num_outputs=args['nmapr_n_units'][l], 
                kernel_size=[args['nmapr_filters'][l], args['nmapr_filters'][l]], \
                stride=[args['nmapr_strides'][l], args['nmapr_strides'][l]],
                padding="same",
                activation_fn=string_to_nl(args['nmapr_nl'][l]))

        last_layer = flatten(last_layer, scope='flatten')

        # TODO: check sharing of activations thingie 
        for l in range(len(args['nmapr_n_hid'])):
            last_layer = fc(last_layer, args['nmapr_n_hid'][l],
            activation_fn=string_to_nl(args['nmapr_nl'][l]))
    r_t = last_layer
    return r_t, memory

# TODO: remember memory here needs to be a tuple
def context_network(args, s_t, r_t, memory, old_c_t, extras, m0, ctx_state_tuple,write_type='lstm'):

    with tf.variable_scope('contextnet'):

        ctx_lstm = tf.contrib.rnn.BasicLSTMCell(args['memory_channels'],
            state_is_tuple=True,
            name='ctx_lstm')

        input_vec = [s_t]
        if args['use_position']:
            input_vec.append(extras['pos'])
        if args['use_orient']:
            input_vec.append(extras['orientation'])
        if args['use_velocity']:
            velocity = extras['pos'] - extras['p_pos']
            input_vec.append(tf.expand_dims(tf.cast(velocity, tf.float32), 0))
        if args['use_timestep']:
            timestep = tf.one_hot(
                    extras['timestep'],
                    args['max_timestep'],
                    on_value=1.0,
                    off_value=0.0)
            input_vec.append(timestep)

        input_vec.append(r_t)
        input_vec = tf.expand_dims(tf.concat(input_vec, 1), 0)
        
        # ctx_hx, ctx_cx = ctx_lstm(input_vec)
        # TODO: check if this is correct
        

        cont_hx, ctx_state_new = tf.nn.dynamic_rnn(ctx_lstm, input_vec,
                                        initial_state=ctx_state_tuple,
                                        dtype=tf.float32)

        map_scale = tf.constant(1.0)
        if args['egocentric']:
            map_scale *= 2.0

        rescale_func = None
        def rescale_max(p):
            return tf.cast(p, tf.float32)/(extras['max_maze_size'] * map_scale) * args['memory_size']
        def rescale_maze(p):
            return tf.cast(p, tf.float32)/(extras['maze_size'] * map_scale) * args['memory_size']
        if args['rescale_max']:
            # Rescale the position data to the maximum map size
            rescale_func = rescale_max
        else:
            # Rescale the position data to the current map size
            rescale_func = rescale_maze

        extras['scaled_pos'] = rescale_func(extras['pos'])
        extras['scaled_p_pos'] = rescale_func(extras['p_pos'])
        extras['scaled_velocity'] = extras['scaled_pos'] - extras['scaled_p_pos']

        # TODO: is do = po - l_po being used anywhere? why not?
        if args['egocentric']:
            vel = extras['scaled_velocity'] # vel = [dx, dy]
            # shift_memory = tf.tile(m0, [1, 1, args['memory_size'], args['memory_size']])
            srcboundy = (
                    tf.cast(tf.maximum(0.0, vel[0]), tf.int64), 
                    tf.cast(tf.minimum(float(args['memory_size']), float(args['memory_size']) + vel[0]), tf.int64)
                )
            srcboundx = (
                    tf.cast(tf.maximum(0.0, vel[1]), tf.int64), 
                    tf.cast(tf.minimum(float(args['memory_size']), float(args['memory_size']) + vel[1]), tf.int64)
                )
            dstboundy = (
                    tf.cast(tf.maximum(0.0,-vel[0]), tf.int64), 
                tf.cast(tf.minimum(float(args['memory_size']), float(args['memory_size']) - vel[0]), tf.int64)
            )
            dstboundx = (
                    tf.cast(tf.maximum(0.0,-vel[1]), tf.int64), 
                    tf.cast(tf.minimum(float(args['memory_size']), float(args['memory_size']) - vel[1]), tf.int64)
            )
            
            shift_memory = tf.Variable(
                0.01 * np.random.randn(1, args['memory_channels'], args['memory_size'], args['memory_size']),
                dtype=tf.float32)

            shift_memory[:, :, dstboundy[0]:dstboundy[1], dstboundx[0]:dstboundx[1]].assign(
                memory[:,:,srcboundy[0]:srcboundy[1],srcboundx[0]:srcboundx[1]])


        query = fc(cont_hx, args['memory_channels'], activation_fn=None) # C
        ctx_input = tf.concat([tf.squeeze(input_vec, 0), r_t, tf.squeeze(old_c_t, 0)], 1)


        # Context Read from memory
        mem = shift_memory[0]
        memory_matrix = tf.reshape(mem, [mem.shape[0], -1])
        context_scores = tf.matmul(tf.squeeze(query, 0), memory_matrix) # (1xC) x (CxWH) --> 1xWH
        context_prob = tf.nn.softmax(context_scores)
        context_prob_map = tf.reshape(context_prob, (mem.shape[1], mem.shape[2]))

        context_prob_map_expand = tf.expand_dims(
            tf.tile(tf.expand_dims(context_prob_map, 0),
            [args['memory_channels'],1,1]), 0)

        c_t = tf.reduce_sum(
            tf.reshape(shift_memory * context_prob_map_expand, (1,args['memory_channels'], -1)),
            axis=2
        )

    return c_t, cont_hx, shift_memory, ctx_state_new


def write_network(args, s_t, r_t, c_t, memory):
    with tf.variable_scope('writenet'):
        write_py, write_px = args['memory_size'] // 2, args['memory_size'] // 2
        write_output_size = args['memory_channels']
        old_write = memory[:,:,write_py,write_px]
        write_input = tf.expand_dims(tf.concat([s_t, r_t, c_t, old_write], axis=1), 0)
        write_update_gru = tf.contrib.rnn.GRUCell(write_output_size)
        
        w_t, state = tf.nn.dynamic_rnn(write_update_gru,
            write_input,
            initial_state=old_write,
            dtype=tf.float32,
        )

        # TODO: this will NOT work (slice assignment)
        memory[:,:,write_py,write_px].assign(w_t)
    
    return w_t, memory
