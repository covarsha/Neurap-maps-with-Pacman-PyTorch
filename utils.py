import math
import numpy as np

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
            shape=[None] + input_dims,
            name='inputs')
        last_layer = inputs
        for l in range(len(args['n_units'])):
            last_layer = conv2d(inputs=last_layer,
                num_outputs=args['n_units'][l],
                kernel_size=[args['filters'][l], args['filters'][l]], \
                stride=[args['strides'][l], args['strides'][l]],
                padding="same",
                activation=string_to_nl(args['nl'][l]))

        last_layer = flatten(last_layer)

        for l in range(len(args['n_hid'])):
            last_layer = fc(last_layer, args['n_hid'][l], activation_fn=string_to_nl(args['nl'][l]))
        print(last_layer)
        return last_layer, inputs


def read_network(args):
    with tf.variable_scope('readnet'):
        memory = tf.placeholder(tf.float32,
            shape=[None, args['memory_channels'],
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
                data_format="NHWC",#NCHW
                activation_fn=string_to_nl(args['nmapr_nl'][l]))
            print(last_layer)

        before_flatten = last_layer
        # last_layer = flatten(last_layer)
        after_flatten = flatten(last_layer)
        shape = last_layer.get_shape().as_list()
        dim = np.prod(shape[1:])
        last_layer = tf.reshape(last_layer, [-1, dim])
        print(last_layer)
        # last_layer = after_flatten
        # TODO: check sharing of activations thingie
        for l in range(len(args['nmapr_n_hid'])):
            last_layer = fc(last_layer, args['nmapr_n_hid'][l],
                activation_fn=string_to_nl(args['nmapr_nl'][l]))

    r_t = last_layer
    return r_t, memory, before_flatten, after_flatten

def context_network(args, s_t, r_t, memory, old_c_t, extras,ctx_state_tuple,write_type='lstm'):

    with tf.variable_scope('contextnet'):

        ctx_lstm = tf.contrib.rnn.BasicLSTMCell(args['memory_channels'],
            state_is_tuple=True,
            name='ctx_lstm')

        input_vec = [s_t]

        velocity = extras['pos'] - extras['p_pos']
        input_vec.append(tf.cast(velocity, tf.float32))

        timestep = tf.squeeze(tf.one_hot(
                extras['timestep'],
                args['max_timestep'],
                on_value=1.0,
                off_value=0.0), 1)
        input_vec.append(timestep)

        input_vec.append(r_t)
        input_vec = tf.expand_dims(tf.concat(input_vec, 1), 1)
        ####Add old context####### 
        print(input_vec)
        # ctx_hx, ctx_cx = ctx_lstm(input_vec)
        # TODO: check if this is correct
        cont_hx, ctx_state_new_tuple = tf.nn.dynamic_rnn(ctx_lstm, input_vec,
                                        initial_state=ctx_state_tuple,
                                        dtype=tf.float32)


        # TODO: move these to NeuralMap.shift_memory
        map_scale = tf.constant(1.0)
        if args['egocentric']:
            map_scale *= 2.0


        # TODO: is do = po - l_po being used anywhere? why not?

        query = fc(cont_hx, args['memory_channels'], activation_fn=None) # ?xC
        #ctx_input = tf.concat([tf.squeeze(input_vec, 0), r_t, tf.squeeze(old_c_t, 0)], 1)


        # Context Read from memory (?,C,H,W) --> (?,C,WH)
        memory_matrix = tf.reshape(memory, [-1, args['memory_channels'], args['memory_size'] * args['memory_size']])
        context_scores = tf.matmul(query, memory_matrix) # (?xC) x (?xCxWH) --> ?xWH

        print(context_scores)
        context_prob = tf.nn.softmax(context_scores)
        print(context_prob)
        context_prob_map = tf.reshape(context_prob, (-1, 1, args['memory_size'], args['memory_size']))

        print(context_prob_map)
        context_prob_map_expand = tf.tile(context_prob_map,
            [1,args['memory_channels'],1,1])

        print(context_prob_map_expand)
        print(memory * context_prob_map_expand)
        print('--------------')
        c_t = tf.reduce_sum(
            tf.reshape(memory * context_prob_map_expand, (-1,args['memory_channels'], args['memory_size'] * args['memory_size'])),
            axis=2
        )

    return c_t, cont_hx, ctx_state_new_tuple


def write_network(args, s_t, r_t, c_t, memory):
    with tf.variable_scope('writenet'):
        write_py, write_px = args['memory_size'] // 2, args['memory_size'] // 2
        write_output_size = args['memory_channels']
        old_write = memory[:,:,write_py,write_px]
        print([s_t, r_t, c_t, old_write])
        write_input = tf.expand_dims(tf.concat([s_t, r_t, c_t, old_write], axis=1), 1)
        print(write_input)
        write_update_gru = tf.contrib.rnn.GRUCell(write_output_size)

        print('write_input', write_input)
        print('initial_state',old_write)
        w_t, state = tf.nn.dynamic_rnn(write_update_gru,
            write_input,
            initial_state=old_write,
            dtype=tf.float32,
        )
        print('w_t', w_t)


    return w_t
