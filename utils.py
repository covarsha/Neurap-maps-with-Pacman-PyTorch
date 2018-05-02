import tensorflow as tf
import numpy as np

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

def basenet(args, inputs):
    with tf.variable_scope('statenet'):
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
        return last_layer

def read_network(args, memory, reuse=False):
    with tf.variable_scope('readnet', reuse=reuse):
        last_layer = memory

        for l in range(len(args['nmapr_n_units'])):
            last_layer = conv2d(inputs=last_layer,
                num_outputs=args['nmapr_n_units'][l],
                kernel_size=[args['nmapr_filters'][l], args['nmapr_filters'][l]], \
                stride=[args['nmapr_strides'][l], args['nmapr_strides'][l]],
                padding="same",
                data_format="NCHW",
                activation_fn=string_to_nl(args['nmapr_nl'][l]))

        before_flatten = last_layer
        # last_layer = flatten(last_layer)
        after_flatten = flatten(last_layer)
        shape = last_layer.get_shape().as_list()
        dim = np.prod(shape[1:])
        last_layer = tf.reshape(last_layer, [-1, dim])
        # last_layer = after_flatten
        # TODO: check sharing of activations thingie
        for l in range(len(args['nmapr_n_hid'])):
            last_layer = fc(last_layer, args['nmapr_n_hid'][l],
                activation_fn=string_to_nl(args['nmapr_nl'][l]))
    r_t = last_layer
    return r_t

def context_network(args, s_t, r_t, memory, old_c_t, velocity, timestep, ctx_state_tuple,write_type='lstm', reuse=False):

    with tf.variable_scope('contextnet', reuse=reuse):

        ctx_lstm = tf.contrib.rnn.BasicLSTMCell(args['memory_channels'],
            state_is_tuple=True,
            name='ctx_lstm')

        input_vec = [s_t]

        input_vec.append(tf.cast(velocity, tf.float32))

        timestep = tf.squeeze(tf.one_hot(
                timestep,
                args['max_timestep'],
                on_value=1.0,
                off_value=0.0), 1)
        input_vec.append(timestep)

        input_vec.append(r_t)
        input_vec.append(old_c_t)
        input_vec = tf.expand_dims(tf.concat(input_vec, 1), 1)

        cont_hx, ctx_state_new_tuple = tf.nn.dynamic_rnn(ctx_lstm, input_vec,
                                        initial_state=ctx_state_tuple,
                                        dtype=tf.float32)


        # TODO: move these to NeuralMap.shift_memory
        map_scale = tf.constant(1.0)
        if args['egocentric']:
            map_scale *= 2.0

        # TODO: is do = po - l_po being used anywhere? why not?

        query = fc(cont_hx, args['memory_channels'], activation_fn=None) # ?xC


        # Context Read from memory (?,C,H,W) --> (?,C,WH)
        memory_matrix = tf.reshape(memory, [-1, args['memory_channels'], args['memory_size'] * args['memory_size']])
        context_scores = tf.matmul(query, memory_matrix) # (?xC) x (?xCxWH) --> ?xWH

        context_prob = tf.nn.softmax(context_scores)
        context_prob_map = tf.reshape(context_prob, (-1, 1, args['memory_size'], args['memory_size']))

        context_prob_map_expand = tf.tile(context_prob_map,
            [1,args['memory_channels'],1,1])

        c_t = tf.reduce_sum(
            tf.reshape(memory * context_prob_map_expand, (-1,args['memory_channels'], args['memory_size'] * args['memory_size'])),
            axis=2
        )

    return c_t, cont_hx, ctx_state_new_tuple

def write_network(args, s_t, r_t, c_t, memory, write_loc, nenv, reuse=False):
    with tf.variable_scope('writenet', reuse=reuse):
        write_py, write_px = memory.shape[2] // 2, memory.shape[3] // 2
        write_output_size = args['memory_channels']
        old_write = memory[:,:,write_py,write_px]
        write_input = tf.expand_dims(tf.concat([s_t, r_t, c_t, old_write], axis=1), 1)
        write_update_gru = tf.contrib.rnn.GRUCell(write_output_size)

        shape = tf.constant([nenv, args['memory_channels'], args['memory_size'], args['memory_size']])

        w_t, state = tf.nn.dynamic_rnn(write_update_gru,
            write_input,
            initial_state=old_write,
            dtype=tf.float32,
        )
        memory = memory - tf.scatter_nd(write_loc, tf.gather_nd(memory, write_loc), shape)
        memory = memory + tf.scatter_nd(write_loc, tf.reshape(w_t, [nenv * args['memory_channels']]), shape)
    return w_t, memory

def batch_to_seq(h, nbatch, nsteps, flat=False):
    if flat:
        h = tf.reshape(h, [nbatch, nsteps])
    else:
        h = tf.reshape(h, [nbatch, nsteps, -1])
    return [tf.squeeze(v, [1]) for v in tf.split(axis=1, num_or_size_splits=nsteps, value=h)]

def seq_to_batch(h, flat = False):
    shape = h[0].get_shape().as_list()
    if not flat:
        assert(len(shape) > 1)
        nh = h[0].get_shape()[-1].value
        return tf.reshape(tf.concat(axis=1, values=h), [-1, nh])
    else:
        return tf.reshape(tf.stack(values=h, axis=1), [-1])

def get_model(args, nbatch, nsteps, inputs, memory, c_t, ctx_state_tuple,
                    pos, p_pos, timestep, masks, reuse=False):

    nenv = nbatch // nsteps

    write_loc = np.zeros((nenv * args['memory_channels'], 4), dtype=np.int32)
    write_loc = np.zeros((nenv * args['memory_channels'], 4), dtype=np.int32)

    for e in range(nenv):
        for c in range(args['memory_channels']):
            write_loc[args['memory_channels'] * e + c, 0 ] = e
            write_loc[args['memory_channels'] * e + c, 1 ] = c
            write_loc[args['memory_channels'] * e + c, 2: ] = args['memory_size'] // 2
    write_loc = tf.constant(write_loc)

    feats = []
    with tf.variable_scope("model", reuse=reuse):
        # nbatch x ?
        statenet_out = basenet(args, inputs)
        velocity = pos - p_pos


        s_ts = batch_to_seq(statenet_out, nenv, nsteps)
        vels = batch_to_seq(velocity, nenv, nsteps)
        timesteps = batch_to_seq(timestep, nenv, nsteps)
        masks = batch_to_seq(masks, nenv, nsteps)

        sizes = tf.constant(np.tile([[nenv, args['memory_channels'], args['memory_size'], args['memory_size']]], [nenv, 1]))

        i = 0
        for s_t, vel, tm, m in zip(s_ts, vels, timesteps, masks):
            net_reuse = i > 0
            shifted_memories = []
            vel = tf.cast(vel, tf.int32)
            for m_ix in range(nenv):
                shifted_memories.append(tf.expand_dims(
                    tf.slice(memory[m_ix, :, :, :] * (1 - m[m_ix]),
                             [0, vel[m_ix,0] + 1,vel[m_ix,1]],
                             [args['memory_channels'], args['memory_size'], args['memory_size']]
                        ), 0))
            memory = tf.concat(shifted_memories, axis=0, name='shifted_memory')
            r_t = read_network(args, memory, reuse=net_reuse)

            # mask c_t
            c_t = c_t * (1 - m)
            ctx_state_tuple = tf.nn.rnn_cell.LSTMStateTuple(ctx_state_tuple[0] * (1 - m), ctx_state_tuple[1] * (1 - m))

            c_t, cont_hx, ctx_state_tuple = context_network(args, s_t, r_t, memory, c_t, vel, tm, ctx_state_tuple, reuse=net_reuse)
            w_t, memory = write_network(args, s_t, r_t, c_t, memory, write_loc, nenv, reuse=net_reuse)
            with tf.variable_scope('feats_network', reuse=net_reuse):
                f_t = fc(
                    tf.concat([tf.squeeze(cont_hx,1), c_t, tf.squeeze(w_t, 1)], 1),
                        args['memory_channels'],
                        activation_fn=tf.nn.elu)
                feats.append(f_t)
            i += 1
        feats = seq_to_batch(feats)
    return memory, c_t, ctx_state_tuple, feats
