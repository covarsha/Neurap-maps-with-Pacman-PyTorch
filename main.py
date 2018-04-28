#!/usr/bin/env python3
import gym
import gym_pacman
import argparse
import sys
from baselines import logger,bench

import nmap_ppo as ppo
import multiprocessing
import tensorflow as tf
import numpy as np
import os
import os.path as osp
import pdb

from nmap import NeuralMapPolicy




def train(args, num_timesteps):

    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    tf.Session(config=config).__enter__()
    num_sub_in_grp = 4
    args['task']='BerkeleyPacmanPO-v0'
    # TODO: make it possible for multiple environments
    seed=0
    def make_env_vec(seed):
        def make_env():
            env = gym.make('BerkeleyPacmanPO-v0')
            #env.seed(seed)
            MONITORDIR = osp.join('savedir', 'monitor')
            if not osp.exists(MONITORDIR):
                os.makedirs(MONITORDIR)
            monitor_path = osp.join(MONITORDIR, '%s-%d'%(args['task'], seed))
            env = bench.Monitor(env, monitor_path, allow_early_resets=True)
            #env = gym.wrappers.Monitor(env, MONITORDIR, force=True, 
            #        video_callable=lambda episode_id: True)
            if 'Atari' in str(env.__dict__['env']):
                env = wrap_deepmind(env, frame_stack=True)
            return env
        return ppo.PacmanDummyVecEnv([make_env for _ in range(num_sub_in_grp)])

    envobj = make_env_vec(np.random.randint(0, 2**31-1))
    #env = gym.make('BerkeleyPacmanPO-v0')
    args['max_maze_size'] = envobj.envs[0].env.MAX_MAZE_SIZE
    args['maze_size'] = envobj.envs[0].env.maze_size
    # policy = NeuralMapPolicy(args, env, input_dims)
    print ('Reached')
    ppo.learn(env=envobj, nsteps=12, nminibatches=1,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=lambda f : f * 0.1,
        total_timesteps=int(num_timesteps * 1.1),
        nmap_args=args)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Neural Map Argument Parser')
    parser.add_argument('--nmapw_nl',dest='nmapw_nl',type=str, default='gru')
    parser.add_argument('--memory_size',dest='memory_size',type=int, default=30)
    parser.add_argument('--memory_channels',dest='memory_channels',type=int, default=256)
    parser.add_argument('--rescale_max',dest='rescale_max',type=bool, default=False)
    parser.add_argument('--egocentric',dest='egocentric',type=bool, default=True)
    parser.add_argument('--access_orient',dest='access_orient',type=bool, default=False)
    parser.add_argument('--erasure_on',dest='erasure_on',type=bool, default=False)
    parser.add_argument('--diffusion_on',dest='diffusion_on',type=bool, default=False)
    parser.add_argument('--use_position',dest='use_position',type=bool, default=False)
    parser.add_argument('--use_orient',dest='use_orient',type=bool, default=False)
    parser.add_argument('--use_velocity',dest='use_velocity',type=bool, default=True)
    parser.add_argument('--use_timestep',dest='use_timestep',type=bool, default=True)
    parser.add_argument('--max_timestep',dest='max_timestep',type=int, default=5) #Use 500 for testing
    #parser.add_argument('--nmapr_n_units',dest='nmapr_n_units',type=, default=)
    #parser.add_argument('--nmapr_filters',dest='nmapr_filters',type=, default=)
    #parser.add_argument('--nmapr_strides',dest='nmapr_strides',type=, default=)
    #parser.add_argument('--nmapr_padding',dest='nmapr_padding',type=, default=)
    #parser.add_argument('--nmapr_nl',dest='nmapr_nl',type=, default=)
    #parser.add_argument('--nmapr_n_hid',dest='nmapr_n_hid',type=, default=)
    #parser.add_argument('--n_units',dest='n_units',type=, default=)
    #parser.add_argument('--filters',dest='filters',type=, default=)
    #parser.add_argument('--strides',dest='strides',type=, default=)
    #parser.add_argument('--padding',dest='padding',type=, default=)
    #parser.add_argument('--nl',dest='nl',type=, default=)
    #parser.add_argument('--n_hid',dest='n_hid',type=, default=)
    #parser.add_argument('--input_dims',dest='input_dims',type=, default=)
    parser.add_argument('--rank',dest='rank',type=int, default=1)
    parser.add_argument('--output_size',dest='output_size',type=int, default=4)
    parser.add_argument('--env',dest='env',type=str, default='BerkeleyPacmanPO-v0')
    return parser.parse_args()




def main(args):
    seed = 1
    argment = parse_arguments()
    args={}

    args['nmapw_nl'] = argment.nmapw_nl
    args['memory_size'] = argment.memory_size
    args['memory_channels'] = argment.memory_channels
    args['rescale_max'] = argment.rescale_max
    args['egocentric'] = argment.egocentric
    args['access_orient'] = argment.access_orient
    args['erasure_on'] = argment.erasure_on
    args['diffusion_on'] = argment.diffusion_on
    args['use_position'] = argment.use_position
    args['use_orient'] = argment.use_orient
    args['use_velocity'] = argment.use_velocity
    args['use_timestep'] = argment.use_timestep
    args['max_timestep'] = argment.max_timestep
    args['nmapr_n_units'] = [8, 8, 8]
    args['nmapr_filters'] = [3, 3, 3]
    args['nmapr_strides'] = [1, 2, 2]
    args['nmapr_padding'] = [1, 0, 0]
    args['nmapr_nl'] = ['relu', 'relu', 'relu', 'relu', 'tanh']
    args['nmapr_n_hid'] = [256, 32]
    args['n_units'] = []
    args['filters'] = []
    args['strides'] = []
    args['padding'] = []
    args['nl'] = ['relu']
    args['n_hid'] = [128]
    args['seed'] = seed
    args['input_dims'] = [84,84,3]
    args['lr'] = 0.0001
    args['max_maze_size'] = (11,11)
    args['maze_size'] = (7,7)
    args['num_updates'] = 7500
    args['max_episode_length'] = 50
    args['gamma'] = 0.001
    args['tau'] = 0.001
    rank = argment.rank
    output_size = argment.output_size
    env = argment.env

    logger.configure()
    train(args, num_timesteps=1e5)

if __name__ == '__main__':
    main(sys.argv)

