import gym
import gym_pacman
import tensorflow as tf
import numpy as np
import argparse
import sys
from nmap import NeuralMapModel


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
    args['nl'] = []
    args['n_hid'] = []
    args['seed'] = seed
    input_dims = [84,84,3]
    args['lr'] = 0.0001
    args['num_updates'] = 7500
    args['max_episode_length'] = 50
    args['gamma'] = 0.001
    args['tau'] = 0.001
    rank = argment.rank
    output_size = argment.output_size
    env = argment.env

    env = gym.make('BerkeleyPacmanPO-v0')
    nmap_model = NeuralMapModel(args, env, input_dims)
    nmap_model.run()
    
if __name__ == '__main__':
    main(sys.argv)
