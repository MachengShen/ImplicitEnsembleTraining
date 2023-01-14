import os
import argparse
import importlib

parser = argparse.ArgumentParser()
parser.add_argument('-env', type=str, choices=['simple_tag',
                                               'simple_world_comm',
                                               'simple_crypto',
                                               'simple',
                                               'simple_push',
                                               'simple_spread',
                                               'tennis',
                                               'foozpong',
                                               'pong',
                                               'leduc_holdem',
                                               'texas_holdem',
                                               'connect_four',
                                               'basketball_pong',
                                               'volleyball_pong',
                                               'tictactoe',
                                               'starcraft',
                                               'derk',
                                               'ant',
                                               'cheetah',
                                               'hopper',
                                               ],
                    help='env_id')
parser.add_argument('-training_setting', type=str, choices=['single_policy',
                                                             'simple_ensemble',
                                                             'implicit_ensemble'],
                    help='train setting')
parser.add_argument('-logdir', type=str,
                    help='logging directory')
parser.add_argument('--fix_latent', action='store_true',
                    help='fix latent variable in IET')
parser.add_argument('--ensemble_size', type=int, default=3,
                    help='ensemble size in simple ensemble training')
parser.add_argument('--main_ckpt', type=str,
                    help='main checkpoint path to restore')
parser.add_argument('--test_ckpt', type=str,
                    help='main checkpoint path to restore')
parser.add_argument('--no_render', action='store_true',
                    help='main checkpoint path to restore')
parser.add_argument('--num_cpus', type=int, default=6,
                    help='number of cpu cores to run the code')
parser.add_argument('--train_step_multiplier', type=int, default=6000,
                    help='parameter determining the total steps for training')

def process_cmd_args(args):
    if args.env.startswith('simple'):
        args.env = 'pettingzoo.mpe.' + args.env + '_v2'
    if args.env == 'tennis':
        args.env = 'pettingzoo.atari.' + args.env + '_v2'
    if args.env in ['pong', 'foozpong', 'basketball_pong', 'volleyball_pong']:
        args.env = 'pettingzoo.atari.' + args.env + '_v1'
    if args.env in ['leduc_holdem',
                    'connect_four',
                    'texas_holdem',
                    'texas_holdem_no_limit',
                    'tictactoe']:
        args.env = 'pettingzoo.classic.' + args.env + '_v2'
    if args.env == 'starcraft':
        args.env = 'implicitPBT_module.env_wrapper.starcraft_competitive_pettingzoo_wrapper'
    if args.env == 'derk':
        args.env = 'gym_derk.envs'
    if args.env == 'ant':
        args.env = 'implicitPBT_module.env_wrapper.roboschool_ant_racer_rllib_wrapper'
    if args.env == 'cheetah':
        args.env = 'implicitPBT_module.env_wrapper.roboschool_cheetah_racer_rllib_wrapper'
    if args.env == 'hopper':
        args.env = 'implicitPBT_module.env_wrapper.roboschool_hopper_racer_rllib_wrapper'
    args.env = importlib.import_module(args.env)
    if args.main_ckpt and args.main_ckpt.startswith('~/'):
        args.main_ckpt = os.path.expanduser(args.main_ckpt)
    if args.test_ckpt and args.test_ckpt.startswith('~/'):
        args.test_ckpt = os.path.expanduser(args.test_ckpt)
    return args

cmd_args = parser.parse_args()
cmd_args = process_cmd_args(cmd_args)