import os, sys
import argparse
import pprint
import yaml
import logging
import random, os
import numpy as np
import torch

def parse_options():

    parser = argparse.ArgumentParser(description='Machine Perception Code Skeleton')


    ###################
    # Global arguments
    ###################
    global_group = parser.add_argument_group('global')
    global_group.add_argument('--config', type=str, default='config.yaml', 
                               help='Path to config file to replace defaults')
    global_group.add_argument('--save-root', type=str, default='./checkpoints/', 
                               help="outputs path")
    global_group.add_argument('--exp-name', type=str, default='test',
                               help="Experiment name.")
    global_group.add_argument('--seed', type=int, default=123)
    global_group.add_argument(
        '--log_level', action='store', type=int, default=logging.INFO,
        help='Logging level to use globally, DEBUG: 10, INFO: 20, WARN: 30, ERROR: 40.')
        
    ###################
    # Arguments for dataset
    ###################
    data_group = parser.add_argument_group('dataset')
    data_group.add_argument('--data-root', type=str, default='data/public',
                            help='Path to dataset')
    data_group.add_argument('--num-rays-per-img', type=int, default=512,
                            help='Number of rays to sample per image')
    data_group.add_argument('--bg-color', default='white',
                            choices=['white', 'black'],
                            help='Background color')

    ###################
    # Arguments for optimizer
    ###################
    optim_group = parser.add_argument_group('optimizer')
    optim_group.add_argument('--lr', type=float, default=0.001, 
                             help='Learning rate.')
    optim_group.add_argument('--beta1', type=float, default=0.5,
                                help='Beta1.')
    optim_group.add_argument('--beta2', type=float, default=0.999,
                                help='Beta2.')
    optim_group.add_argument('--weight-decay', type=float, default=0, 
                             help='Weight decay.')


    ###################
    # Arguments for training
    ###################
    train_group = parser.add_argument_group('train')
    train_group.add_argument('--is_continue', action='store_true', default='store_false')
    train_group.add_argument('--checkpoint', type=str, default='')
    train_group.add_argument('--epochs', type=int, default=800, 
                             help='Number of epochs to run the training.')
    train_group.add_argument('--batch-size', type=int, default=2, 
                             help='Batch size for the training.')
    train_group.add_argument('--workers', type=int, default=0,
                             help='Number of workers for the data loader. 0 means single process.')
    train_group.add_argument('--save-every', type=int, default=50, 
                             help='Save the model at every N epoch.')
    train_group.add_argument('--log-every', type=int, default=100,
                             help='write logs to wandb at every N iters')
    train_group.add_argument('--mask_weight', type=float, default=0.1)
    train_group.add_argument('--igr_weight', type=float, default=0.1)

    ###################
    # Arguments for Point Sampling
    ###################
    sample_group = parser.add_argument_group('sampling')
    sample_group.add_argument('--num-pts-per-ray', type=int, default=128,
                                help='Number of points to sample per ray')
    sample_group.add_argument('--near', type=float, default=1.0,
                                help='Near plane')
    sample_group.add_argument('--far', type=float, default=3.0,
                                help='Far plane')
    
    ###################
    # Arguments for model
    ###################
    model = parser.add_argument_group('model')
    model.add_argument('--n_samples', type=int, default=64)
    model.add_argument('--n_importance', type=int, default=64)
    model.add_argument('--n_outside', type=int, default=0)
    model.add_argument('--up_sample_steps', type=int, default=4)
    model.add_argument('--perturb', type=int, default=2)
    model.add_argument('--renderer', type=dict, default=dict())
    model.add_argument('--sdf_net', type=dict, default=dict())
    model.add_argument('--var_net', type=dict, default=dict())
    model.add_argument('--color_net', type=dict, default=dict())

   ###################
    # Arguments for validation
    ###################
    valid_group = parser.add_argument_group('validation')
    valid_group.add_argument('--valid-every', type=int, default=10,
                             help='Frequency of running validation.')
    valid_group.add_argument('--mip', type=int, default=2, 
                            help='downsample factor for the validation set')
    valid_group.add_argument('--sdf_thres', type=float, default=0.0)
    valid_group.add_argument('---chunk-size', type=int, default=10240, 
                            help='max number of points to process at once during validation')
    valid_group.add_argument('--num-pts-per-ray-render', type=int, default=128,
                                help='Number of points to sample per ray during validation')
    valid_group.add_argument('--save-img', action='store_true',
                                help='Save rendered images during validation')
    ###################
    # Arguments for wandb
    ###################
    wandb_group = parser.add_argument_group('wandb')
    
    wandb_group.add_argument('--wandb-id', type=str, default=None,
                             help='wandb id')
    wandb_group.add_argument('--wandb', action='store_true',
                             help='Use wandb')
    wandb_group.add_argument('--wandb-name', default='default', type=str,
                             help='wandb_name')

    return parser


def parse_yaml_config(config_path, parser):
    """Parses and sets the parser defaults with a yaml config file.

    Args:
        config_path : path to the yaml config file.
        parser : The parser for which the defaults will be set.
        parent : True if parsing the parent yaml. Should never be set to True by the user.
    """
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    list_of_valid_fields = []
    for group in parser._action_groups:
        group_dict = {list_of_valid_fields.append(a.dest) for a in group._group_actions}
    list_of_valid_fields = set(list_of_valid_fields)
    
    defaults_dict = {}

    # Loads child parent and overwrite the parent configs
    # The yaml files assumes the argument groups, which aren't actually nested.
    for key in config_dict:
        for field in config_dict[key]:
            if field not in list_of_valid_fields:
                raise ValueError(
                    f"ERROR: {field} is not a valid option. Check for typos in the config."
                )
            defaults_dict[field] = config_dict[key][field]


    parser.set_defaults(**defaults_dict)

def argparse_to_str(parser, args=None):
    """Convert parser to string representation for Tensorboard logging.

    Args:
        parser (argparse.parser): Parser object. Needed for the argument groups.
        args : The parsed arguments. Will compute from the parser if None.
    
    Returns:
        args    : The parsed arguments.
        arg_str : The string to be printed.
    """
    
    if args is None:
        args = parser.parse_args()

    if args.config is not None:
        parse_yaml_config(args.config, parser)

    args = parser.parse_args()

    args_dict = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        args_dict[group.title] = vars(argparse.Namespace(**group_dict))

    pp = pprint.PrettyPrinter(indent=2)
    args_str = pp.pformat(args_dict)
    args_str = f'```{args_str}```'

    return args, args_str

#####################################################################################
#####################################################################################

def seed_everything(seed: int):
    """
    EREN: Fixing seeds for every library, I think their implementation 
    with 3 seeds does not allow reproducibility properly
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False