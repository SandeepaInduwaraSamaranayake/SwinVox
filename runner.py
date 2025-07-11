#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
# Modified by Sandeepa Samaranayake <sandeepasamaranayake@outlook.com>

import logging
import matplotlib
import numpy as np
import os
import sys
# Fix problem: no $DISPLAY environment variable
matplotlib.use('Agg')

from argparse import ArgumentParser
from pprint import pprint

from config import cfg
from core.train import train_net
from core.test import test_net
from utils.lr_finder import find_lr
from utils.optuna_tune import optuna_tune


def get_args_from_command_line():
    parser = ArgumentParser(description='Parser of Runner of SwinVox')
    parser.add_argument('--gpu',
                        dest='gpu_id',
                        help='GPU device id to use [cuda0]',
                        default=cfg.CONST.DEVICE,
                        type=str)
    parser.add_argument('--rand', dest='randomize', help='Randomize (do not use a fixed seed)', action='store_true')
    parser.add_argument('--test', dest='test', help='Test neural networks', action='store_true')
    parser.add_argument('--batch-size',
                        dest='batch_size',
                        help='name of the net',
                        default=cfg.CONST.BATCH_SIZE,
                        type=int)
    parser.add_argument('--epoch', dest='epoch', help='number of epoches', default=cfg.TRAIN.NUM_EPOCHS, type=int)
    parser.add_argument('--weights', dest='weights', help='Initialize network from the weights file', default=None)
    parser.add_argument('--out', dest='out_path', help='Set output path', default=cfg.DIR.OUT_PATH)
    parser.add_argument('--lr_find', 
                        dest='lr_find', 
                        help='Run Learning Rate Finder', 
                        action='store_true')
    parser.add_argument('--optuna_tune', 
                        dest='optuna_tune', 
                        help='Run Optuna hyperparameter tuning', 
                        action='store_true')
    args = parser.parse_args()
    return args


def main():
    # Get args from command line
    args = get_args_from_command_line()

    if args.gpu_id is not None:
        cfg.CONST.DEVICE = args.gpu_id
    if not args.randomize:
        np.random.seed(cfg.CONST.RNG_SEED)
    if args.batch_size is not None:
        cfg.CONST.BATCH_SIZE = args.batch_size
    if args.epoch is not None:
        cfg.TRAIN.NUM_EPOCHS = args.epoch
    if args.out_path is not None:
        cfg.DIR.OUT_PATH = args.out_path
    if args.weights is not None:
        cfg.CONST.WEIGHTS = args.weights
        if not args.test:
            cfg.TRAIN.RESUME_TRAIN = True

    # Set GPU to use
    if type(cfg.CONST.DEVICE) == str:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE

    if args.lr_find:
        find_lr(cfg) # Call the LR finder function
        logging.info('Then run normal training without the --lr_find flag.')
        sys.exit(0) # Exit after finding LR, user needs to manually update config

    if args.optuna_tune:
        optuna_tune(cfg)
        logging.info('Then run normal training with updated config_best.py.')
        sys.exit(0)

    # Print config
    print('Use config:')
    pprint(cfg)

    # Start train/test process
    if not args.test:
        train_net(cfg)
    else:
        if 'WEIGHTS' in cfg.CONST and os.path.exists(cfg.CONST.WEIGHTS):
            test_net(cfg)
        else:
            logging.error('Please specify the file path of checkpoint.')
            sys.exit(2)


if __name__ == '__main__':
    # Check python version
    if sys.version_info < (3, 0):
        raise Exception("Please follow the installation instruction on 'https://github.com/SandeepaInduwaraSamaranayake/SwinVox'")

    # Setup logger
    logging.basicConfig(format='[%(levelname)s] %(asctime)s %(message)s', level=logging.INFO, force=True)

    main()
