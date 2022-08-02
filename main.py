"""
   MGA-bot: main.py
   Command-line Interface for MTTOD.
   Copyright 2022 SPMI, Yucheng Cai
"""

import random
import argparse
import os
import logging

import torch
import numpy as np

from config import global_config as cfg
from model import Model
from utils.utils import parse_arg_cfg
from inference import validate_fast

def main(cfg):
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()
    cfg = parse_arg_cfg(cfg,args)

    cfg.mode = args.mode
    if 'test' in args.mode:
        cfg.eval_load_path=cfg.model_path
    else:  # train
        if cfg.exp_path in ['', 'to be generated']:
            experiments_path = './models' 
            if cfg.exp_no == '':
                cfg.exp_no = cfg.exp_no + cfg.mode + '_'
            if cfg.only_dst:
                cfg.exp_no = cfg.exp_no + 'dst_'
            print('exp_no:',cfg.exp_no)
            cfg.exp_path = os.path.join(experiments_path,'{}_{}_sd{}_lr{}_bs{}_ga{}'.format('-'.join(cfg.exp_domains),
                                                                          cfg.exp_no, cfg.seed, cfg.lr, cfg.batch_size,
                                                                          cfg.gradient_accumulation_steps))
            if cfg.posterior_train:
                cfg.exp_path = os.path.join(cfg.exp_path,'pos_model')
            if 'test' not in cfg.mode:
                print('save path:', cfg.exp_path)
            if cfg.save_log:
                if not os.path.exists(cfg.exp_path):
                    os.mkdir(cfg.exp_path)

            cfg.eval_load_path = cfg.exp_path

    cfg._init_logging_handler(args.mode) #to be modified in future
    
    # fix random seed, can be also set in other files 
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    m = Model(cfg)
    m.reader._load_data()
    if args.mode =='pretrain' or args.mode=='train':
        m.train()
    elif args.mode =='VL':
        pass
    elif args.mode =='jsa':
        pass
    elif args.mode == 'ST':
        pass
    elif (args.mode =='test_flow') or (args.mode =='test_gui'):#for future settings
        pass
    elif args.mode =='test_pos':
        pass
    elif args.mode =='test':  
        logging.info('Load model from :{}'.format(cfg.eval_load_path))
        validate_fast(m,data='test')

if __name__ == "__main__":
    main(cfg)
