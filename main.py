import random as rd
import argparse
from sys import exit
rd.seed(101)
import torch
from torch import nn
import time
import numpy as np
from tqdm import tqdm
import subprocess

from GraphEnc.encoder import getEncoder
from RecAgent.agent import getAgent
from utils import split_data

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='',
                        help= 'Dataset name')
    parser.add_argument('--root', type=str, default='datasets',
                        help= 'Datasets root directory')
    parser.add_argument('--sim_mode', type=str, default='stats',
                        help= 'Similarity mode for relevance score')
    
    # Graph encoder params
    parser.add_argument('--vis', nargs='?', default=-1,
                        help='we only want test value.')
    parser.add_argument('--test_only', nargs='?', default=False,
                        help='we only want test value.')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--enc_batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--enc_lr', type=float, default=5e-4,
                        help='Learning rate.')
    parser.add_argument('--regs', type=float, default=1e-5,
                        help='Regularization.')
    parser.add_argument('--epoch', type=int, default=1600,
                        help='Number of epoch.')
    parser.add_argument('--Ks', nargs='?', default= [20],
                        help='Evaluate on Ks optimal items.')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='log\'s interval epoch while training')
    parser.add_argument('--verbose', type=int, default=5,
                        help='Interval of evaluation.')
    parser.add_argument('--saveID', type=str, default="",
                        help='Specify model save path.')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping point.')
    parser.add_argument('--checkpoint', type=str, default='./',
                        help='Specify model save path.')
    parser.add_argument('--modeltype', type=str, default= 'BC_LOSS',
                        help='Specify model save path.')
    parser.add_argument('--cuda', type=int, default=0,
                        help='Specify which gpu to use.')
    parser.add_argument('--IPStype', type=str, default='cn',
                        help='Specify the mode of weighting')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of GCN layers')
    parser.add_argument('--codetype', type=str, default='train',
                        help='Calculate overlap with Item pop')
    parser.add_argument('--max2keep', type=int, default=10,
                        help='max checkpoints to keep')
    parser.add_argument('--infonce', type=int, default=1,
                        help='whether to use infonce loss or not')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers in data loader')
    parser.add_argument('--neg_sample', type=int, default=-1,
                        help='negative sample ratio.')    

    # MACR
    parser.add_argument('--alpha', type=float, default=1e-3,
                        help='alpha')
    parser.add_argument('--beta', type=float, default=1e-3,
                        help='beta')
    parser.add_argument('--c', type=float, default=30.0,
                        help='Constant c.')
    #CausE
    parser.add_argument('--cf_pen', type=float, default=0.05,
                        help='Imbalance loss.')
    
    #SAM-REG
    parser.add_argument('--rweight', type=float, default=0.05)
    parser.add_argument('--sam',type=bool,default=True)
    parser.add_argument('--pop_test',type=bool,default=False)

    #SimpleX
    parser.add_argument('--w_neg', type=float, default=1)
    parser.add_argument('--neg_margin',type=float, default=0.4)
    
    #BC_LOSS
    parser.add_argument('--tau1', type=float, default=0.07,
                        help='temperature parameter for L1')
    parser.add_argument('--tau2', type=float, default=0.1,
                        help='temperature parameter for L2')
    parser.add_argument('--w_lambda', type=float, default=0.5,
                        help='weight for combining l1 and l2.')
    parser.add_argument('--freeze_epoch',type=int,default=5)

    # RL Agent params
    parser.add_argument('--topk', type=int, default=10,
                        help= 'top-K recommendation')
    parser.add_argument('--obswindow', type=int, default=10,
                        help= 'Observe window. Older observations wont be considered for reward calculation')
    parser.add_argument('--agent_batch', type=int, default=64,
                        help= 'Batch size for agent training')
    parser.add_argument('--memory', type=int, default=20000,
                        help= 'Size of replay memory. Older transitions will be discarded')
    parser.add_argument('--replace_freq', type=int, default=99,
                        help= 'Frequency for updating DQN target network')
    parser.add_argument('--agent_lr', type=float, default=0.01,
                        help= 'Learning rate for agent training')
    parser.add_argument('--epsilon', type=float, default=0.95,
                        help= 'Epsilon greedy factor')
    parser.add_argument('--gamma', type=float, default=0.90,
                        help= 'Discount factor')
    parser.add_argument('--tau', type=float, default=0.01,
                        help= 'Fuse eval_net into target_net with temperature tau')
    parser.add_argument('--episode_max', type=int, default=100,
                        help= 'Maximum number of episode')
    parser.add_argument('--step_max', type=int, default=10000,
                        help= 'Number of DQN update iteration for each episode')
    # parser.add_argument('--j', type=int, default=16,
    #                     help= 'ThreadPoolExecutor max_workers')

    args = parser.parse_args()

    print('Preparing data ...')
    split_data(args)
    print('Setup finished!')
    print('####################')

    user_emb, item_emb, repr_user = None, None, None

    # Enable GCN embeddings
    if args.sim_mode != 'stats':

        # Graph encoder
        print('GCF training starts...')
        encoder, data = getEncoder(args)
        print('####################')

        # Get user, item embeddings
        user_emb = encoder.embed_user.weight.detach()
        item_emb = encoder.embed_item.weight.detach()

        # Build representative user embeddings for each item
        repr_user = []
        for item in range(data.n_items):
            ru_item = 0 # Representative user for item
            for user in data.train_item_list[item]:
                ru_item = ru_item + user_emb[user]
            ru_item = ru_item / len(data.train_item_list[item])
            repr_user.append(ru_item)
        
        repr_user = nn.Embedding.from_pretrained(repr_user)

    # Interactive RL Agent
    print('RL Agent training starts...')
    agent = getAgent(repr_user, item_emb, args)
    pass