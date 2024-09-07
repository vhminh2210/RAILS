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
    parser.add_argument('--nov_beta', type=float, default=0.4,
                        help= 'novelty temperature')
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
    parser.add_argument('--eta', type=float, default=0.90,
                        help= 'Forgetting factor')
    parser.add_argument('--tau', type=float, default=0.01,
                        help= 'Fuse eval_net into target_net with temperature tau')
    parser.add_argument('--episode_max', type=int, default=100,
                        help= 'Maximum number of episode')
    parser.add_argument('--step_max', type=int, default=10000,
                        help= 'Number of DQN update iteration for each episode')
    parser.add_argument('--dqn_mode', type=str, default='vanilla',
                        help= 'DQN update mode: vanilla/ddqn')
    parser.add_argument('--dueling_dqn', action='store_true', default=False,
                        help= 'Enable Dueling DQN')
    parser.add_argument('--j', type=int, default=8,
                        help= 'ThreadPoolExecutor max_workers')
    parser.add_argument('--cql_mode', type=str, default='none',
                        help= 'Conservative Q learning mode: none, cql_H, cql_Rho')
    parser.add_argument('--cql_alpha', type=float, default=2.0,
                        help= 'Conservative Q learning weight')
    parser.add_argument('--cql_invZ', type=float, default=1.0,
                        help= 'Inverse normalization factor Z for CQL(rho)')
    parser.add_argument('--seq_ratio', type=float, default=0.2,
                        help= 'Sequential partition ratio')
    parser.add_argument('--rare_ratio', type=float, default=0.3,
                        help= 'Rare-action partition ratio')
    parser.add_argument('--rand_ratio', type=float, default=0.5,
                        help= 'Random partition ratio')
    parser.add_argument('--rare_thresh', type=float, default=0.3,
                        help= 'Rarity threshold')
    parser.add_argument('--n_augment', type=int, default=5,
                        help= 'Number of augmented transition generated per existed transition')

    args = parser.parse_args()
    assert args.epoch >= args.freeze_epoch

    start_time = time.time()
    print('####################')
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
        user_emb, item_emb = encoder.compute()

        user_emb, item_emb = user_emb.detach(), item_emb.detach()

        # Normalize user embeddings
        for i in range(user_emb.shape[0]):
            user_emb[i] = user_emb[i] / torch.linalg.norm(user_emb[i])
        print('User embeddings shape:', user_emb.shape)

        # Normalize item embeddings
        for i in range(item_emb.shape[0]):
            item_emb[i] = item_emb[i] / torch.linalg.norm(item_emb[i])
        print('Item embeddings shape:', item_emb.shape)

        # Build representative user embeddings for each item
        repr_user = []
        for item in range(data.n_items):
            ru_item = 0 # Representative user for item
            for user in data.train_item_list[item]:
                ru_item = ru_item + user_emb[user] / torch.linalg.norm(user_emb[user])
            ru_item = ru_item / len(data.train_item_list[item])
            # Normalize representatives embedding
            # ru_item = ru_item / torch.linalg.norm(ru_item)
            repr_user.append(ru_item)
        repr_user = torch.stack(repr_user, axis= 0)
        print('Representative user embeddings shape:', repr_user.shape)
        
        repr_user = nn.Embedding.from_pretrained(repr_user)
        item_emb = nn.Embedding.from_pretrained(item_emb)

        print(np.linalg.norm(repr_user.weight.detach()[1]))
        print(np.linalg.norm(item_emb.weight.detach()[1]))

    # Interactive RL Agent
    print('RL Agent training starts...')
    agent = getAgent(repr_user, item_emb, args)
    print('####################')
    print('Runtime:', time.time() - start_time, 'seconds')