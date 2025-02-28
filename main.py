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

import os

from GraphEnc.encoder import getEncoder
from RecAgent.agent import getAgent
from utils import split_data, get_minmax_freq, crossrec_prep

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='',
                        help= 'Dataset name')
    parser.add_argument('--root', type=str, default='datasets',
                        help= 'Datasets root directory')
    parser.add_argument('--sim_mode', type=str, default='stats',
                        help= 'Similarity mode for relevance score')
    parser.add_argument('--user_lam', type=float, default=1.,
                        help= 'User weights for representatives')
    parser.add_argument('--pretrained_graph', action='store_true', default=False,
                        help= 'Use pretrained graph')
    parser.add_argument('--ckpt_dir', type=str, default='weights',
                        help= 'Checkpoint directory')
    parser.add_argument('--ckpt', type=str, default='',
                        help= 'Checkpoint name')
    
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
    parser.add_argument('--epsilon', type=float, default=0.7,
                        help= 'Epsilon exploration factor')
    parser.add_argument('--n_proposal', type=int, default=500,
                        help= 'Number of proposal items')
    parser.add_argument('--gamma', type=float, default=0.90,
                        help= 'Discount factor')
    parser.add_argument('--eta', type=float, default=1.0,
                        help= 'Forgetting factor')
    parser.add_argument('--tau', type=float, default=0.01,
                        help= 'Fuse eval_net into target_net with temperature tau')
    parser.add_argument('--episode_max', type=int, default=100,
                        help= 'Maximum number of episode')
    parser.add_argument('--step_max', type=int, default=10000,
                        help= 'Number of DQN update iteration for each episode')
    parser.add_argument('--epoch_max', type=int, default=10,
                        help= 'Maximum number of epoch')
    parser.add_argument('--dqn_mode', type=str, default='vanilla',
                        help= 'DQN update mode: vanilla/ddqn')
    parser.add_argument('--dueling_dqn', action='store_true', default=False,
                        help= 'Enable Dueling DQN')
    parser.add_argument('--noisy_net', action='store_true', default=False,
                        help= 'Enable NoisyNet for DQN')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help= 'Dropout rate')
    parser.add_argument('--j', type=int, default=8,
                        help= 'ThreadPoolExecutor max_workers')
    parser.add_argument('--cql_mode', type=str, default='none',
                        help= 'Conservative Q learning mode: none, cql_H, cql_Rho')
    parser.add_argument('--cql_alpha', type=float, default=2.0,
                        help= 'Conservative Q learning weight')
    parser.add_argument('--cql_invZ', type=float, default=1.0,
                        help= 'Inverse normalization factor Z for CQL(rho)')
    parser.add_argument('--seq_ratio', type=float, default=0.3,
                        help= 'Sequential partition ratio')
    parser.add_argument('--rare_ratio', type=float, default=0.2,
                        help= 'Rare-action partition ratio')
    parser.add_argument('--rand_ratio', type=float, default=0.5,
                        help= 'Random partition ratio')
    parser.add_argument('--rare_thresh', type=float, default=0.1,
                        help= 'Rarity threshold')
    parser.add_argument('--n_augment', type=int, default=5,
                        help= 'Number of augmented transition generated per existed transition')
    parser.add_argument('--n_aug_scale', type=int, default=-1,
                        help= 'Number of sample scales for augmentation. -1 for dense augmentation')
    parser.add_argument('--eval_freq', type=int, default=1,
                        help= 'Frequency of RL Agent evaluation')
    parser.add_argument('--min_obs', type=int, default=2,
                        help= 'Exclude all users with less than `min_obs` training interaction ')
    parser.add_argument('--policy', type=str, default='max',
                        help= 'Policy mode: max, stochastic or gradient')
    parser.add_argument('--all_episodes', action='store_true', default=False,
                        help= 'Enable training on full trainset')
    parser.add_argument('--eval_graph', action='store_true', default=False,
                        help= 'Enable evaluation on trained encoder')
    parser.add_argument('--action_proposal', action='store_true', default=False,
                        help= 'Enable action proposal. Deprecated for CrossRec')
    parser.add_argument('--episode_batch', type=int, default=1,
                        help= 'Number of episode per evaluation batch / Learn frequency')
    parser.add_argument('--num_hidden', type=int, default=256,
                        help= 'Number of hidden activations for DQN.')
    parser.add_argument('--eval_query', action='store_true', default=False,
                        help= 'Enable query evaluation mode')
    parser.add_argument('--eval_coldstart', action='store_true', default=False,
                        help= 'Enable cold start user evaluation mode')
    parser.add_argument('--crossrec_topsim', type=int, default=10,
                        help= 'Number of CrossRec neighbors user. Default to k=10')

    args = parser.parse_args()
    assert args.epoch >= args.freeze_epoch
    assert args.user_lam <= 1.

    if args.eval_query:
        assert 'query.txt' in os.listdir(os.path.join(args.root, args.dataset))

    if args.cuda < 0:
        device = 'cpu'
    else:
        device = f'cuda:{args.cuda}'
    args.device = device

    if args.episode_batch * (args.n_aug_scale + 1) * args.n_augment > args.memory * args.seq_ratio:
        print('WARNING: Current configurations may cause skips on sequential partition!')
        print('Please modify episode_batch, n_aug_scale, n_augment if needed!')

    # if args.num_gpu > 0:
    #     try:
    #         assert args.cuda >= 0
    #     except:
    #         raise ValueError('One CUDA device must be specified for --cuda regarding --num_gpu > 0')

    start_time = time.time()
    print('####################')
    print('Device used:', device)
    print('Preparing data ...')
    split_data(args)
    print('Setup finished!')
    print('####################')

    user_emb, item_emb, repr_user = None, None, None
    crossrec_bundle = None

    # Enable GCN embeddings
    if args.sim_mode in ['user_embedding', 'item_embedding']:

        # Graph encoder
        print('GCF training starts...')
        encoder, data = getEncoder(args)
        freq, min_freq, max_freq = get_minmax_freq(os.path.join(args.root, args.dataset, 'train.txt'), data.n_items)
        print(f'Min frequency: {min_freq}. Max frequency: {max_freq}')
        print('####################')

        # Get user, item embeddings
        user_emb, item_emb = encoder.compute()

        user_emb, item_emb = user_emb.detach().cpu(), item_emb.detach().cpu()

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
        wild_items = []
        for item in range(data.n_items):
            ru_item = torch.zeros_like(user_emb[0]) # Representative user for item

            if item in data.train_item_list.keys():
                attn_sum = 0
                for user in data.train_item_list[item]:
                    attn = torch.dot(user_emb[user], item_emb[item])
                    attn_sum += attn
                    ru_item = ru_item + attn * user_emb[user]
                # ru_item = ru_item / len(data.train_item_list[item]) # Normalized by item popularity
                ru_item = ru_item / attn_sum
            else:
                # Wild item found!
                wild_items.append(item)

            repr_user.append(ru_item)

        wild_items = torch.tensor(wild_items).int()
        repr_user = torch.stack(repr_user, axis= 0) # n_item, embedding_dim
        # repr_user[wild_items] = torch.mean(repr_user, dim= 0)
        item_emb[wild_items] = torch.zeros_like(item_emb[0])
        print(f'{wild_items.shape[0]} wild items found!')
        print('Representative user embeddings shape:', repr_user.shape)

        repr_user = args.user_lam * repr_user + (1. - args.user_lam) * item_emb
        
        repr_user = nn.Embedding.from_pretrained(repr_user)
        item_emb = nn.Embedding.from_pretrained(item_emb)
        user_emb = nn.Embedding.from_pretrained(user_emb)

        print(torch.linalg.norm(repr_user.weight.detach()[1]))
        print(torch.linalg.norm(item_emb.weight.detach()[1]))
        print(torch.linalg.norm(user_emb.weight.detach()[1]))

        args.n_users = user_emb.weight.shape[0]
        args.n_items = item_emb.weight.shape[0]
        args.wild_items = wild_items.tolist()

        print('nUsers:', args.n_users, 'nItems:', args.n_items)
        print('Wild items:', sorted(args.wild_items))

    elif args.sim_mode == 'crossrec':
        data = getEncoder(args, data_only= True)
        freq, min_freq, max_freq = get_minmax_freq(os.path.join(args.root, args.dataset, 'train.txt'), data.n_items)

        # Handling wild items
        wild_items = []
        for item in range(data.n_items):
            if item not in data.train_item_list.keys():
                wild_items.append(item)
        
        # Prepare crossrec bundle
        tfidf_item, tfidf_user, r_bar = crossrec_prep(data)
        crossrec_bundle = {
            'tfidf_item' : tfidf_item,
            'tfidf_user' : tfidf_user,
            'mean_rating' : r_bar,
            'item2user' : data.train_item_list
        }

        args.crossrec_bundle = crossrec_bundle

        args.n_users = data.n_users
        args.n_items = data.n_items
        args.wild_items = wild_items

    else:
        raise NotImplementedError('ERROR: `stats` similarity mode is deprecated!')
    
    if args.all_episodes:
        args.episode_max = args.n_users

    # Interactive RL Agent
    agent = getAgent(repr_user, user_emb, item_emb, args.wild_items, min_freq, max_freq, freq, args)
    print('####################')
    print('Runtime:', time.time() - start_time, 'seconds')