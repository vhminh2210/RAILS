import argparse
import os
import pandas as pd

from .train import train_dqn
from .util.datasplit_util import data_split
from .util.jsondict_util import load_dict
from .util.popularity_util import item_popularity_generate
from .util.quality_util import item_quality_generate
from .util.simmatrix_util import sim_matrix_generate

def getAgent(repr_user, item_emb, args):
    train_df = None
    test_df = None
    item_sim_dict = None
    item_quality_dict = None
    item_pop_dict = None
    max_item_id = None
    item_list = None
    mask_list = None

    # Load data
    dataset = args.dataset
    dat_dir = os.path.join(args.root, args.dataset)
    dat_path = os.path.join(dat_dir, f'{args.dataset}.dat')
    if os.path.exists(dat_path):
        df = pd.read_csv(dat_path, sep=',',
                         names=['user_id', 'item_id', 'ratings', 'timestamp'])
        max_item_id = df['item_id'].max()
        item_list = df['item_id'].tolist()
        mask_list = list(set(list(range(max_item_id))) - set(item_list))
        mat_path = os.path.join(dat_dir, f'{args.dataset}.mat')

        # Generate pre-calculated similarity, popularity and quality matrix
        if os.path.exists(mat_path):
            item_sim_dict = load_dict(mat_path)
        else:
            sim_matrix_generate(dat_path, mat_path)
            item_sim_dict = load_dict(mat_path)
        qua_path = os.path.join(dat_dir, f'{args.dataset}.qua')
        if os.path.exists(qua_path):
            item_quality_dict = load_dict(qua_path)
        else:
            item_quality_generate(dat_path, qua_path)
            item_quality_dict = load_dict(qua_path)
        pop_path = os.path.join(dat_dir, f'{args.dataset}.pop')
        if os.path.exists(pop_path):
            item_pop_dict = load_dict(pop_path)
        else:
            item_popularity_generate(dat_path, pop_path)
            item_pop_dict = load_dict(pop_path)

        # Train - val - test split
        train_path = os.path.join(dat_dir, f'{args.dataset}.train')
        valid_path = os.path.join(dat_dir, f'{args.dataset}.valid')
        test_path = os.path.join(dat_dir, f'{args.dataset}.test')
        if (os.path.exists(train_path)) \
                & (os.path.exists(valid_path)) \
                & (os.path.exists(test_path)):
            train_df = pd.read_csv(train_path, sep=',',
                                   names=['user_id', 'item_id', 'ratings', 'timestamp'])
            valid_df = pd.read_csv(valid_path, sep=',',
                                   names=['user_id', 'item_id', 'ratings', 'timestamp'])
            test_df = pd.read_csv(test_path, sep=',',
                                  names=['user_id', 'item_id', 'ratings', 'timestamp'])
        else:
            data_split(dat_path, train_path, valid_path, test_path)
            train_df = pd.read_csv(train_path, sep=',',
                                   names=['user_id', 'item_id', 'ratings', 'timestamp'])
            valid_df = pd.read_csv(valid_path, sep=',',
                                   names=['user_id', 'item_id', 'ratings', 'timestamp'])
            test_df = pd.read_csv(test_path, sep=',',
                                  names=['user_id', 'item_id', 'ratings', 'timestamp'])
    else:
        print("Please check if the dataset file exists!")

    # Train DQN
    if args.sim_mode == 'item_embedding' or args.sim_mode == 'user_embedding':
        assert item_emb is not None
    if args.sim_mode == 'user_embedding':
        assert repr_user is not None

    return train_dqn(train_df, test_df,
              item_sim_dict, item_quality_dict, item_pop_dict,
              max_item_id, item_list, mask_list, repr_user, item_emb, args)
