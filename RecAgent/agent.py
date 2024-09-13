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

        # Train - val - test split
        train_path = os.path.join(dat_dir, f'{args.dataset}.train')
        valid_path = os.path.join(dat_dir, f'{args.dataset}.val')
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
            
        max_item_id = train_df['item_id'].max()
        if args.sim_mode != 'stats':
            max_item_id = item_emb.weight.shape[0] - 1 # When the training sets doesnt contain all possible item
        item_list = train_df['item_id'].tolist()
        mask_list = list(set(list(range(max_item_id + 1))) - set(item_list))
            
        # Popularity dict: {item1 : popularity_percentile_of_item1, ...}
        # NOTE: Least popular item: percentile = 0.0, Most popular item: percentile = 1.0
        pop_path = os.path.join(dat_dir, f'{args.dataset}.pop')
        if os.path.exists(pop_path):
            item_pop_dict = load_dict(pop_path)
        else:
            # item_popularity_generate(dat_path, pop_path)
            item_popularity_generate(train_path, pop_path) # Use training set to generate popularity info
            item_pop_dict = load_dict(pop_path)
    else:
        print("Please check if the dataset file exists!")

    # Train DQN
    if args.sim_mode == 'item_embedding' or args.sim_mode == 'user_embedding':
        assert item_emb is not None
    if args.sim_mode == 'user_embedding':
        assert repr_user is not None

    return train_dqn(train_df, test_df, item_pop_dict,
                    max_item_id, item_list, mask_list, repr_user, item_emb, args)
