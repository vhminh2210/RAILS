import random as rd
rd.seed(101)
import collections
from types import new_class
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from .parse import parse_args
import time
import torch
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from .util import randint_choice
import os


# Helper function used when loading data from files
def helper_load(filename):
    user_dict_list = {}
    item_dict = set()

    with open(filename) as f:
        for line in f.readlines():
            line = line.strip('\n').split(' ')
            if len(line) == 0:
                continue
            line = [int(i) for i in line]
            user = line[0]
            items = line[1:]
            item_dict.update(items)
            if len(items) == 0:
                continue
            user_dict_list[user] = items

    return user_dict_list, item_dict,

def helper_load_train(filename):
    user_dict_list = {}
    item_dict = set()
    item_dict_list = {}
    trainUser, trainItem = [], []

    with open(filename) as f:
        for line in f.readlines():
            line = line.strip('\n').split(' ')
            if len(line) == 0:
                continue
            line = [int(i) for i in line]
            user = line[0]
            items = line[1:]
            item_dict.update(items)
            # LGN
            trainUser.extend([user] * len(items))
            trainItem.extend(items)
            if len(items) == 0:
                continue
            user_dict_list[user] = items

            for item in items:
                if item in item_dict_list.keys():
                    item_dict_list[item].append(user)
                else:
                    item_dict_list[item] = [user]

    return user_dict_list, item_dict, item_dict_list, trainUser, trainItem

class Data:
    
    def __init__(self, args):
        self.path = os.path.join(args.root, args.dataset)
        # self.small_path=args.root + args.dataset+".mid"+"/"
        self.train_file = os.path.join(self.path, 'train.txt')
        self.valid_file = os.path.join(self.path, 'val.txt')
        self.test_ood_file = os.path.join(self.path, 'test_ood.txt')
        self.test_id_file = os.path.join(self.path, 'test.txt')
        self.batch_size = args.enc_batch_size
        self.neg_sample = args.neg_sample
        self.sam=args.sam
        self.IPStype = args.IPStype
        if args.cuda >= 0:
            self.device = torch.device(args.cuda)
        else:
            self.device = 'cpu'
        self.modeltype = args.modeltype
        self.small_num=5000
        self.user_pop_max = 0
        self.item_pop_max = 0
        self.infonce = args.infonce
        self.num_workers = args.num_workers

        # Number of total users and items
        self.n_users, self.n_items, self.n_observations = 0, 0, 0
        self.users = []
        self.items = []
        self.population_list = []
        self.weights = []

        # List of dictionaries of users and its observed items in corresponding dataset
        # {user1: [item1, item2, item3...], user2: [item1, item3, item4],...}
        # {item1: [user1, user2], item2: [user1, user3], ...}
        self.train_user_list = collections.defaultdict(list)
        self.valid_user_list = collections.defaultdict(list)
        self.test_ood_user_list = collections.defaultdict(list)
        self.test_id_user_list = collections.defaultdict(list)

        # Used to track early stopping point
        self.best_valid_recall = -np.inf
        self.best_valid_epoch, self.patience = 0, 0

        self.train_item_list = collections.defaultdict(list)
        self.Graph = None
        self.trainUser, self.trainItem, self.UserItemNet = [], [], []
        self.n_interactions = 0
        self.test_ood_item_list = []
        self.test_id_item_list = []

        #Dataloader 
        self.train_data = None
        self.train_loader = None

    def load_data(self):
        self.train_user_list, train_item, self.train_item_list, self.trainUser, self.trainItem = helper_load_train(
            self.train_file)
        self.valid_user_list, valid_item = helper_load(self.valid_file)
        self.test_ood_user_list, self.test_ood_item_list = helper_load(self.test_ood_file)
        self.test_id_user_list, self.test_id_item_list = helper_load(self.test_id_file)
        self.pop_dict_list = []

        temp_lst = [train_item, valid_item, self.test_ood_item_list, self.test_id_item_list]

        self.users = list(set(self.train_user_list.keys()))
        self.items = list(set().union(*temp_lst))

        # Item and users are 0-based integers
        self.n_users = max([int(x) for x in self.users]) + 1
        self.n_items = max([int(x) for x in self.items]) + 1

        for i in range(self.n_users):
            if i in self.train_user_list.keys():
                self.n_observations += len(self.train_user_list[i])
                self.n_interactions += len(self.train_user_list[i])
            else:
                self.train_user_list[i] = []
            if i in self.valid_user_list.keys():
                self.n_interactions += len(self.valid_user_list[i])
            if i in self.test_id_user_list.keys():
                self.n_interactions += len(self.test_id_user_list[i])
            if i in self.test_ood_user_list.keys():
                self.n_interactions += len(self.test_ood_user_list[i])


        pop_dict = {}
        for item, users in self.train_item_list.items():
            pop_dict[item] = len(users) + 1
        for item in range(0, self.n_items):
            if item not in pop_dict.keys():
                pop_dict[item] = 1

            # Item popularity
            self.population_list.append(pop_dict[item])

        pop_user = {key: len(value) for key, value in self.train_user_list.items()}
        pop_item = {key: len(value) for key, value in self.train_item_list.items()}

        # Item popularity
        self.pop_item = pop_item
        sorted_pop_user = list(set(list(pop_user.values())))
        sorted_pop_item = list(set(list(pop_item.values())))
        sorted_pop_user.sort()
        sorted_pop_item.sort()

        # Maximum popularity rankings
        self.n_user_pop = len(sorted_pop_user)
        self.n_item_pop = len(sorted_pop_item)
        user_idx = {}
        item_idx = {}
        for i, item in enumerate(sorted_pop_user):
            user_idx[item] = i
        for i, item in enumerate(sorted_pop_item):
            item_idx[item] = i

        # Users and Items popularity rankings
        self.user_pop_idx = np.zeros(self.n_users, dtype=int)
        self.item_pop_idx = np.zeros(self.n_items, dtype=int)
        for key, value in pop_user.items():
            self.user_pop_idx[key] = user_idx[value]
        for key, value in pop_item.items():
            self.item_pop_idx[key] = item_idx[value]

        #self.item_pop_idx = torch.tensor(self.item_pop_idx).cuda(self.device)

        user_pop_max = max(self.user_pop_idx)
        item_pop_max = max(self.item_pop_idx)

        self.user_pop_max = user_pop_max
        self.item_pop_max = item_pop_max        

        self.weights = self.get_weight()
        self.weight_dict={i:self.weights[i] for i in range(len(self.weights))}
        self.sorted_weight=sorted(self.weight_dict.items(),key=lambda x: x[1])

        self.sample_pos_small={}
        self.sample_pos_big={}
        lo=0
        hi=1
        while hi<len(self.weights):
            if self.sorted_weight[hi][1]>self.sorted_weight[lo][1]:

                for i in range(lo,hi):
                    self.sample_pos_small[self.sorted_weight[i][0]]=hi
                lo=hi
            hi+=1     
        for i in range(lo,hi):
            self.sample_pos_small[self.sorted_weight[i][0]]=hi

        lo=len(self.weights)-2
        hi=len(self.weights)-1
        while lo>=0:
            if self.sorted_weight[lo][1]<self.sorted_weight[hi][1]:

                for i in range(hi,lo,-1):
                    self.sample_pos_big[self.sorted_weight[i][0]]=lo
                hi=lo
            lo-=1
        
        for i in range(hi,lo,-1):
            self.sample_pos_big[self.sorted_weight[i][0]]=lo

        self.sample_items = np.array(self.items, dtype=int)

        if self.modeltype == 'CausE':
            self.train_data = TrainDataset_cause(self.modeltype, self.users, self.train_user_list, self.n_observations, \
                                                self.n_interactions, self.pop_item, self.n_items, self.infonce, self.neg_sample, self.items, self.sample_items)
        else:
            self.train_data = TrainDataset(self.modeltype, self.users, self.train_user_list, self.user_pop_idx, self.item_pop_idx, \
                                        self.neg_sample, self.n_observations, self.n_items, self.sample_items, self.weights, self.infonce, self.items)

        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)
        
    def get_weight(self):

        if 's' in self.IPStype:
            pop = self.population_list
            pop = np.clip(pop, 1, max(pop))
            pop = pop / max(pop)
            return pop


        pop = self.population_list
        pop = np.clip(pop, 1, max(pop))
        pop = pop / np.linalg.norm(pop, ord=np.inf)
        pop = 1 / pop

        if 'c' in self.IPStype:
            pop = np.clip(pop, 1, np.median(pop))
        if 'n' in self.IPStype:
            pop = pop / np.linalg.norm(pop, ord=np.inf)

        return pop

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):

        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                # dist_mat=np.load_npy(self.path+'/dist_mat.npy')
                # dist_mat=dist_mat[:self.n_users, self.n_users:]
                # self.dist_mat=np.exp(-(dist_mat-1)/2)+1
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except:
                print("generating adjacency matrix")
                s = time.time()
                adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                self.trainItem = np.array(self.trainItem)
                self.trainUser = np.array(self.trainUser)
                self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                              shape=(self.n_users, self.n_items))
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.tocsr()
                sp.save_npz(self.path + '/adj_mat.npz', adj_mat)

                adj_mat = adj_mat.todok()


                # Laplacian smoothing
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time.time()
                print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)
            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(self.device)

        return self.Graph
  
class TrainDataset(torch.utils.data.Dataset):

    def __init__(self, modeltype, users, train_user_list, user_pop_idx, item_pop_idx, neg_sample, \
                n_observations, n_items, sample_items, weights, infonce, items):
        self.modeltype = modeltype
        self.users = users
        self.train_user_list = train_user_list
        self.user_pop_idx = user_pop_idx
        self.item_pop_idx = item_pop_idx
        self.neg_sample = neg_sample
        self.n_observations = n_observations
        self.n_items = n_items
        self.sample_items = sample_items
        self.weights = weights
        self.infonce = infonce
        self.items = items

    def __getitem__(self, index):

        index = index % len(self.users)
        user = self.users[index]
        if self.train_user_list[user] == []:
            pos_items = 0
        else:
            pos_item = rd.choice(self.train_user_list[user])

        user_pop = self.user_pop_idx[user]
        pos_item_pop = self.item_pop_idx[pos_item]
        pos_weight = self.weights[pos_item]

        if self.infonce == 1 and self.neg_sample == -1:

            return user, pos_item, user_pop, pos_item_pop, pos_weight

        elif self.infonce == 1 and self.neg_sample != -1:

            neg_items = randint_choice(self.n_items, size=self.neg_sample, exclusion=self.train_user_list[user])
            neg_items_pop = self.item_pop_idx[neg_items]

            return user, pos_item, user_pop, pos_item_pop, pos_weight, torch.tensor(neg_items).long(), neg_items_pop

        else:
            while True:
                neg_item = self.items[rd.randint(0, self.n_items -1)]
                if neg_item not in self.train_user_list[user]:
                    break
        
            neg_item_pop = self.item_pop_idx[neg_item]
            return user, pos_item, user_pop, pos_item_pop, pos_weight, neg_item, neg_item_pop

    def __len__(self):
        return self.n_observations

class TrainDataset_cause(torch.utils.data.Dataset):
    
    def __init__(self, modeltype, users, train_user_list, n_observations, n_interactions, pop_item, n_items, infonce, neg_sample, items, sample_items):
        self.modeltype = modeltype
        self.users = users
        self.train_user_list = train_user_list
        self.n_observations = n_observations
        self.n_interactions = n_interactions
        self.pop_item = pop_item
        self.n_items = n_items
        self.infonce = infonce
        self.neg_sample = neg_sample
        self.items = items
        self.sample_items = sample_items

    def __getitem__(self, index):

        index = index % len(self.users)
        user = self.users[index]
        pos_item = rd.choice(self.train_user_list[user])

        if self.infonce == 1 and self.neg_sample != -1:
            neg_items = self.get_neg_sample(user)
            neg_item = neg_items[0]
        else:
            while True:
                neg_item = self.items[rd.randint(0, self.n_items -1)]
                if neg_item not in self.train_user_list[user]:
                    break

        weight = 0.1 * self.n_interactions/len(self.pop_item)/self.pop_item[pos_item]
        if weight >= 1:
            weight = 0
        rad = rd.random()
        if rad < weight:
            pos_item += self.n_items
            neg_item += self.n_items  

        all_item = [pos_item, neg_item]
        ctrl_item = [i+self.n_items if i<self.n_items else i-self.n_items for i in all_item]

        return user, pos_item, neg_item, torch.tensor(all_item), torch.tensor(ctrl_item)

    def __len__(self):
        return self.n_observations


