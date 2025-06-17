import os
import time
import yaml
import pickle
import random as rd
rd.seed(101)

import torch
from torch import nn
import numpy as np
from tqdm import tqdm

from eval_metrics import *

from RecAgent.model.dqn import DQN

### Load RL agent
rails, config = DQN.load_ckpt('ckpt/d1-r1')
rails.to_device('cpu') # Or cuda:0

### Cold-start historical interaction
interaction_history = [1, 3, 5, 7, 9]

### Setup environment
env = DQN.set_env(rails, interaction_history, wild_items= None)

### Predict top-K action
rec_list = rails.choose_action(obs= interaction_history,
                               env= env,
                               topK= 10)

print('Recommendation list:', rec_list)

### Testing with cold-start data

## Load additionals
wild_items = config['args']['wild_items'] # Unseen items

# Item popularity score
with open('ckpt/d1-r1/item_pop_dict.pkl', 'rb') as file:
    item_pop_dict = pickle.load(file) 
    file.close()

item_pop_dict = {int(k): v for k, v in item_pop_dict.items()} # Cast keys to int

## Load query-test pairs
with open('datasets/d1-fold/Round1/query.txt', 'r') as file:
    qLines = file.readlines()
    file.close()

with open('datasets/d1-fold/Round1/test.txt', 'r') as file:
    tLines = file.readlines()
    file.close()

queries = []
for line in qLines:
    words = [int(x) for x in line.split() if int(x) not in wild_items]
    queries.append(words[1:])

tests = []
for line in tLines:
    words = [int(x) for x in line.split() if int(x) not in wild_items]
    tests.append(words[1:])

## Infer and Evaluate
assert len(queries) == len(tests)

rec_list, test_list = [], []

for i in tqdm(range(len(tests))):
    query = queries[i]
    test = tests[i]

    env = DQN.set_env(rails, query, wild_items)
    rec = rails.choose_action(obs= query, 
                              env= env, 
                              topK= 10)

    if len(rec) == 0 or len(test) == 0:
        continue # Skip queries with only unseen items!

    rec_list.append(rec)
    test_list.append(test)
    

mean_precision, mean_recall = precison(rec_list, test_list), recall(rec_list, test_list)
mean_epc, mean_coverage = epc(rec_list, test_list, item_pop_dict), coverage(rec_list, test_list, item_pop_dict)

print(f'Precision@10: {mean_precision:.4f}')
print(f'Recall@10: {mean_recall:.4f}')
print(f'EPC@10: {mean_epc:.4f}')
print(f'Coverage@10: {mean_coverage:.4f}')