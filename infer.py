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

from RecAgent.model.dqn import DQN

# Load RL agent
rails = DQN.load_ckpt('ckpt/d1-r1')

# Cold-start historical interaction
interaction_history = [1, 3, 5, 7, 9]

# Setup environment
env = DQN.set_env(rails, interaction_history)

# Predict top-K action
rec_list = rails.choose_action(obs= interaction_history,
                               env= env,
                               topK= 10)

print('Recommendation list:', rec_list)