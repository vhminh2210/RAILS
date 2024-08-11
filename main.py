import random as rd
import argparse
from sys import exit
rd.seed(101)
import torch
import time
import numpy as np
from tqdm import tqdm

from GraphEnc.encoder import getEncoder
from RecAgent.agent import getAgent

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # RL Agent params
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--obswindow', type=int, default=10)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--memory', type=int, default=20000)
    parser.add_argument('--replace_freq', type=int, default=99)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epsilon', type=float, default=0.95)
    parser.add_argument('--gamma', type=float, default=0.90)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--episode_max', type=int, default=100)
    parser.add_argument('--step_max', type=int, default=10000)
    parser.add_argument('--j', type=int, default=16)

    args = parser.parse_args()

    # Graph encoder
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    print('GCF training starts...')
    encoder = getEncoder(args)
    print('####################')

    # Interactive RL Agent
    print('RL Agent training starts...')
    agent = getAgent(args)
    pass