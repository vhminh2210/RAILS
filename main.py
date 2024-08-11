import random as rd
import argparse
from sys import exit
rd.seed(101)
import torch
import time
import numpy as np
from tqdm import tqdm

from GraphEnc.encoder import getEncoder

if __name__ == "__main__":

    # Graph encoder
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    encoder = getEncoder(args)

    # Interactive RL Agent
    pass