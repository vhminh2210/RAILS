import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='', help= 'Experiment directory')
parser.add_argument('--id', type=str, default='-1', help= 'Query id')
args = parser.parse_args()

ROOT = os.path.join('vis', args.exp, args.id)

if not os.path.exists(ROOT):
    os.makedirs(ROOT)

