import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

def parseList(filename):
    resDict = {}

    with open(filename, 'r') as file:
        Lines = file.readlines()
        file.close()
    
    for line in Lines:
        words = [int(x) for x in line.split()]
        if words[0] in resDict.keys():
            print('WARNING: Some users appear more than once in rec/test list!')
        resDict[words[0]] = words[1 : ]

    return resDict

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='', help= 'Experiment directory')
parser.add_argument('--id', type=int, default='-1', help= 'Query id')
args = parser.parse_args()

ROOT = os.path.join('vis', args.exp, str(args.id))

if not os.path.exists('vis'):
    os.mkdir('vis')

print(ROOT)

if not os.path.exists(ROOT):
    os.makedirs(ROOT)

# if not os.path.exists(os.path.join(ROOT, 'reclist')):
#     os.makedirs(os.path.join(ROOT, 'reclist'))

# if not os.path.exists(os.path.join(ROOT, 'testlist')):
#     os.makedirs(os.path.join(ROOT, 'testlist'))

# Parse experiments
recPath = os.path.join(args.exp, 'reclist.txt')
testPath = os.path.join(args.exp, 'testlist.txt')

recDict = parseList(recPath)
testDict = parseList(testPath)

freq = np.loadtxt(os.path.join(args.exp, 'freq.txt'))

# Query
queries = []
if args.id == -1:
    queries = testDict.keys()
else:
    try:
        assert args.id in testDict.keys()
    except:
        raise ValueError(f"User {args.id} is not available!")
    queries = [args.id]

# Plots

# Reclist
if args.id == -1:
    query_loader = tqdm(queries)
else:
    query_loader = queries

for query in query_loader:
    save_pth = os.path.join(ROOT, f'vis_{query}.png')

    plt.figure(figsize= (20, 8))

    plt.subplot(121)
    for i in range(len(recDict[query])):
        item = recDict[query][i]
        if item in testDict[query]:
            plt.plot(i, freq[item], 'ro')
            plt.annotate(int(freq[item]), (i, freq[item]))

    plt.plot(freq[recDict[query]])
    plt.xlabel('Rec. Ranking')
    plt.ylabel('Popularity')
    plt.title(f'reclist_{query}')

    n_bin = min(10, len(testDict))

    plt.subplot(122)
    plt.hist(freq[testDict[query]], bins= n_bin)
    plt.xlabel('Popularity')
    plt.ylabel('Frequency')
    plt.title(f'testlist_{query}')
    plt.savefig(save_pth)

    plt.clf()
    plt.close()