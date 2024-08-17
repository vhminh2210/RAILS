import random as rd
import pandas as pd
import os
import csv

def agent_data(filename, args):

    if filename == 'dat':
        TXT = os.path.join(args.root, args.dataset, f'{args.dataset}.txt')
    else:
        TXT = os.path.join(args.root, args.dataset, f'{filename}.txt')
    CSV = os.path.join(args.root, args.dataset, f'{args.dataset}.{filename}')

    fields = ['user_id', 'item_id', 'ratings', 'timestamp']

    with open(TXT, 'r') as file:
        Lines = file.readlines()
        file.close()

    rows = []

    for line in Lines:
        line = line.strip('\n').split()
        user = line[0]
        items = line[1:]
        for item in items:
            rows.append({
                'user_id' : user,
                'item_id' : item,
                'ratings' : 1,
                'timestamp' : 1000
            })

    with open(CSV, 'w', newline='') as file:
        csvwritter = csv.DictWriter(file, fieldnames= fields)
        csvwritter.writerows(rows)
    file.close()

def split_data(args, train_ratio= 0.8, val_ratio= 0.1, test_ratio= 0.1, seed= 101):

    assert train_ratio + val_ratio + test_ratio == 1
    train, val, test = [], [], []

    TXT = os.path.join(args.root, args.dataset, f'{args.dataset}.txt')
    TRAIN = os.path.join(args.root, args.dataset, 'train.txt')
    VAL = os.path.join(args.root, args.dataset, 'val.txt')
    TEST = os.path.join(args.root, args.dataset, 'test.txt')
    TEST_OOD = os.path.join(args.root, args.dataset, 'test_ood.txt')

    with open(TXT, 'r') as file:
        Lines = file.readlines()
        file.close()

    for line in Lines:
        line = line.strip('\n').split()
        user = line[0]
        items = line[1:]

        trainsize = int(len(items) * train_ratio)
        valsize = int(len(items) * val_ratio)
        testsize = len(items) - trainsize - valsize

        mask = []
        for i in range(trainsize):
            mask.append(0)
        for i in range(valsize):
            mask.append(1)
        for i in range(testsize):
            mask.append(2)

        rd.seed(seed)
        rd.shuffle(mask)

        trainitem, valitem, testitem = [], [], []
        trainstr, valstr, teststr = str(user), str(user), str(user)

        for i in range(len(mask)):
            if mask[i] == 0:
                trainitem.append(i)
                trainstr = trainstr + ' ' + str(i)
            elif mask[i] == 1:
                valitem.append(i)
                valstr = valstr + ' ' + str(i)
            elif mask[i] == 2:
                testitem.append(i)
                teststr = teststr + ' ' + str(i)

        train.append(trainstr + '\n')
        val.append(valstr + '\n')
        test.append(teststr + '\n')

    # Data for graph encoders
    with open(TRAIN, 'w') as file:
        file.writelines(train)
    file.close()

    with open(VAL, 'w') as file:
        file.writelines(val)
    file.close()

    with open(TEST, 'w') as file:
        file.writelines(test)
    file.close()

    with open(TEST_OOD, 'w') as file:
        file.writelines(test)
    file.close()

    # Data for RL agents
    agent_data('dat', args)
    agent_data('train', args)
    agent_data('val', args)
    agent_data('test', args)