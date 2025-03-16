import os
from sklearn.metrics import ndcg_score
import numpy as np

def _precision(reclist, testlist, k=10):
    '''
    precision = n_match / n_rec
    '''
    rec = reclist
    tst = testlist

    assert len(rec) >= k

    n_match = 0
    cnt = 0

    for r in rec:
        if r in tst: n_match += 1
        cnt += 1
        if cnt == k: break

    return float(n_match / k)

def precison(all_reclist, all_testlist, k=10):
    '''
    Average precision over testset
    '''
    assert len(all_reclist) == len(all_testlist)
    P = len(all_reclist)

    res = 0
    for p in range(P):
        res += _precision(all_reclist[p], all_testlist[p], k)
    return float(res / P)

def _recall(reclist, testlist, k=10):
    '''
    recall = n_match / n_tst
    '''
    rec = reclist
    tst = testlist

    assert len(rec) >= k

    n_match = 0
    cnt = 0

    for r in rec:
        if r in tst: n_match += 1
        cnt += 1
        if cnt == k: break

    return float(n_match / len(tst)) 

def recall(all_reclist, all_testlist, k=10):
    '''
    Average recall over testset
    '''
    assert len(all_reclist) == len(all_testlist)
    P = len(all_reclist)

    res = 0
    for p in range(P):
        res += _recall(all_reclist[p], all_testlist[p], k)
    return float(res / P)

def coverage(all_reclist, all_testlist, pop_dict, k=10):
    '''
    Catalog coverage.

    Popularity is calculated over trainset.
    '''
    rec_items = []
    tst_items = []

    for reclist in all_reclist:
        assert len(reclist) >= k
        rec_items.extend(reclist[:k])

    for testlist in all_testlist:
        tst_items.extend(testlist)

    rec_items = list(set(rec_items))
    tst_items = list(set(tst_items))

    # rec_items = [x for x in rec_items if x in tst_items]
    
    return float(len(rec_items) / len(pop_dict.keys()))

def epc(all_reclist, all_testlist, pop_dict, k=10):
    '''
    Expected Popularity Complement.

    Popularity is calculated over trainset
    '''
    assert len(all_reclist) == len(all_testlist)
    P = len(all_reclist) # Number of testing projects

    numer, denom = 0.0, 0.0

    for p in range(P):
        rec = all_reclist[p]
        tst = all_testlist[p]

        assert len(rec) >= k

        for idx in range(k):
            r = rec[idx]
            if r not in tst or r not in pop_dict.keys(): continue # rel(r, p) = 0 and ignore accidental wild items
            numer += float(1. - pop_dict[r]) / np.log2(idx + 2) # idx is 0-based, hence log2(idx + 2) is calculated
            denom += float(1. / np.log2(idx + 2))

    return float(numer / denom)

def lightgcn_ndcg(rank, ground_truth):
    '''
    https://github.com/kuandeng/LightGCN/blob/master/evaluator/python/evaluate_foldout.py#L36
    '''
    len_rank = len(rank)
    len_gt = len(ground_truth)
    idcg_len = min(len_gt, len_rank)

    # calculate idcg
    idcg = np.cumsum(1.0 / np.log2(np.arange(2, len_rank + 2)))
    idcg[idcg_len:] = idcg[idcg_len-1]

    # idcg = np.cumsum(1.0/np.log2(np.arange(2, len_rank+2)))
    dcg = np.cumsum([1.0/np.log2(idx+2) if item in ground_truth else 0.0 for idx, item in enumerate(rank)])
    result = dcg/idcg
    return result

def ndcg(all_reclist, all_testlist, k=10, mode= 'ranking'):
    '''
    Average NDCG over testsets
    '''
    assert len(all_reclist) == len(all_testlist)
    P = len(all_reclist)

    res = 0.0

    for p in range(P):
        rec = all_reclist[p]
        tst = all_testlist[p]

        y_true = []
        y_score = []

        if mode == 'lightgcn':
            res += lightgcn_ndcg(rank= rec, ground_truth= tst)[k-1] # Get ndcg@k
        
        else:
            for i, r in enumerate(rec):
                if mode == 'binary':
                    y_score.append(1)
                elif mode == 'ranking':
                    y_score.append(len(rec) - i) # Decreasing relevence score, which is >= 1
                else:
                    raise NotImplementedError(f'ndcg mode {mode} not implemented!')
                if r in tst: 
                    y_true.append(1)
                else:
                    y_true.append(0)

            # print(y_true)
            # print(y_score)
            res += ndcg_score(y_true= [y_true], y_score= [y_score], k= k)

    return float(res / P)