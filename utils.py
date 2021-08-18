import random
import heapq
import numpy as np
import metrics
import os 
import pandas as pd
from collections import defaultdict
import pickle as pk

Ks = [5, 10, 20, 50, 100]


def load_user2item(read_file):
    user2item = {}
    for line in open(read_file, 'r').readlines():
        line = line.strip().split(' ')
        user2item[int(line[0])] = [int(i) for i in line[1:]]
    return user2item

def data_iterator(all_users, batch_size):
    # [train_u_indices, train_v_indices, train_labels]

    # shuffle labels and features
    max_idx = len(all_users)
    random.shuffle(list(all_users))

    # Does not yield last remainder of size less than batch_size

    for i in range(max_idx//batch_size):
        user_batch = all_users[i*batch_size:(i+1)*batch_size]
        yield user_batch



def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    # user_pos_test: groudtruth positive item list
    # test_items: unwatiched items for a user
    # rating: the scores on all items
    # Ks:  [k1, k2, kn]  (for top-k)
    item_score = {}
    # item2score
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    return r


def get_performance(user_pos_test, r, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []
    # r: 
    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio)}


def save_csv(args, ret, result_dir, Ks):
    result_file = {}
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    pkl_name = 'results-h%d-lr%g.pkl'%(args.hidden_edge_dimen, args.lr)
    csv_name = 'results_csv-h%d-lr%g.csv'%(args.hidden_edge_dimen, args.lr)

    result_file = {}
    result_file['pkl'] = os.path.join(result_dir, pkl_name)
    result_file['csv'] = os.path.join(result_dir, csv_name)

    if not os.path.exists(os.path.sep.join([result_file['pkl']])):
        pkl_data = defaultdict(dict)
    else:
        with open(os.path.sep.join([result_file['pkl']]), 'rb') as f:
            pkl_data = pk.load(f)
    for key in ret.keys():
        if key == 'auc':
            metric = key
            value = ret[key]
            pkl_data[args.model_type][metric] = value
        else:
            for i, k in enumerate(Ks):
                metric = key + '@' + str(k)
                value = ret[key][i]
                pkl_data[args.model_type][metric] = value

    with open(os.path.sep.join([result_file['pkl']]), 'wb') as f:
        pk.dump(pkl_data, f)
    
    data =defaultdict(dict)
    for method in pkl_data:
        for metric in pkl_data[method]:
            data[metric][method] = '{:.4f}'.format(pkl_data[method][metric])
    
    data = pd.DataFrame(data)
    data.to_csv(os.path.sep.join([result_file['csv']]))
