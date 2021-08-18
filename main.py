# -*- coding = utf-8 -*-
from __future__ import division
from __future__ import print_function
# from preprocess import save_lines_list
import scipy.io as scio
import copy
import gc
import time
import numpy as np
import os
import argparse 
import torch
import torch.nn as nn
#import torch.optim as optim
# from utils import data_iterator, normalize_adj, sim_tensor_from_triplets
from models import ECGAT

from collections import defaultdict
import pickle as pkl
from loader_hin import HIN_Graph_Loader
# from train import train_valid
# train.py
from tqdm import tqdm
import torch.optim as optim
from utils import data_iterator, ranklist_by_sorted, get_performance, save_csv
# import nni
# from nni.utils import merge_parameter

import multiprocessing
cores = multiprocessing.cpu_count() // 2


def get_args():
    parser = argparse.ArgumentParser(description='ECGAT')

    # Dataset
    parser.add_argument('-d', '--dataset', default='Douban_Book_split', help='Dataset name: Douban_Book|Douban_Movie|Yelp_Business ')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
    parser.add_argument('--model_type', default='Ours', help='Model type')
    # Optimization arguments
    parser.add_argument('--epochs', default=2, type=int,
                        help='Number of epochs to train. If <=0, only testing will be done.')
    parser.add_argument('--wd', default=5e-4, type=float, help='Weight decay')
    parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--dropout', default=0.3, type=float, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--l2_weight', type=float, default=1e-3, help='')
    parser.add_argument('--margin', type=float, default=0.05, help='')

    parser.add_argument('--diffuse', type=int, default  =2, help='Total number of steps for diffusion')

    # Learning process arguments
    # Model
    parser.add_argument('--seed', default=1234, type=int, help='Seed for random initialisation')
    parser.add_argument('--size_neighbors', type=int, default=10, help='size_neighbors in the U**B')
    parser.add_argument('--hidden_edge_dimen', default=10, type=int, help='Dimension of the latent factors of relations/meta-graphs')
    # H_dim, Z_l_dimen, fusion_latent_dimen, Z_union_dimen
    parser.add_argument("--embed_size", type=int, default=128, help="embedding size of user and item embeddings")

    return parser.parse_args()


args = get_args()
os.environ["CUDA_VISIBLE_DEVICES"]="4"
Ks = [5, 10, 20, 50, 100]


np.random.seed(args.seed)
torch.manual_seed(args.seed)
use_cuda = torch.cuda.is_available()
n_gpu = torch.cuda.device_count()
if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

data_dir = '../processed_data/' + args.dataset
sim_res_dir = data_dir + '/sim_res'

output_dir = 'models/' + args.dataset
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

precess_from_file = True

global metapaths
if 'Douban_Book' in args.dataset:  
    metapaths = ['UB', 'UBUB', 'UBAB']  # 'UBPubBUB',  'UBYBUB' , 'UUB', 'UGUB', 'UBABUB', 'UBPubB', 'UBYB'
elif 'Douban_Movie' in args.dataset:
    metapaths = ['UM', 'UMUM', 'UMDM', 'UMAM', 'UMTM']  # 'UGUM' 
elif 'Yelp' in args.dataset:
    metapaths = ['UB', 'UBUB',  'UBCaB', 'UBCiB']  # 'UBCiBU'  'UUB', 'UCompUB',
else:
    print('the dataset is not supported here')
    exit(0)

data_loader = HIN_Graph_Loader(args, data_dir, sim_res_dir, metapaths)

tot_num_users, tot_num_items = data_loader.tot_num_users, data_loader.tot_num_items

model = ECGAT(user_vocab_size=tot_num_users, item_vocab_size=tot_num_items, 
                args=args,
                edge_feat_dimen=len(metapaths)
                )

def valid_one_user(x):
    rating = x[0]
    u = x[1]
    #user u's items in the training set
    try:
        training_items = data_loader.train_user2item[u]
    except Exception:
        training_items = []
    #user u's items in the test set
    user_pos_test = data_loader.valid_user2item[u]

    all_items = set(range(data_loader.tot_num_items))
    if u in data_loader.test_user2item:
        test_items = list(all_items - set(training_items)-set(data_loader.test_user2item[u]))
    else:
        test_items = list(all_items - set(training_items))
    r = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, Ks)


def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    # ratings is the predicted score on all items for the user u.
    #uid
    u = x[1]
    #user u's items in the training set
    try:
        training_items = data_loader.train_user2item[u]
    except Exception:
        training_items = []
    #user u's items in the test set
    user_pos_test = data_loader.test_user2item[u]


    all_items = set(range(data_loader.tot_num_items))
    if u in data_loader.valid_user2item:
        test_items = list(all_items - set(training_items) - set(data_loader.valid_user2item[u]))
    else:
        test_items = list(all_items - set(training_items))

    r = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, Ks)


def train_valid(args, data_loader, model, output_dir, Ks):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(),
                            lr=args.lr,
                            weight_decay=args.wd)
    
    all_users = list(data_loader.all_users)
    valid_users = list(data_loader.valid_user2item.keys())
    n_valid_users = len(valid_users)

    pool = multiprocessing.Pool(cores)

    best_precision = 0.0
    for epoch in tqdm(range(args.epochs), desc='Epoch: '):
        
        model.train()
        for batch_users in data_iterator(all_users, args.batch_size):

            u_ids, i_ids, adj_ub, edge_mat, anchors, pos_items, neg_items \
            = data_loader.sample_train_sugraph(batch_users)
            if use_cuda:
                u_ids, i_ids = u_ids.cuda(), i_ids.cuda()
                adj_ub, edge_mat = adj_ub.cuda(), edge_mat.cuda()
                anchors, pos_items, neg_items = anchors.cuda(), pos_items.cuda(), neg_items.cuda()
            with torch.set_grad_enabled(True):
                loss = model('train', u_ids, i_ids, edge_mat, adj_ub, anchors, pos_items, neg_items)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        model.eval()
        count = 0
        result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
                  'hit_ratio': np.zeros(len(Ks))}
        for batch_users in data_iterator(valid_users, args.batch_size):
            u_ids, i_ids, adj_ub, edge_mat, batch_users, predict_items = data_loader.sample_valid_graph(batch_users)
            if use_cuda:
                u_ids, i_ids = u_ids.cuda(), i_ids.cuda()
                adj_ub, edge_mat = adj_ub.cuda(), edge_mat.cuda()
                batch_users, predict_items = batch_users.cuda(), predict_items.cuda()

            with torch.set_grad_enabled(False):
                rate_batch = model('predict', u_ids, i_ids, edge_mat, adj_ub, batch_users, predict_items)
                # self.predict(u_ids, i_ids, edge2features, adj_ub, anchors, pos_inputs)

            user_batch_rating_uid = zip(rate_batch.cpu().numpy(), u_ids.cpu().numpy())
            batch_result = pool.map(valid_one_user, user_batch_rating_uid)
        
            count += len(batch_result)

            for re in batch_result:   # the result of every user  ret is a dict, whose items each 
                result['precision'] += re['precision']/n_valid_users
                result['recall'] += re['recall']/n_valid_users
                result['ndcg'] += re['ndcg']/n_valid_users
                result['hit_ratio'] += re['hit_ratio']/n_valid_users

        precision = result['recall'][1]
        # nni.report_intermeidate_result(precision)
        if precision > best_precision:
            best_precision = precision
            best_model_dict = copy.deepcopy(model.state_dict())
    
    pth_name = 'model-h%d-lr%g.pth'%(args.hidden_edge_dimen, args.lr)

    torch.save(best_model_dict, os.path.join(output_dir, pth_name))



def test(args, data_loader, model, output_dir, Ks):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()

    pth_name = 'model-h%d-lr%g.pth'%(args.hidden_edge_dimen, args.lr)
    model.load_state_dict(torch.load(os.path.join(output_dir, pth_name)))
   
    all_users = list(data_loader.all_users)
    test_users = list(data_loader.test_user2item.keys())
    n_test_users = len(test_users)

    pool = multiprocessing.Pool(cores)
        
    model.eval()
    count = 0
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
                'hit_ratio': np.zeros(len(Ks))}
    for batch_users in data_iterator(test_users, args.batch_size):
        u_ids, i_ids, adj_ub, edge_mat, batch_users, predict_items = data_loader.sample_valid_graph(batch_users)
        if use_cuda:
            u_ids, i_ids = u_ids.cuda(), i_ids.cuda()
            adj_ub, edge_mat = adj_ub.cuda(), edge_mat.cuda()
            batch_users, predict_items = batch_users.cuda(), predict_items.cuda()

        with torch.set_grad_enabled(False):
            rate_batch = model('predict', u_ids, i_ids, edge_mat, adj_ub, batch_users, predict_items)

        user_batch_rating_uid = zip(rate_batch.cpu().numpy(), u_ids.cpu().numpy())
        batch_result = pool.map(test_one_user, user_batch_rating_uid)

        count += len(batch_result)

        for re in batch_result:   # the result of every user  ret is a dict, whose items each 
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users

    # print('best_recall', result['recall'][1])   
    # nni.report_final_result(result['precision'][-1])
    save_csv(args, result, '../results/%s' % args.dataset, Ks)




if __name__ == '__main__':
    args = get_args()
    train_valid(args, data_loader, model, output_dir, Ks)
    test(args, data_loader, model, output_dir, Ks)

