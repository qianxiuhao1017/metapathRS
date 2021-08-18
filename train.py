# train.py
from tqdm import tqdm
import torch
import numpy as np

import torch.optim as optim
from utils import data_iterator, ranklist_by_sorted, get_performance # valid_one_user, test_one_user

import multiprocessing
cores = multiprocessing.cpu_count() // 2


from main import *


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
    # user_pos_test: groudtruth positive item list
    # test_items: unwatiched items for a user
    # rating: the scores on all items
    # Ks: [k1, k2, kn]  (for top-k)

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
    '''
    adj_ub_valid, edge2features_valid, u_ids_valid, i_ids_valid, ratings_valid \
        = construct_subgraph(ratings_valid, sim_dict, adj_u2b_dict, adj_b2u_dict,
                                total_num_nodes, tot_num_users)
    
    u_ids, i_ids, adj_ub, edge_mat, anchors, pos_items, neg_items \
    = data_loader.sample_train_sugraph()
   
    adj_ub_valid, edge2features_valid = adj_ub_test.cuda(), edge2features_test.cuda()
    '''
    pool = multiprocessing.Pool(cores)

    best_precision = 0.0
    for epoch in tqdm(range(args.epochs), desc='Epoch: '):
        '''
        model.train()
        optimizer.zero_grad()
        for batch_users in data_iterator(all_users, args.batch_size):

            u_ids, i_ids, adj_ub, edge_mat, anchors, pos_items, neg_items \
            = data_loader.sample_train_sugraph(batch_users)
            if use_cuda:
                u_ids, i_ids = u_ids.cuda(), i_ids.cuda()
                adj_ub, edge_mat = adj_ub.cuda(), edge_mat.cuda()
                anchors, pos_items, neg_items = anchors.cuda(), pos_items.cuda(), neg_items.cuda()
            with torch.set_grad_enabled(True):
                loss = model('train', u_ids, i_ids, edge_mat, adj_ub, anchors, pos_items, neg_items)
                loss.backward()
                optimizer.step()
        '''
        model.eval()
        count = 0
        result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
                  'hit_ratio': np.zeros(len(Ks))}
        print('eval')
        for batch_users in data_iterator(valid_users, args.batch_size):
            u_ids, i_ids, adj_ub, edge_mat, batch_users, predict_items = data_loader.sample_valid_graph(batch_users)
            if use_cuda:
                u_ids, i_ids = u_ids.cuda(), i_ids.cuda()
                adj_ub, edge_mat = adj_ub.cuda(), edge_mat.cuda()
                batch_users, predict_items = batch_users.cuda(), predict_items.cuda()
            print('edge_mat.shape', edge_mat.shape)
            with torch.set_grad_enabled(False):
                rate_batch = model('predict', u_ids, i_ids, edge_mat, adj_ub, batch_users, predict_items)
            print('finish eval')



            user_batch_rating_uid = zip(rate_batch.cpu().numpy(), batch_users.cpu().numpy())
            batch_result = pool.map(valid_one_user, user_batch_rating_uid)
        
            count += len(batch_result)

            for re in batch_result:   # the result of every user  ret is a dict, whose items each 
                result['precision'] += re['precision']/n_valid_users
                result['recall'] += re['recall']/n_valid_users
                result['ndcg'] += re['ndcg']/n_valid_users
                result['hit_ratio'] += re['hit_ratio']/n_valid_users

        precision = result['precision'][-1]
        if precision > best_precision:
            best_precision = precision
            best_model_dict = copy.deepcopy(model.state_dict())
            
    torch.save(best_model_dict, output_dir + 'model.pth')
