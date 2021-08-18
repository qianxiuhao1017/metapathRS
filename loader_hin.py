# HIN_Graph_Loader()

import os
import random
import collections

import torch
import numpy as np
import pandas as pd

from utils import load_user2item

class HIN_Graph_Loader(object):
    def __init__(self, args, data_dir, sim_res_dir, metapaths):
        self.batch_size = args.batch_size
        self.data_dir = data_dir
        self.sim_res_dir = sim_res_dir
        self.seed = args.seed
        self.size_neighbors = args.size_neighbors
        self.load_user2item()
        self.load_n_users_items()
        self.metapaths = metapaths
        self.get_sim_data()
    
    def load_user2item(self):
        self.train_user2item = load_user2item(os.path.join(self.data_dir, 'train.txt'))
        self.valid_user2item = load_user2item(os.path.join(self.data_dir, 'valid.txt'))
        self.test_user2item = load_user2item(os.path.join(self.data_dir, 'test.txt'))
        self.all_users = self.train_user2item.keys()
        self.train_item2user = {}
        for u in self.train_user2item:
            for item in self.train_user2item[u]:
                if item not in self.train_item2user:
                    self.train_item2user[item] = []
                self.train_item2user[item].append(u)

    def load_n_users_items(self):
        n_users_items = np.loadtxt(os.path.join(self.sim_res_dir,'n_users_items.txt'), dtype=np.int)
        self.tot_num_users, self.tot_num_items = n_users_items[0], n_users_items[1]
        self.all_item_set = set(list(range(self.tot_num_items)))
        
    def load_metapath(self, meta_file):
        UI2sim = {}
        for line in open(meta_file, 'r').readlines():
            line = line.strip().split(' ')
            u, i = int(line[0]),  int(line[1])
            UI2sim[(u, i)] = float(line[2])
            if u not in self.adj_u2b_dict:
                self.adj_u2b_dict[u] = set()
            self.adj_u2b_dict[u].add(i)
            if i not in self.adj_b2u_dict:
                self.adj_b2u_dict[i] = set()
            self.adj_b2u_dict[i].add(u)
        return UI2sim
        
    def get_sim_data(self):
        self.sim_dict = {}
        self.adj_u2b_dict, self.adj_b2u_dict = {}, {}
        for meta in self.metapaths:
            sim_file = os.path.join(self.sim_res_dir, meta + '.dat')
            self.sim_dict[meta] = self.load_metapath(sim_file)
    
    def sample_train_sugraph(self, batch_users):
        anchors_list = []
        pos_list = []
        neg_list = []
        for u in batch_users:
            unwatched_set = self.all_item_set - set(self.train_user2item[u])
            if u in self.valid_user2item:
                unwatched_set -= set(self.valid_user2item[u])
            if u in self.test_user2item:
                unwatched_set -= set(self.test_user2item[u])
            train_neg_item = np.random.choice(list(unwatched_set), len(self.train_user2item[u]), replace=False)
            for pos, neg in zip(self.train_user2item[u], train_neg_item):
                anchors_list.append(u)
                pos_list.append(pos)  #  + self.tot_num_users
                neg_list.append(neg)  #  + self.tot_num_users

        u_nodes_list, i_nodes_list, adj_ub, edge_mat, anchors_list, pos_list, neg_list = self.sample_sugraph(anchors_list, pos_list, neg_list)

        # 用 u_ids, i_ids remap anchors, pos_items, neg_items 
        # remap()

        u_ids = torch.LongTensor(u_nodes_list)
        i_ids = torch.LongTensor(i_nodes_list)
        anchors = torch.LongTensor(anchors_list)
        pos_items = torch.LongTensor(pos_list)
        neg_items = torch.LongTensor(neg_list)
        
        return u_ids, i_ids, adj_ub, edge_mat, anchors, pos_items, neg_items
    
    def sample_sugraph(self, anchors, pos_items, neg_items):
        u_nodes_set_0 = set(anchors)
        i_nodes_set_0 = set(pos_items + neg_items)

        u_nodes_set = set()
        i_nodes_set = set()

        for u in list(u_nodes_set_0):
            i_nodes_set.union(set(self.train_user2item[u]))
                
        for i in list(i_nodes_set_0):
            if i in self.train_item2user:
                u_nodes_set.union(set(self.train_item2user[i]))

        for u in list(u_nodes_set_0):
            tmp_set = self.adj_u2b_dict[u]
            sample_size = min(len(tmp_set), self.size_neighbors)
            i_nodes_set.union(set(np.random.choice(list(tmp_set), sample_size, replace=False)))
        
        for i in list(i_nodes_set_0):
            if i in self.adj_b2u_dict:
                tmp_set = self.adj_b2u_dict[i]
                sample_size = min(len(tmp_set), self.size_neighbors)
                u_nodes_set.union(set(np.random.choice(list(tmp_set), sample_size, replace=False))) 

        u_nodes_list = list(u_nodes_set | u_nodes_set_0)
        i_nodes_list = list(i_nodes_set | i_nodes_set_0)
        
        adj_ub = torch.zeros(len(u_nodes_list), len(i_nodes_list))
        edge_mat = torch.zeros(len(u_nodes_list)*len(i_nodes_list), len(self.metapaths))

        user_remap = {u_id: i for i, u_id in enumerate(u_nodes_list)}
        item_remap = {i_id: i for i, i_id in enumerate(i_nodes_list)}

        n_col = len(i_nodes_list)
        for uid in u_nodes_list:
            i = user_remap[uid]
            for bid in i_nodes_list:
                j = item_remap[bid]
                if bid in self.adj_u2b_dict[uid]:
                    adj_ub[i, j] = 1
                    for l, meta in enumerate(self.metapaths):
                        if (uid, bid) in self.sim_dict[meta]:
                            edge_mat[i*n_col + j, l] = self.sim_dict[meta][(uid, bid)]
        # remap 一下 anchors, pos_items, neg_items
        anchors = [user_remap[u] for u in anchors]
        pos_items = [item_remap[i] for i in pos_items]
        neg_items = [item_remap[i] for i in neg_items]
        return u_nodes_list, i_nodes_list, adj_ub, edge_mat, anchors, pos_items, neg_items 


    def get_valid(self):
        users = self.valid_user2item.keys()
        for u in users:
            item_candidates = self.all_item_set - set(self.train_user2item[u])
            if u in self.test_user2item:
                item_candidates -= set(self.test_user2item[u])
        
  
    def sample_valid_graph(self, u_nodes_list):
        '''
        unwatched_set = self.all_item_set - set(train_user2item[u])
        if u in self.valid_user2item:
            unwatched_set -= set(self.valid_user2item[u])
        '''
        
        i_nodes_list = list(range(self.tot_num_items))
        
        adj_ub = torch.zeros(len(u_nodes_list), len(i_nodes_list))
        edge_mat = torch.zeros(len(u_nodes_list)*len(i_nodes_list), len(self.metapaths))

        user_remap = {u_id: i for i, u_id in enumerate(u_nodes_list)}
        item_remap = {i_id: i for i, i_id in enumerate(i_nodes_list)}
        n_col = len(i_nodes_list)
        for uid in u_nodes_list:
            i = user_remap[uid]
            for bid in i_nodes_list:
                j = item_remap[bid]
                if bid in self.adj_u2b_dict[uid]:
                    adj_ub[i, j] = 1
                    for l, meta in enumerate(self.metapaths):
                        if (uid, bid) in self.sim_dict[meta]:
                            edge_mat[i*n_col+j, l] = self.sim_dict[meta][(uid, bid)]
        # remap 一下 anchors, pos_items, neg_items
        users = [user_remap[u] for u in u_nodes_list]
        predict_items = [item_remap[i] for i in i_nodes_list]


        u_ids = torch.LongTensor(u_nodes_list)
        i_ids = torch.LongTensor(i_nodes_list)
        users = torch.LongTensor(users)
        predict_items = torch.LongTensor(predict_items)
        
        return u_ids, i_ids, adj_ub, edge_mat, users, predict_items
    