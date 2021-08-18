import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class ECGAT(nn.Module):
    def __init__(self, user_vocab_size, item_vocab_size, args, edge_feat_dimen):
        super(ECGAT, self).__init__()
        self.dropout = args.dropout
        self.margin = args.margin
        self.l2_weight = args.l2_weight
        self.edge_feat_dimen = edge_feat_dimen
        # self.W = nn.Parameter(torch.zeros(size=(node_feat_dimen, args.hidden[0]), device=args.device))
        # nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.user_embeddings = nn.Embedding(user_vocab_size, args.embed_size)  
        self.item_embeddings = nn.Embedding(item_vocab_size, args.embed_size)
        # self.mlp = MLP(edge_feat_dimen, args)
        # self.ec_conv = MPE(edge_feat_dimen, args)
        self.ec_conv = ECConvBranch(edge_feat_dimen, args)


    def predict(self, u_ids, i_ids, edge2features, adj_ub, user_ids, item_ids):

        """
        user_ids:   number of users to evaluate   (n_eval_users) 
        item_ids:   number of items to evaluate   (n_eval_items)
        """
        u_feat = self.user_embeddings(u_ids)
        i_feat = self.item_embeddings(i_ids)
        Z_u, Z_i = self.ec_conv(u_feat, i_feat, edge2features, adj_ub)

        user_embeds = Z_u[user_ids]
        item_embeds = Z_i[item_ids]
        ui_score = torch.matmul(user_embeds, item_embeds.transpose(0, 1))     # (n_eval_users, n_eval_items)
        '''
        user_embeds = self.embeddings(user_ids)
        item_embeds = self.embeddings(item_ids)
        ui_score = torch.matmul(user_embeds, item_embeds.transpose(0, 1)) 
        '''
        return ui_score
    

    def cal_loss(self, u_ids, i_ids, edge2features, adj_ub, anchors, pos_inputs, neg_inputs):
        u_feat = self.user_embeddings(u_ids)
        i_feat = self.item_embeddings(i_ids)

        Z_u, Z_i = self.ec_conv(u_feat, i_feat, edge2features, adj_ub)

        anchor_embeds = Z_u[anchors]
        pos_embeds = Z_i[pos_inputs]
        neg_embeds = Z_i[neg_inputs]

        neg_aff = torch.sum(torch.mul(anchor_embeds, neg_embeds), axis = 1)

        triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=2)
        loss = triplet_loss(anchor_embeds, pos_embeds, neg_embeds)
        return loss
    
    def forward(self, mode, u_ids, i_ids, edge2features, adj_ub, anchors, pos_inputs, neg_inputs=None):
        if mode == 'train':
            return self.cal_loss(u_ids, i_ids, edge2features, adj_ub, anchors, pos_inputs, neg_inputs)
        elif mode == 'predict':
            return self.predict(u_ids, i_ids, edge2features, adj_ub, anchors, pos_inputs)


    '''
    def _hinge_loss(self, inputs1, inputs2, neg_samples, hard_neg_samples=None):
        aff =tf.sigmoid(self.affinity(inputs1, inputs2))
        neg_aff = tf.sigmoid(self.affinity(inputs1, neg_samples))
        diff = tf.nn.relu(tf.subtract(neg_aff,aff - self.margin), name='diff')
        loss = tf.sum(diff)
        self.neg_shape = tf.shape(neg_aff)
        return loss 
    '''

    '''
    def affinity(self, inputs1, inputs2):
        """ Affinity score between batch of inputs1 and inputs2.
        Args:
            inputs1: tensor of shape [batch_size x feature_size].
        """
        # shape: [batch_size, input_dim1]
        if self.bilinear_weights:
            prod = tf.matmul(inputs2, tf.transpose(self.vars['weights']))
            self.prod = prod
            result = tf.sum(inputs1 * prod, axis=1)
        else:
            result = tf.sum(inputs1 * inputs2, axis=1)
            # 对应位相乘， inputs1， inputs2 必须维度相等 
        return result


    tf.nn.l2_normalize()
    '''


class ECConvBranch(nn.Module):
    def __init__(self, edge_feat_dimen, args):
        super(ECConvBranch, self).__init__()
        # self.filter_layers = nn.Sequential()
        self.edge_feat_dimen = edge_feat_dimen
        self.dropout = args.dropout
        self.diffuse = args.diffuse
        hidden_edge_dimen = args.hidden_edge_dimen
        
        self.mlp = torch.nn.Sequential(\
                nn.Linear(edge_feat_dimen, hidden_edge_dimen, bias=True),
                nn.BatchNorm1d(hidden_edge_dimen),
                nn.ELU(alpha=1.0, inplace=True),
                nn.Dropout(args.dropout),
                nn.Linear(hidden_edge_dimen, 1, bias=True),
                nn.ELU(alpha=1.0, inplace=True),
                nn.Dropout(args.dropout)
                )


    def forward(self, Zu, Zb, edge_features_u2b, adj_ub):
        n_row, n_col = adj_ub.shape[0], adj_ub.shape[1]
        # Learn alpha graph

        att_map_u2b = self.mlp(edge_features_u2b)
        att_map_u2b = att_map_u2b.view(n_row, n_col)
        att_map_u2b = F.softmax(att_map_u2b, dim=1)
        att_map_b2u = F.softmax(att_map_u2b.transpose(0, 1), dim=1)

        for i in range(self.diffuse):
            Zneighbor_u = torch.matmul(torch.mul(adj_ub, att_map_u2b), Zb)  
            Zneighbor_b = torch.matmul(torch.mul(adj_ub.transpose(0, 1), att_map_b2u), Zu)
            Zu = F.relu(torch.add(Zu, Zneighbor_u), inplace=True)
            Zb = F.relu(torch.add(Zb, Zneighbor_b), inplace=True)
            Zu = F.normalize(Zu, p=2, dim=1)
            Zb = F.normalize(Zb, p=2, dim=1)
        return Zu, Zb
