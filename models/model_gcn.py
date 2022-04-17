import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# mask: https://www.zhihu.com/question/320615749/answer/1080485410
def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e30)

# dependency relation有关的attention在feature上做操作
class Rel_GAT(nn.Module):
    """ 
    Relation gat model, use the embedding of the edges to predict attention weight
    """

    def __init__(self, args, dep_rel_num,  hidden_size=64,  num_layers=3):
        super(Rel_GAT, self).__init__()
        self.args = args
        self.num_layers = num_layers
        self.dropout = nn.Dropout(args.gcn_dropout)
        self.leakyrelu = nn.LeakyReLU(1e-2)


        # gat layer
        # relation embedding, careful initialization?
        self.dep_rel_embed = nn.Embedding(
            dep_rel_num, args.dep_relation_embed_dim)
        nn.init.xavier_uniform_(self.dep_rel_embed.weight)

        # map rel_emb to logits. Naive attention on relations
        layers = [
            nn.Linear(args.dep_relation_embed_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)]
        self.fcs = nn.Sequential(*layers)

    def forward(self, adj, rel_adj, feature):
        denom = adj.sum(2).unsqueeze(2) + 1
        B, N = adj.size(0), adj.size(1)

        rel_adj_V = self.dep_rel_embed(
            rel_adj.view(B, -1))  # (batch_size, L*L, d)

        # gcn layer
        for l in range(self.num_layers):
            # relation based GAT, attention over relations
            
            if True:
                rel_adj_logits = self.fcs(rel_adj_V).squeeze(2)  # (batch_size, L*L)
            else:
                rel_adj_logits = self.A[l](rel_adj_V).squeeze(2)  # (batch_size, L*L)

            dmask = adj.view(B, -1)  # (batch_size, L*L)
            rel_adj_logits = F.softmax(
                mask_logits(rel_adj_logits, dmask), dim=1)
            rel_adj_logits = rel_adj_logits.view(
                *rel_adj.size())  # (batch_size, L, L)

            if l == 0:
                Ax = rel_adj_logits.bmm(feature) # (batch_size, L, D)
            else:
                Ax = rel_adj_logits.bmm(Ax) # (batch_size, L, D)
            Ax = self.dropout(Ax) if l < self.num_layers - 1 else Ax

        return Ax


class GAT(nn.Module):
    """
    GAT module operated on graphs
    """

    def __init__(self, args, in_dim, hidden_size=64, mem_dim=300, num_layers=3):
        super(GAT, self).__init__()
        self.args = args
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.dropout = nn.Dropout(args.gcn_dropout)
        self.leakyrelu = nn.LeakyReLU(1e-2)

        # Standard GAT:attention over feature
        a_layers = [
            nn.Linear(2 * mem_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)]
        self.afcs = nn.Sequential(*a_layers)

        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(num_layers):
            input_dim = self.in_dim if layer == 0 else mem_dim
            self.W.append(nn.Linear(input_dim, mem_dim))

    def forward(self, adj,  feature):
        B, L = adj.size(0), adj.size(1)
        dmask = adj.view(B, -1)  # (batch_size, L*L)
        # gcn layer
        for l in range(self.num_layers):
            # Standard GAT:attention over feature
            #####################################
            if l == 0:
                h = self.W[l](feature) #(B, L, D) D=men_dim
            else:
                h = self.W[l](h) #(B, L, D) D=men_dim
            a_input = torch.cat([h.repeat(1, 1, L).view(
                B, L*L, -1), h.repeat(1, L, 1)], dim=2)  # (B, L*L, 2*D)
            # print("the {} time a_input's shape is {}".format(l,a_input.shape))
            e = self.leakyrelu(self.afcs(a_input)).squeeze(2)  # (B, L*L)
            attention = F.softmax(mask_logits(e, dmask), dim=1)
            attention = attention.view(*adj.size())  # (B, L, L)

            # original gat
            h = attention.bmm(h)  # (B, L, D)
            # print("the {} time after attention h's shape is {}".format(l,h.shape))
            h = self.dropout(h) if l < self.num_layers - 1 else h
            #####################################

        return h #(B,L,D) D=mem_dim


class GCN(nn.Module):
    """ 
    GCN module operated on graphs
    """

    def __init__(self, args, in_dim, mem_dim, num_layers):
        super(GCN, self).__init__()
        self.args = args
        self.in_dim = in_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(args.gcn_dropout)
        # gcn layer
        self.W = nn.ModuleList()

        for layer in range(num_layers):
            input_dim = self.in_dim if layer == 0 else mem_dim
            self.W.append(nn.Linear(input_dim, mem_dim))

    def conv_l2(self):
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def forward(self, adj, feature):
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)

        for l in range(self.num_layers):
            Ax = adj.bmm(feature) #(B,L,D) D=第一次为in_dim之后为men_dim
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](feature)  # self loop
            AxW /= denom

            # gAxW = F.relu(AxW)
            gAxW = AxW
            feature = self.dropout(gAxW) if l < self.num_layers - 1 else gAxW
        return feature, mask  #feature: #(B,L,D) D=men_dim