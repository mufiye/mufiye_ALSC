import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.model_gcn import GAT, GCN, Rel_GAT
from models.model_utils import LinearAttention, DotprodAttention, RelationAttention, Highway, mask_logits
from models.tree import *

class No_Reshaped_GAT_ours(nn.Module):
    def __init__(self, args, dep_rel_num):
        super(No_Reshaped_GAT_ours, self).__init__()
        self.args = args

        num_embeddings, embed_dim = args.glove_embedding.shape
        self.embed = nn.Embedding(num_embeddings, embed_dim)
        self.embed.weight = nn.Parameter(
            args.glove_embedding, requires_grad=False)

        self.dropout = nn.Dropout(args.dropout)
        self.tanh = nn.Tanh()

        if args.highway:
            self.highway_dep = Highway(args.num_layers, args.embedding_dim)
            self.highway = Highway(args.num_layers, args.embedding_dim)

        self.bilstm = nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_size,
                              bidirectional=True, batch_first=True, num_layers=args.num_layers)
        gcn_input_dim = args.hidden_size * 2

        # the changed place
        self.rel_gat = Rel_GAT(args, dep_rel_num = dep_rel_num, num_layers = args.num_gcn_layers).to(args.device)
        self.gat = GAT(args, in_dim = gcn_input_dim, mem_dim = gcn_input_dim, num_layers = args.num_gcn_layers).to(args.device)
        # self.aspect_attention = DotprodAttention().to(args.device)

        # the last mlp part
        last_hidden_size = args.hidden_size * 4
        layers = [
            nn.Linear(last_hidden_size, args.final_hidden_size), nn.ReLU()]
        for _ in range(args.num_mlps-1):
            layers += [nn.Linear(args.final_hidden_size,
                                 args.final_hidden_size), nn.ReLU()]
        self.fcs = nn.Sequential(*layers)
        self.fc_final = nn.Linear(args.final_hidden_size, args.num_classes)

    def forward(self, sentence, aspect, pos_class, dep_tags, text_len, aspect_len, dep_heads, aspect_position):
        '''
        Forward takes:
            sentence: sentence_id of size (batch_size, text_length)
            aspect: aspect_id of size (batch_size, aspect_length)
            pos_class: pos_tag_id of size (batch_size, text_length)
            dep_tags: dep_tag_id of size (batch_size, text_length)
            text_len: (batch_size,) length of each sentence
            aspect_len: (batch_size, ) aspect length of each sentence
            dep_heads: (batch_size, text_length) which node adjacent to that node
            aspect_position: (batch_size, text_length) mask, with the position of aspect as 1 and others as 0
        '''
        fmask = (torch.zeros_like(sentence) != sentence).float()  # (N，L)
        feature = self.embed(sentence)  # (N, L, D)
        # aspect_feature = self.embed(aspect) # (N, L', D)

        feature = self.dropout(feature)
        # aspect_feature = self.dropout(aspect_feature)

        if self.args.highway:
            feature = self.highway(feature)
            # aspect_feature = self.highway(aspect_feature)

        feature, _ = self.bilstm(feature) # (N,L,D)
        # aspect_feature, _ = self.bilstm(aspect_feature) #(N,L,D)

        # aspect_feature = aspect_feature.mean(dim = 1) #(N,D)

        #########################################################
        # do thing about gat, the part changed
        adj, rel_adj = inputs_to_deprel_adj(self.args, dep_heads, dep_tags, text_len)
        adj = adj.to(self.args.device)
        rel_adj = rel_adj.to(self.args.device)

        #########################################################
        # 目前的想法是将gat经过gcn与aspect attention机制的feature与
        # rel_gat的feature进行拼接
        # 待验证正确性、可行性和精确度
        #########################################################
        asp_wn = aspect_position.sum(dim=1).unsqueeze(-1) #(N,L)

        feature_dep = self.rel_gat(adj, rel_adj, feature) #(N, L, D)
        mask_dep = aspect_position.unsqueeze(-1).repeat(1,1,self.args.hidden_size * 2) #(N,L,D)
        feature_dep = (feature_dep*mask_dep).sum(dim=1) / asp_wn #(N,1,D)
        print("feature_dep's shape is {}".format(feature_dep.shape))

        feature_word = self.gat(adj, feature)  #(N, L, D)
        mask_word = aspect_position.unsqueeze(-1).repeat(1,1,self.args.hidden_size * 2) #(N,L,D)
        feature_word = (feature_word*mask_word).sum(dim=1) / asp_wn #(N,1,D)
        print("feature_word's shape is {}".format(feature_word.shape))
        
        #########################################################
        # maybe cat is useful before final mlp part
        feature_out = torch.cat([feature_word, feature_dep], dim = 1) #(N, D') D'=2D
        x = self.dropout(feature_out)
        x = self.fcs(x)
        logit = self.fc_final(x)
        return logit


# the bert base model
# class No_Reshaped_GAT_Bert(nn.Module):
#     def __init__(self, args, dep_rel_num):
#     def forward():

