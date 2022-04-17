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
        # self.gcn = GCN(args, in_dim = gcn_input_dim, mem_dim = gcn_input_dim, num_layers = args.num_gcn_layers).to(args.device)
        self.aspect_attention = DotprodAttention().to(args.device)
        self.dep_attention = RelationAttention(in_dim = gcn_input_dim).to(args.device)

        # the last mlp part
        # gat_nomask_concat
        last_hidden_size = args.hidden_size * 4

        # gat_nomask_noconcat
        # last_hidden_size = args.hidden_size * 2
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
        aspect_feature = self.embed(aspect) # (N, L', D)

        feature = self.dropout(feature)
        aspect_feature = self.dropout(aspect_feature)

        if self.args.highway:
            feature = self.highway(feature)
            aspect_feature = self.highway(aspect_feature)

        feature, _ = self.bilstm(feature) # (N,L,D)
        aspect_feature, _ = self.bilstm(aspect_feature) #(N,L,D)

        aspect_feature = aspect_feature.mean(dim = 1) #(N,D)

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
        # print("aspect position: {}".format(aspect_position))

        # gat_nomask_concat
        feature_dep = self.rel_gat(adj, rel_adj, feature) #(N, L, D)
        feature_dep = self.dep_attention(feature, feature_dep, fmask) #(N, D)
        feature_word = self.gat(adj, feature)  #(N, L, D)
        feature_word = self.aspect_attention(feature_word,aspect_feature,fmask) #(N, D)
        
        # gat_nomask_noconcat
        # feature_word = self.gat(adj, feature)  #(N, L, D)
        # feature_dep = self.rel_gat(adj, rel_adj, feature_word) #(N, L, D)
        # feature_out = self.aspect_attention(feature_dep,aspect_feature,fmask) #(N, D)

        #########################################################
        # gat_nomask_concat
        feature_out = torch.cat([feature_dep, feature_word], dim = 1) # (N, D')
        x = self.dropout(feature_out)
        x = self.fcs(x)
        logit = self.fc_final(x)
        return logit

class No_Reshaped_GAT_NonConcat(nn.Module):
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
        # self.gcn = GCN(args, in_dim = gcn_input_dim, mem_dim = gcn_input_dim, num_layers = args.num_gcn_layers).to(args.device)
        self.aspect_attention = DotprodAttention().to(args.device)

        # the last mlp part
        # gat_nomask_concat
        # last_hidden_size = args.hidden_size * 4

        # gat_nomask_noconcat
        last_hidden_size = args.hidden_size * 2
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
        aspect_feature = self.embed(aspect) # (N, L', D)

        feature = self.dropout(feature)
        aspect_feature = self.dropout(aspect_feature)

        if self.args.highway:
            feature = self.highway(feature)
            aspect_feature = self.highway(aspect_feature)

        feature, _ = self.bilstm(feature) # (N,L,D)
        aspect_feature, _ = self.bilstm(aspect_feature) #(N,L,D)

        aspect_feature = aspect_feature.mean(dim = 1) #(N,D)

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
        # print("aspect position: {}".format(aspect_position))

        # gat_nomask_concat
        # feature_dep = self.rel_gat(adj, rel_adj, feature) #(N, L, D)
        # feature_dep = self.aspect_attention(feature_dep,aspect_feature,fmask) #(N, D)
        # feature_word = self.gat(adj, feature)  #(N, L, D)
        # feature_word = self.aspect_attention(feature_word,aspect_feature,fmask) #(N, D)
        
        # gat_nomask_noconcat
        feature_word = self.gat(adj, feature)  #(N, L, D)
        feature_dep = self.rel_gat(adj, rel_adj, feature_word) #(N, L, D)
        feature_out = self.aspect_attention(feature_dep,aspect_feature,fmask) #(N, D)

        #########################################################
        # gat_nomask_concat
        # feature_out = torch.cat([feature_dep, feature_word], dim = 1) # (N, D')
        x = self.dropout(feature_out)
        x = self.fcs(x)
        logit = self.fc_final(x)
        return logit

class Pure_Bert(nn.Module):
    '''
    Bert for sequence classification.
    '''

    def __init__(self, args, hidden_size=256):
        super(Pure_Bert, self).__init__()

        config = BertConfig.from_pretrained(args.bert_model_dir)
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir)
        self.bert = BertModel.from_pretrained(
            args.bert_model_dir, config=config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        layers = [nn.Linear(
            config.hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, args.num_classes)]
        self.classifier = nn.Sequential(*layers)

    # input_ids, token_type_ids是什么？
    def forward(self, input_ids, token_type_ids):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids)
        # pool output is usually *not* a good summary of the semantic content of the input,
        # you're often better with averaging or poolin the sequence of hidden-states for the whole input sequence.
        pooled_output = outputs[1]
        # pooled_output = torch.mean(pooled_output, dim = 1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits

# # the using concat model with bert 
# class No_Reshaped_GAT_Bert(nn.Module):
#     def __init__(self, args, dep_tag_num):
#         super(No_Reshaped_GAT_Bert, self).__init__()
#         self.args = args

#         # Bert
#         config = BertConfig.from_pretrained(args.bert_model_dir)
#         self.bert = BertModel.from_pretrained(
#             args.bert_model_dir, config=config, from_tf =False)
#         self.dropout_bert = nn.Dropout(config.hidden_dropout_prob)
#         self.dropout = nn.Dropout(args.dropout)
#         args.embedding_dim = config.hidden_size  # 768

#         if args.highway:
#             self.highway_dep = Highway(args.num_layers, args.embedding_dim)
#             self.highway = Highway(args.num_layers, args.embedding_dim)
        
#         gcn_input_dim = args.embedding_dim

#         # about gat

#         last_hidden_size = args.embedding_dim * 2
#         layers = [
#             nn.Linear(last_hidden_size, args.final_hidden_size), nn.ReLU()]
#         for _ in range(args.num_mlps - 1):
#             layers += [nn.Linear(args.final_hidden_size,
#                                  args.final_hidden_size), nn.ReLU()]
#         self.fcs = nn.Sequential(*layers)
#         self.fc_final = nn.Linear(args.final_hidden_size, args.num_classes)
    

#     def forward(self, input_ids, input_aspect_ids, word_indexer, aspect_indexer,input_cat_ids,segment_ids, pos_class, dep_tags, text_len, aspect_len, dep_heads, aspect_position):
#         fmask = (torch.ones_like(word_indexer) != word_indexer).float()  # (N，L)
#         fmask[:,0] = 1
#         outputs = self.bert(input_cat_ids, token_type_ids = segment_ids)
#         feature_output = outputs[0] # (N, L, D)
#         pool_out = outputs[1] #(N, D)

#         # index select, back to original batched size.
#         feature = torch.stack([torch.index_select(f, 0, w_i)
#                                for f, w_i in zip(feature_output, word_indexer)])

#         # do thing about gat, the part changed
#         adj, rel_adj = inputs_to_deprel_adj(self.args, dep_heads, dep_tags, text_len)
#         adj = adj.to(self.args.device)
#         rel_adj = rel_adj.to(self.args.device)

#         # do gat thing
#         feature_dep = self.rel_gat(adj, rel_adj, feature) #(N, L, D)
#         feature_dep = self.aspect_attention(feature_dep,aspect_feature,fmask) #(N, D)

#         #########################################################
#         # gat_nomask_concat
#         feature_out = torch.cat([feature_dep, pool_out], dim = 1) # (N, D')
#         x = self.dropout(feature_out)
#         x = self.fcs(x)
#         logit = self.fc_final(x)
#         return logit