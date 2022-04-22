'''
# the file is for infering
# the process:
1. 读取数据(a sentence and some aspects)
2. 进行这句话的语法解析
3. 将数据传入模型之中
4. 模型处理数据
5. 返回结果
'''

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import logging

# changed
# from load_data import get_rolled_and_unrolled_data, load_and_cache_vocabs_vectors
# from models.model import (Aspect_Text_GAT_ours,
#                     Pure_Bert, Aspect_Bert_GAT, Aspect_Text_GAT_only)
from new_load_data import get_rolled_and_unrolled_data, load_and_cache_vocabs_vectors
from models.new_model import *

from transformers import (BertConfig, BertForTokenClassification,
                                  BertTokenizer)

import json
import spacy
from spacy.tokens import Doc
from spacy.matcher import Matcher

logger = logging.getLogger(__name__)

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

nlp = spacy.load('en_core_web_sm')
# 使用这个分词器就是不区分标点符号了，我们还是要区分一下
# nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
matcher = Matcher(nlp.vocab)

class Inferer:
    def __init__(self, opt):
        self.opt = opt
        # 根据model class以及opt参数设置model
        if self.opt.model_choice == 1:
            self.word_vecs, self.word_vocab, self.dep_tag_vocab, self.pos_tag_vocab = load_and_cache_vocabs_vectors(None, self.opt)
            embedding = torch.from_numpy(np.asarray(self.word_vecs, dtype=np.float32))
            self.opt.glove_embedding = embedding
            self.model = self.opt.model_class(self.opt, self.dep_tag_vocab['len'])
            self.model.load_state_dict(torch.load(self.opt.state_dict_path,map_location=torch.device(self.opt.device)))
            self.model = self.model.to(self.opt.device)
        elif self.opt.model_choice == 2:
            self.word_vecs, self.word_vocab, self.dep_tag_vocab, self.pos_tag_vocab = load_and_cache_vocabs_vectors(None, self.opt)
            self.model = self.opt.model_class(self.opt, self.dep_tag_vocab['len'])
            self.model.load_state_dict(torch.load(self.opt.state_dict_path,map_location=torch.device(self.opt.device)))
            self.model = self.model.to(self.opt.device)
            tokenizer = BertTokenizer.from_pretrained(opt.bert_model_dir)
            self.opt.tokenizer = tokenizer

        # switch model to evaluation mode
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

    
    def evaluate(self, text, aspect):
        # 1. 语法解析并将数据变为json数据集的格式
        # (暂时先用spacy库的语法解析方法，未来或许可以替换为别的更好的方法)
        json_data = []
        doc = nlp(text)
        doc_aspect = nlp(aspect)
        sentence_info = {}
        sentence_info['sentence'] = text
        sentence_info['aspect_sentiment'] = []
        sentence_info['tokens'] = []
        sentence_info['tags'] = []
        sentence_info['predicted_dependencies'] = []
        sentence_info['dependencies'] = []
        sentence_info['predicted_heads'] = []
        # 做预测时情感极性标签是不知道的，但还是要填进去
        sentence_info['aspect_sentiment'].append([aspect,'unknown'])

        pattern = []
        for token in doc_aspect:
            pattern.append({'TEXT': token.text})
        matcher.add('mufiye_pattern', [pattern])
        matches = matcher(doc)
        for match_id, start, end in matches:
            sentence_info['from_to'] = []
            sentence_info['from_to'].append([start,end])
        for token in doc:
            sentence_info['tokens'].append(token.text)
            sentence_info['tags'].append(token.tag_)
            sentence_info['predicted_dependencies'].append(token.dep_.lower())
            if token.dep_.lower() == 'root':
                sentence_info['predicted_heads'].append(0)
                sentence_info['dependencies'].append([token.dep_.lower(),0,token.i+1])
            else:
                sentence_info['predicted_heads'].append(token.head.i+1)
                sentence_info['dependencies'].append([token.dep_.lower(),token.head.i+1,token.i+1])
            
        json_data.append(sentence_info)
        #####################################
        # 测试代码1：写入到文件中看看
        # with open('infer_json_data.json', 'w') as f:
        #     json.dump(json_data, f)
        # print('done', len(json_data))
        #####################################

        # 2. 将json样式的数据转换为model的输入数据的样式(参考load_data)
        # 不需要load_data中的get_dataset()
        # 下面主要的是convert_feature()函数
        _, unrolled_data, _, _ = get_rolled_and_unrolled_data(json_data, self.opt)
        #####################################
        # 测试代码2：打印显示unrolled_data
        # print(unrolled_data)
        # print(type(unrolled_data))
        #####################################
        unrolled_data = unrolled_data[0]
        if self.opt.embedding_type == 'glove':
            unrolled_data['sentence_ids'] = [self.word_vocab['stoi'][w]
                                                for w in unrolled_data['sentence']]
            unrolled_data['aspect_ids'] = [self.word_vocab['stoi'][w]
                                              for w in unrolled_data['aspect']]
        elif self.opt.embedding_type == 'bert':
            cls_token = "[CLS]"
            sep_token = "[SEP]"
            pad_token = 0
            tokens = []
            word_indexer = []
            aspect_tokens = []
            aspect_indexer = []
            for word in unrolled_data['sentence']:
                word_tokens = self.opt.tokenizer.tokenize(word)
                token_idx = len(tokens)
                tokens.extend(word_tokens)
                word_indexer.append(token_idx)
            for word in unrolled_data['aspect']:
                word_aspect_tokens = self.opt.tokenizer.tokenize(word)
                token_idx = len(aspect_tokens)
                aspect_tokens.extend(word_aspect_tokens)
                aspect_indexer.append(token_idx)
            tokens = [cls_token] + tokens + [sep_token]
            aspect_tokens = [cls_token] + aspect_tokens + [sep_token]
            word_indexer = [i+1 for i in word_indexer]
            aspect_indexer = [i+1 for i in aspect_indexer]
            input_ids = self.opt.tokenizer.convert_tokens_to_ids(tokens)
            input_aspect_ids = self.opt.tokenizer.convert_tokens_to_ids(aspect_tokens)
            assert len(word_indexer) == len(unrolled_data['sentence'])
            assert len(aspect_indexer) == len(unrolled_data['aspect'])
            
            input_cat_ids = input_ids + input_aspect_ids[1:]
            segment_ids = [0] * len(input_ids) + [1] * len(input_aspect_ids[1:])

            unrolled_data['input_cat_ids'] = input_cat_ids
            unrolled_data['segment_ids'] = segment_ids
            unrolled_data['input_ids'] = input_ids
            unrolled_data['word_indexer'] = word_indexer
            unrolled_data['input_aspect_ids'] = input_aspect_ids
            unrolled_data['aspect_indexer'] = aspect_indexer
        # 共用的convert_feature部分
        unrolled_data['text_len'] = len(unrolled_data['sentence'])
        unrolled_data['aspect_position'] = [0] * unrolled_data['text_len']
        try:  # find the index of aspect in sentence
            for j in range(unrolled_data['from'], unrolled_data['to']):
                unrolled_data['aspect_position'][j] = 1
        except:
            for term in unrolled_data['aspect']:
                unrolled_data['aspect_position'][unrolled_data['sentence'].index(term)] = 1
        unrolled_data['dep_tag_ids'] = [self.dep_tag_vocab['stoi'][w]
                                           for w in unrolled_data['predicted_dependencies']]
        unrolled_data['pos_class'] = [self.pos_tag_vocab['stoi'][w]
                                             for w in unrolled_data['tags']]
        unrolled_data['aspect_len'] = len(unrolled_data['aspect'])
        
        # 3.关于普通数据到tensor的转换(__getitem__函数，还有allocate函数有关系吗？)
        items = unrolled_data['dep_tag_ids'],unrolled_data['pos_class'], unrolled_data['text_len'],\
            unrolled_data['aspect_len'], unrolled_data['sentiment'],\
            unrolled_data['predicted_heads'], unrolled_data['aspect_position']
        if self.opt.embedding_type == 'glove':
            non_bert_items = unrolled_data['sentence_ids'], unrolled_data['aspect_ids']
            items_tensor = non_bert_items + items
            items_tensor = tuple(torch.unsqueeze(torch.tensor(t),0) for t in items_tensor)
        elif self.opt.embedding_type == 'bert':
            bert_items = unrolled_data['input_ids'], unrolled_data['word_indexer'], unrolled_data['input_aspect_ids'], unrolled_data['aspect_indexer'], unrolled_data['input_cat_ids'], unrolled_data['segment_ids']
            items_tensor = tuple(torch.unsqueeze(torch.tensor(t),0) for t in bert_items)
            items_tensor += tuple(torch.unsqueeze(torch.tensor(t),0) for t in items)
        #####################################
        # 测试代码4：关于pytorch的tensor
        #####################################
        # 4.将数据输入模型并返回结果
        if self.opt.embedding_type == 'glove':
            inputs = {  'sentence': items_tensor[0],
                        'aspect': items_tensor[1], # aspect token
                        'dep_tags': items_tensor[2], # reshaped
                        'pos_class': items_tensor[3],
                        'text_len': items_tensor[4],
                        'aspect_len': items_tensor[5],
                        'dep_heads': items_tensor[7],
                        'aspect_position': items_tensor[8]}
        else:
            inputs = {  'input_ids': items_tensor[0],
                        'input_aspect_ids': items_tensor[2],
                        'word_indexer': items_tensor[1],
                        'aspect_indexer': items_tensor[3],
                        'input_cat_ids': items_tensor[4],
                        'segment_ids': items_tensor[5],
                        'dep_tags': items_tensor[6],
                        'pos_class': items_tensor[7],
                        'text_len': items_tensor[8],
                        'aspect_len': items_tensor[9],
                        'dep_heads': items_tensor[11],
                        'aspect_position': items_tensor[12]}
        t_outputs = self.model(**inputs)
        t_probs = F.softmax(t_outputs, dim=-1).cpu().numpy()

        return t_probs

'''
refer from https://github.com/songyouwei/ABSA-PyTorch/blob/master/infer_example.py
'''

def infering(text, aspect, model_choice=1, dataset_choice=1,use_cuda=False):
    # setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    # 已补全: model_classes, 指模型名称及其对应的class
    model_classes = {
        'gat_noMix': No_Mix_GAT_our,
        'gat_bert': No_Reshaped_GAT_Bert,
    }

    # 已补全: Option对象
    class Option(object): pass
    opt=Option()
    opt.inInfer = True
    opt.model_choice = model_choice
    if dataset_choice == 1:
        opt.dataset_name = 'rest'
    elif dataset_choice == 2:
        opt.dataset_name = 'laptop'
    else:
        opt.dataset_name = 'twitter'
    if use_cuda:
        opt.if_cuda = True
    else:
        opt.if_cuda = False
    if opt.model_choice == 1:
        # about model parameters
        opt.model_class = model_classes['gat_noMix']
        opt.highway = True
        opt.embedding_type = 'glove'
        opt.glove_dir = './datasets/Glove'
        opt.num_gcn_layers = 2
        opt.embedding_dim = 300
        opt.dep_relation_embed_dim = 300
        opt.hidden_size = 100
        opt.final_hidden_size = 100
        opt.num_classes = 3
        if opt.dataset_name == 'rest':
            opt.state_dict_path = 'saved_models/noMix_model/gat_noMix_our_rest.pth'
            opt.num_layers = 1
            opt.num_mlps = 2
            opt.dropout = 0.8
            opt.gcn_dropout = 0.2
        elif opt.dataset_name == 'laptop':
            opt.state_dict_path = 'saved_models/noMix_model/gat_noMix_our_laptop.pth'
            opt.num_layers = 2
            opt.num_mlps = 3
            opt.dropout = 0.7
            opt.gcn_dropout = 0.8
        elif opt.dataset_name == 'twitter':
            opt.state_dict_path = 'saved_models/noMix_model/gat_noMix_our_twitter.pth'
            opt.num_layers = 1
            opt.num_mlps = 2
            opt.dropout = 0.5
            opt.gcn_dropout = 0.0
    elif opt.model_choice == 2:
        # about model parameters
        opt.model_class = model_classes['gat_bert']
        opt.highway = False
        opt.embedding_type = 'bert'
        opt.bert_model_dir = 'models/bert_base'
        opt.num_gcn_layers = 2
        opt.dep_relation_embed_dim = 300
        opt.final_hidden_size = 256
        opt.num_classes = 3
        if opt.dataset_name == 'rest':
            opt.state_dict_path = 'saved_models/bert_base_model/gat_bert_base_rest.pth'
            opt.num_mlps = 2
            opt.dropout = 0.3
            opt.gcn_dropout = 0.3
        elif opt.dataset_name == 'laptop':
            opt.state_dict_path = 'saved_models/bert_base_model/gat_bert_base_laptop.pth'
            opt.num_mlps = 3
            opt.dropout = 0.0
            opt.gcn_dropout = 0.5
        elif opt.dataset_name == 'twitter':
            opt.state_dict_path = 'saved_models/bert_base_model/gat_bert_base_twitter.pth'
            opt.num_mlps = 2
            opt.dropout = 0.0
            opt.gcn_dropout = 0.2

    # 关于用于infer的设备
    device = torch.device('cuda' if opt.if_cuda and torch.cuda.is_available() else 'cpu')    
    opt.device = device
    logger.info('Device is %s', opt.device)

    # 测试代码1：测试语法解析是否正确
    # inf.evaluate('Two wasted steaks -- what a crime!', 'steaks')

    inf = Inferer(opt)
    # {'negative': 0, 'positive': 1, 'neutral': 2}
    # 第一组测试：正确答案为negative
    # t_probs = inf.evaluate('the service is terrible', 'service')
    # print(t_probs.argmax(axis=-1))

    # 第二组测试：正确答案为negative（有过一个报错，注意单引号和双引号）
    # t_probs = inf.evaluate("It sounds like there may be a replacement for charlie sheen on '' Two and a Half Men . '' What do you think ? .", "charlie sheen")
    # print(t_probs.argmax(axis=-1))

    # 第三组测试：正确答案为neural
    # t_probs = inf.evaluate("wtf ? hilary swank is coming to my school today , just to chill . lol wow", "hilary swank")
    # print(t_probs.argmax(axis=-1))

    # 第四组测试：正确答案为positive
    # t_probs = inf.evaluate("ok love game is on . . lady gaga , my dear , you need new music ; -RRB-", "lady gaga")
    # print(t_probs.argmax(axis=-1))

    # 第五组测试：正确答案为positive
    t_probs = inf.evaluate(text, aspect)
    return t_probs.argmax(axis=-1).item()


if __name__ == '__main__':
    # {'negative': 0, 'positive': 1, 'neutral': 2}
    # twitter case
    # positive
    text_rest1 = "The bread is top notch as well."
    aspect_rest1 = "bread"
    # negative
    text_rest2 = "In fact, this was not a Nicoise salad and was barely eatable."
    aspect_rest2 = "Nicoise salad"
    # neutral
    text_rest3 = "The pizza is the best if you like thin crusted pizza."
    aspect_rest3 = "thin crusted pizza"

    # laptop case
    # positive
    text_laptop1 = "Boot time is super fast, around anywhere from 35 seconds to 1 minute."
    aspect_laptop1 = "Boot time"
    # negative
    text_laptop2 = "I can barely use any usb devices because they will not stay connected properly."
    aspect_laptop2 = "usb devices"
    # neutral
    text_laptop3 = "I took it back for an Asus and same thing- blue screen which required me to remove the battery to reset."
    aspect_laptop3 = "battery"

    # twitter case
    # positive
    text_twitter1 = "3 by britney spears is an amazing song"
    aspect_twitter1 = "britney spears"
    # negative
    text_twitter2 = "God damn . That Sony remote for google is fucking hideeeeeous !"
    aspect_twitter2 = "google"
    # neutral
    text_twitter3 = "my 3-year-old was amazed yesterday to find that ' real ' 10 pin bowling is nothing like it is on the wii ..."
    aspect_twitter3 = "wii"

    # model choice为1或2, dataset choice为1~3
    print("---------------------------------------------------")
    print("for test rest noMix model")
    result = infering(text_rest1, aspect_rest1, 1, 1, False)
    print("the result is: {}".format(result))
    result = infering(text_rest2, aspect_rest2, 1, 1, False)
    print("the result is: {}".format(result))
    result = infering(text_rest3, aspect_rest3, 1, 1, False)
    print("the result is: {}".format(result))
    print("---------------------------------------------------")

    print("---------------------------------------------------")
    print("for test laptop noMix model")
    result = infering(text_laptop1, aspect_laptop1, 1, 2, False)
    print("the result is: {}".format(result))
    result = infering(text_laptop2, aspect_laptop2, 1, 2, False)
    print("the result is: {}".format(result))
    result = infering(text_laptop3, aspect_laptop3, 1, 2, False)
    print("the result is: {}".format(result))
    print("---------------------------------------------------")

    print("---------------------------------------------------")
    print("for test twitter noMix model")
    result = infering(text_twitter1, aspect_twitter1, 1, 3, False)
    print("the result is: {}".format(result))
    result = infering(text_twitter2, aspect_twitter2, 1, 3, False)
    print("the result is: {}".format(result))
    result = infering(text_twitter3, aspect_twitter3, 1, 3, False)
    print("the result is: {}".format(result))
    print("---------------------------------------------------")

    print("---------------------------------------------------")
    print("for test rest bert-base model")
    result = infering(text_rest1, aspect_rest1, 2, 1, False)
    print("the result is: {}".format(result))
    result = infering(text_rest2, aspect_rest2, 2, 1, False)
    print("the result is: {}".format(result))
    result = infering(text_rest3, aspect_rest3, 2, 1, False)
    print("the result is: {}".format(result))
    print("---------------------------------------------------")
    
    print("---------------------------------------------------")
    print("for test laptop bert-base model")
    result = infering(text_laptop1, aspect_laptop1, 2, 2, False)
    print("the result is: {}".format(result))
    result = infering(text_laptop2, aspect_laptop2, 2, 2, False)
    print("the result is: {}".format(result))
    result = infering(text_laptop3, aspect_laptop3, 2, 2, False)
    print("the result is: {}".format(result))
    print("---------------------------------------------------")

    print("---------------------------------------------------")
    print("for test twitter bert-base model")
    result = infering(text_twitter1, aspect_twitter1, 2, 3, False)
    print("the result is: {}".format(result))
    result = infering(text_twitter2, aspect_twitter2, 2, 3, False)
    print("the result is: {}".format(result))
    result = infering(text_twitter3, aspect_twitter3, 2, 3, False)
    print("the result is: {}".format(result))
    print("---------------------------------------------------")


