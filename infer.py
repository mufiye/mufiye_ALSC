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
from load_data import get_rolled_and_unrolled_data, load_and_cache_vocabs_vectors
from models.model import (Aspect_Text_GAT_ours,
                    Pure_Bert, Aspect_Bert_GAT, Aspect_Text_GAT_only)

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
        if self.opt.modelName == 'gat_our' or self.opt.modelName == 'gat_only':
            self.word_vecs, self.word_vocab, self.dep_tag_vocab, self.pos_tag_vocab = load_and_cache_vocabs_vectors(None, self.opt)
            embedding = torch.from_numpy(np.asarray(self.word_vecs, dtype=np.float32))
            self.opt.glove_embedding = embedding
            self.model = self.opt.model_class(self.opt, self.dep_tag_vocab['len'], self.pos_tag_vocab['len'])
            self.model.load_state_dict(torch.load(self.opt.state_dict_path,map_location=torch.device(self.opt.device)))
            self.model = self.model.to(self.opt.device)
        elif self.opt.modelName == 'gat_bert':
            self.word_vecs, self.word_vocab, self.dep_tag_vocab, self.pos_tag_vocab = load_and_cache_vocabs_vectors(None, self.opt)
            self.model = self.opt.model_class(self.opt, self.dep_tag_vocab['len'], self.pos_tag_vocab['len'])
            self.model.load_state_dict(torch.load(self.opt.state_dict_path,map_location=torch.device(self.opt.device)))
            self.model = self.model.to(self.opt.device)
        else: #pure_bert case
            self.model = self.opt.model_class(self.opt)
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

        # 2. 将json样式的数据转换为model的输入数据的样式(按流程照搬照抄就行)
        # 不需要load_data中的get_dataset()
        _, unrolled_data, _, _ = get_rolled_and_unrolled_data(json_data, self.opt)
        if self.opt.embedding_type == 'glove':
            
        elif self.opt.embedding_type == 'bert':
        return


# parser是为了区分不同的模型，以及是否使用cuda
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gat_our', action='store_true',
                        help='GAT_our')
    parser.add_argument('--pure_bert', action='store_true',
                        help='Cat text and aspect, [cls] to predict.')
    parser.add_argument('--gat_bert', action='store_true',
                        help='Cat text and aspect, [cls] to predict.')
    parser.add_argument('--gat_ouly', action='store_true',
                        help='GAT_only')
    parser.add_argument('--if_cuda', default=True, action='store_false',
                        help='about use cuda or not')
    return parser.parse_args()

# 打印这些参数
def check_args(args):
    logger.info(vars(args))

'''
refer from https://github.com/songyouwei/ABSA-PyTorch/blob/master/infer_example.py
'''

def main():
    # setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    args = parse_args()
    check_args(args)

    # 已补全: model_classes, 指模型名称及其对应的class
    model_classes = {
        'gat_our': Aspect_Text_GAT_ours,
        'gat_bert': Aspect_Bert_GAT,
        'pure_bert': Pure_Bert,
        'gat_only': Aspect_Text_GAT_only,
    }

    # 待补全: Option对象, 待补全
    class Option(object): pass
    opt=Option()
    opt.inInfer = True
    if args.gat_our:
        # about model parameters
        opt.modelName = 'gat_our'
        opt.model_class = model_classes['gat_our']
        opt.state_dict_path = 'saved_models/state_dict/best_model/gat_our_twitter_acc_0.7038_f1_0.6977'
        opt.highway = True
        opt.num_layers = 2
        opt.num_heads = 9
        opt.num_gcn_layers = 1
        opt.gcn_mem_dim = 300
        opt.embedding_type = 'glove'
        opt.embedding_dim = 300
        opt.dep_relation_embed_dim = 300
        opt.hidden_size = 300
        opt.final_hidden_size = 400
        opt.num_mlps = 1
        opt.gat_attention_type = 'dotprod'
        opt.dropout = 0.6
        opt.num_classes = 3
        # about dataset parameters
        opt.dataset_name = 'twitter'
        opt.glove_dir = './datasets/Glove'
        opt.add_non_connect = True
        opt.multi_hop = True
        opt.max_hop = 4
    elif args.gat_bert:
        # about model parameters
        opt.modelName = 'gat_bert'
        opt.model_class = model_classes['gat_bert']
        opt.state_dict_path = 'saved_models/state_dict/best_model/gat_bert_twitter_acc_0.7645_f1_0.7506'
        opt.highway = False
        opt.num_layers = 2
        opt.num_heads = 6
        opt.num_gcn_layers = 1
        opt.gcn_mem_dim = 300
        opt.embedding_type = 'bert'
        opt.dep_relation_embed_dim = 300
        opt.hidden_size = 200
        opt.final_hidden_size = 300
        opt.num_mlps = 2
        opt.gat_attention_type = 'dotprod'
        opt.bert_model_dir = 'models/bert_base'
        opt.dropout = 0.2
        opt.num_classes = 3
        # about dataset parameters
        opt.dataset_name = 'twitter'
        opt.add_non_connect = True
        opt.multi_hop = True
        opt.max_hop = 4
    # pure_bert与gat_only先不考虑
    elif args.pure_bert:
        opt.modelName = 'pure_bert'
        opt.model_class = model_classes['pure_bert']
    else:
        opt.modelName = 'gat_only'
        opt.model_class = model_classes['gat_only']

    # 关于用于infer的设备
    device = torch.device('cuda' if args.if_cuda and torch.cuda.is_available() else 'cpu')    
    opt.device = device
    logger.info('Device is %s', opt.device)

    # 测试代码1：测试语法解析是否正确
    # inf.evaluate('Two wasted steaks -- what a crime!', 'steaks')

    inf = Inferer(opt)
    t_probs = inf.evaluate('the service is terrible', 'service')
    # print(t_probs.argmax(axis=-1) - 1)

if __name__ == '__main__':
    main()

