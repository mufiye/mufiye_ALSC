"""
Prepare the word vector for the dataset
from https://github.com/sunkaikai/Covolution_over_Dependency_Tree_EMNLP2019
"""

import json
import argparse
import numpy as np
from collections import Counter
import pickle

VOCAB_PREFIX = ['[PAD]','[UNK]']

def parse_args():
    parser = argparse.ArgumentParser('Prepare vocab for dataset.')
    parser.add_argument('--dataset', help='TACRED directory.')
    parser.add_argument('--lower', default=True, help='If specified, lowercase all words.')
    parser.add_argument('--wv_file', default='glove.840B.300d.txt', help='Glove vector file')
    parser.add_argument('--wv_dim', type=int, default=300, help='word vector dimension.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # 待修改
    train_file = '../datasets/'+args.dataset+'/train.json'
    test_file = '../datasets/'+args.dataset+'/test.json'

    # 加载tokens
    print("loading tokens...")
    train_tokens, train_pos, train_deprel, train_max_len = load_tokens(train_file)
    test_tokens, test_pos, test_deprel, test_max_len = load_tokens(test_file)
    if args.lower:
        train_tokens, test_tokens = \
        [[t.lower() for t in tokens] for tokens in (train_tokens, test_tokens)]

    '''
    question: 关于pkl和npy文件格式的读写
    '''
    vocab_file = '../datasets/'+args.dataset+'/vocab.pkl'
    emb_file = '../datasets/'+args.dataset+'/embedding.npy'

    # 读取glove文件中的词向量
    print("loading glove words...")
    glove_vocab = load_glove_vocab(args.wv_file, args.wv_dim)
    print("{} words loaded from glove.".format(len(glove_vocab)))

    # 构建词表
    print("building vocab...")
    v = build_vocab(train_tokens+test_tokens, glove_vocab)

    # 查看数据中出现的但是词表中没有的词
    print("calculating oov...")
    datasets = {'train': train_tokens, 'test': test_tokens}
    for dname, d in datasets.items():
        total, oov = count_oov(d, v)
        print("{} oov: {}/{} ({:.2f}%)".format(dname, oov, total, oov*100.0/total))

    # 构建词向量
    print("building embeddings...")
    embedding = build_embedding(args.wv_file, v, args.wv_dim)
    print("embedding size: {} x {}".format(*embedding.shape))

    # 保存词表和词向量
    print("dumping to files...")
    with open(vocab_file, 'wb') as outfile:
        pickle.dump(v, outfile)
    np.save(emb_file, embedding)

    # 存储deprel和pos信息(本来还有个post，但是好像没啥用，不过可以先处理了)
    print('saving the dicts...')
    ret = dict()
    pos_list = VOCAB_PREFIX+list(set(train_pos+test_pos))
    pos_dict = {pos_list[i]:i for i in range(len(pos_list))}
    deprel_list = VOCAB_PREFIX+list(set(train_deprel+test_deprel))
    deprel_dict = {deprel_list[i]:i for i in range(len(deprel_list))}
    max_len = max(train_max_len, test_max_len)
    postion_list = VOCAB_PREFIX+list(range(-max_len, max_len))
    postion_dict = {postion_list[i]:i for i in range(len(postion_list))}
    ret['pos'] = pos_dict
    ret['deprel'] = deprel_dict
    ret['postion'] = postion_dict  # 与aspect词的相对距离
    ret['polarity'] = {'positive':0, 'negative':1, 'neutral':2}
    open('../datasets/'+args.dataset+'/constant.py', 'w').write(str(ret))

    print("all done.")


'''
desciption: 读取数据集文件中的token,pos和deprel
'''
def load_tokens(filename):
    with open(filename) as infile:
        data = json.load(infile)
        tokens = []
        pos = []  # 词性
        deprel = []  # 依赖关系
        max_len = 0
        for d in data:
            tokens.extend(d['token'])
            pos.extend(d['pos'])
            deprel.extend(d['deprel'])
            max_len = max(len(d['token']),max_len)
    print("{} tokens from {} examples loaded from {}.".format(len(tokens), len(data), filename))
    return tokens, list(set(pos)), list(set(deprel)), max_len

'''
desciption: 
# 读取Glove文件中的token
# Glove文件的构成，每行为token加上word vector
'''
def load_glove_vocab(filename, wv_dim):
    vocab = set()
    with open('../datasets/Glove/'+filename, encoding='utf-8') as f:
        for line in f:
            elems = line.split()
            token = ''.join(elems[0:-wv_dim])
            vocab.add(token)
    return vocab

'''
desciption: 构建词表
'''
def build_vocab(tokens, glove_vocab):
    counter = Counter(t for t in tokens)
    v = sorted([t for t in counter if t in glove_vocab], key=counter.get, reverse=True)
    v = VOCAB_PREFIX + v
    print("vocab built with {}/{} words.".format(len(v), len(counter)))
    return v

'''
desciption: 构建词向量矩阵
'''
def build_embedding(wv_file, vocab, wv_dim):
    vocab_size = len(vocab)
    emb = np.random.uniform(-1, 1, (vocab_size, wv_dim))
    emb[0] = 0  # pad vector
    w2id = {w:i  for i,w in enumerate(vocab)}
    with open('../datasets/Glove/'+wv_file, encoding='utf-8') as f:
        for line in f:
            elems = line.split()
            token = ''.join(elems[0:-wv_dim])
            if token in w2id:
                emb[w2id[token]] = [float(v) for v in elems[-wv_dim:]]
    return emb

'''
desciption: 查看数据集中出现但是词表中没有出现的词
'''
def count_oov(tokens, vocab):
    c = Counter(t for t in tokens)
    total = sum(c.values())
    matched = sum(c[t] for t in vocab)
    return total, total-matched

if __name__ == '__main__':
    main()
