import logging
import json
import os
import pickle
import linecache

from collections import Counter, defaultdict
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

import nltk
from nltk import word_tokenize
from nltk.tokenize import TreebankWordTokenizer

logger = logging.getLogger(__name__)

# 读取经过处理的数据集
def load_datasets_and_vocabs(args):
    train_data, test_data = get_dataset(args.dataset_name)

    # question: 关于rolled和unrolled
    _, train_all_unrolled, _, _ = get_rolled_and_unrolled_data(train_data, args)
    _, test_all_unrolled, _, _ = get_rolled_and_unrolled_data(test_data, args)

    logger.info('****** After unrolling ******')
    logger.info('Train set size: %s', len(train_all_unrolled))
    logger.info('Test set size: %s', len(test_all_unrolled))

    # Build word vocabulary(part of speech, dep_tag) and save pickles.
    word_vecs, word_vocab, dep_tag_vocab, pos_tag_vocab = load_and_cache_vocabs_vectors(
        train_all_unrolled+test_all_unrolled, args)
    if args.embedding_type == 'glove':
        embedding = torch.from_numpy(np.asarray(word_vecs, dtype=np.float32))
        args.glove_embedding = embedding
    
    train_dataset = ALSC_Dataset(
        train_all_unrolled, args, word_vocab, dep_tag_vocab, pos_tag_vocab)
    test_dataset = ALSC_Dataset(
        test_all_unrolled, args, word_vocab, dep_tag_vocab, pos_tag_vocab)

    return train_dataset, test_dataset, word_vocab, dep_tag_vocab, pos_tag_vocab

# 读取json文件
def read_json_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        return data

# 获取未经处理的数据集——json对象list
def get_dataset(dataset_name):
    rest_train = 'datasets/SemEval-2014/Restaurant/Restaurants_Train_v2_biaffine_depparsed_with_energy.json'
    rest_test = 'datasets/SemEval-2014/Restaurant/Restaurants_Test_Gold_biaffine_depparsed_with_energy.json'

    laptop_train = 'datasets/SemEval-2014/Laptop/Laptop_Train_v2_biaffine_depparsed.json'
    laptop_test = 'datasets/SemEval-2014/Laptop/Laptops_Test_Gold_biaffine_depparsed.json'

    twitter_train = 'datasets/Twitter/train_biaffine.json'
    twitter_test = 'datasets/Twitter/test_biaffine.json'

    ds_train = {'rest': rest_train,
                'laptop': laptop_train, 'twitter': twitter_train}
    ds_test = {'rest': rest_test,
               'laptop': laptop_test, 'twitter': twitter_test}

    train = list(read_json_data(ds_train[dataset_name]))
    logger.info('# Read %s Train set: %d', dataset_name, len(train))

    test = list(read_json_data(ds_test[dataset_name]))
    logger.info("# Read %s Test set: %d", dataset_name, len(test))
    return train, test

# 加载词表和词向量(如果不存在这两个文件，则保存词表和词向量文件)
def load_and_cache_vocabs_vectors(data, args):
    # pkl用于存储词表
    rest_dir = 'datasets/SemEval-2014/Restaurant'
    laptop_dir = 'datasets/SemEval-2014/Laptop'
    twitter_dir = 'datasets/Twitter'
    pkl_dirs = {
        'rest': rest_dir, 'laptop': laptop_dir, 'twitter': twitter_dir
    }
    pkls_path = os.path.join(pkl_dirs[args.dataset_name], 'pkls')
    if not os.path.exists(pkls_path):
        os.makedirs(pkls_path)
    if args.embedding_type == 'glove':
        cached_word_vocab_file = os.path.join( 
            pkls_path, 'cached_{}_{}_word_vocab.pkl'.format(args.dataset_name, args.embedding_type))
        if os.path.exists(cached_word_vocab_file):
            logger.info('Loading word vocab from %s', cached_word_vocab_file)
            with open(cached_word_vocab_file, 'rb') as f:
                word_vocab = pickle.load(f)
        else:
            logger.info('Creating word vocab from dataset %s',
                        args.dataset_name)
            word_vocab = build_text_vocab(data)
            logger.info('Word vocab size: %s', word_vocab['len'])
            logging.info('Saving word vocab to %s', cached_word_vocab_file)
            # 存储词表(word vocab)文件
            with open(cached_word_vocab_file, 'wb') as f:
                pickle.dump(word_vocab, f, -1)
        
        cached_word_vecs_file = os.path.join(pkls_path, 'cached_{}_{}_word_vecs.pkl'.format(
            args.dataset_name, args.embedding_type))
        if os.path.exists(cached_word_vecs_file):
            logger.info('Loading word vecs from %s', cached_word_vecs_file)
            with open(cached_word_vecs_file, 'rb') as f:
                word_vecs = pickle.load(f)
        else:
            logger.info('Creating word vecs from %s', args.glove_dir)
            word_vecs = load_glove_embedding(
                word_vocab['itos'], args.glove_dir, 0.25, args.embedding_dim)
            logger.info('Saving word vecs to %s', cached_word_vecs_file)
            # 存储词向量(word vector)文件
            with open(cached_word_vecs_file, 'wb') as f:
                pickle.dump(word_vecs, f, -1)

    # BERT have it's own vocab and embeddings.
    else: 
        word_vocab = None
        word_vecs = None

    # Build vocab of dependency tags
    cached_original_dep_tag_vocab_file = os.path.join(
        pkls_path, 'cached_{}_original_dep_tag_vocab.pkl'.format(args.dataset_name))
    if os.path.exists(cached_original_dep_tag_vocab_file):
        logger.info('Loading vocab of dependency tags from %s',
                    cached_original_dep_tag_vocab_file)
        with open(cached_original_dep_tag_vocab_file, 'rb') as f:
            dep_tag_vocab = pickle.load(f)
    else:
        logger.info('Creating vocab of dependency tags.')
        dep_tag_vocab = build_dep_tag_vocab(data, min_freq=0)
        logger.info('Saving dependency tags vocab, size: %s, to file %s',
                    dep_tag_vocab['len'], cached_original_dep_tag_vocab_file)
        with open(cached_original_dep_tag_vocab_file, 'wb') as f:
            pickle.dump(dep_tag_vocab, f, -1)
    
    # Build vocab of part of speech tags
    cached_pos_tag_vocab_file = os.path.join(
        pkls_path, 'cached_{}_pos_tag_vocab.pkl'.format(args.dataset_name))
    if os.path.exists(cached_pos_tag_vocab_file):
        logger.info('Loading vocab of part-of-speech tags from %s',
                    cached_pos_tag_vocab_file)
        with open(cached_pos_tag_vocab_file, 'rb') as f:
            pos_tag_vocab = pickle.load(f)
    else:
        logger.info('Creating vocab of part-of-speech tags.')
        pos_tag_vocab = build_pos_tag_vocab(data, min_freq=0)
        logger.info('Saving part-of-speech tags vocab, size: %s, to file %s',
                    pos_tag_vocab['len'], cached_pos_tag_vocab_file)
        with open(cached_pos_tag_vocab_file, 'wb') as f:
            pickle.dump(pos_tag_vocab, f, -1)
    
    return word_vecs, word_vocab, dep_tag_vocab, pos_tag_vocab

def _default_unk_index():
    return 1


def build_text_vocab(data, vocab_size=100000, min_freq=2):
    counter = Counter()
    for d in data:
        s = d['sentence']
        counter.update(s)
    itos = ['[PAD]','[UNK]']
    min_freq = max(min_freq, 1)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        itos.append(word)
    stoi = defaultdict(_default_unk_index)
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}


# POS vocab
def build_pos_tag_vocab(data, vocab_size=1000, min_freq=1):
    counter = Counter()
    for d in data:
        tags = d['tags']
        counter.update(tags)
    
    itos = ['<pad>']
    min_freq = max(min_freq, 1)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict()
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}

# question: is the unk needed?
# build vocab about dependency relationship
def build_dep_tag_vocab(data, vocab_size=1000, min_freq=0):
    counter = Counter()
    for d in data:
        tags = d['predicted_dependencies']
        counter.update(tags)
    
    itos = ['<pad>', '<unk>']
    min_freq = max(min_freq, 1)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        if word == '<pad>':
            continue
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict(_default_unk_index)
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}


# 读取glove embedding文件中的词向量信息
# question: uniform scale是什么？
def load_glove_embedding(word_list, glove_dir, uniform_scale, dimension_size):
    glove_words = []
    with open(os.path.join(glove_dir, 'glove.840B.300d.txt'), 'r', encoding='utf-8') as fopen:
        for line in fopen:
            glove_words.append(line.strip().split(' ')[0])
    word2offset = {w:i for i, w in enumerate(glove_words)}
    word_vectors = []
    for word in word_list:
        if word in word2offset:
            line = linecache.getline(os.path.join(
                glove_dir,'glove.840B.300d.txt'), word2offset[word]+1)
            assert(word == line[:line.find(' ')].strip())
            word_vectors.append(np.fromstring(
                line[line.find(' '):].strip(), sep=' ', dtype=np.float32))
        elif word == '<pad>':
            word_vectors.append(np.zeros(dimension_size, dtype=np.float32))
        else:
            word_vectors.append(
                np.random.uniform(-uniform_scale, uniform_scale, dimension_size))
    return word_vectors

# 创新点：能否利用其它三种数据形式（现有的数据集中每一条数据都只有一个aspect）
def get_rolled_and_unrolled_data(input_data, args):
    '''
    In input_data, each sentence could have multiple aspects with different sentiments.
    Our method treats each sentence with one aspect at a time, so even for
    multi-aspect-multi-sentiment sentences, we will unroll them to single aspect sentence.

    Perform reshape_dependency_tree to each sentence with aspect

    return:
        all_rolled:
                a list of dict
                    {sentence, tokens, pos_tags, pos_class, aspects(list of aspects), sentiments(list of sentiments)
                    froms, tos, dep_tags, dep_index, dependencies}
        all_unrolled:
                unrolled, with aspect(single), sentiment(single) and so on...
        mixed_rolled:
                Multiple aspects and multiple sentiments, ROLLED.
        mixed_unrolled:
                Multiple aspects and multiple sentiments, UNROLLED.
    '''
    # A hand-picked set of part of speech tags that we see contributes to ABSA.
    # opinionated_tags = ['JJ', 'JJR', 'JJS', 'RB', 'RBR',
    #                     'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

    all_rolled = []
    all_unrolled = []
    mixed_rolled = []
    mixed_unrolled = []

    # Make sure the tree is successfully built.
    zero_dep_counter = 0

    # Sentiment counters
    total_counter = defaultdict(int)
    mixed_counter = defaultdict(int)
    sentiments_lookup = {'negative': 0, 'positive': 1, 'neutral': 2}

    logger.info('*** Start processing data(unrolling and reshaping) ***')
    
    # for seeking 'but' examples
    for e in input_data:
        e['tokens'] = [x.lower() for x in e['tokens']]
        aspects = []
        sentiments = []
        froms = []
        tos = []

        # Classify based on POS-tags

        pos_class = e['tags']
        # Iterate through aspects in a sentence
        for i in range(len(e['aspect_sentiment'])):
            aspect = e['aspect_sentiment'][i][0].lower()
            # We would tokenize the aspect while at it.
            aspect = word_tokenize(aspect)
            if not args.inInfer:
                sentiment = sentiments_lookup[e['aspect_sentiment'][i][1]]
            else:
                sentiment = -1
            frm = e['from_to'][i][0]
            to = e['from_to'][i][1]

            aspects.append(aspect)
            sentiments.append(sentiment)
            froms.append(frm)
            tos.append(to)

            total_counter[e['aspect_sentiment'][i][1]] += 1

            # Unrolling
            all_unrolled.append(
                {'sentence': e['tokens'], 'tags': e['tags'], 'pos_class': pos_class, 'aspect': aspect, 'sentiment': sentiment,
                    'predicted_dependencies': e['predicted_dependencies'], 'predicted_heads': e['predicted_heads'],
                 'from': frm, 'to': to, 'dependencies': e['dependencies']})


        # All sentences with multiple aspects and sentiments rolled.
        all_rolled.append(
            {'sentence': e['tokens'], 'tags': e['tags'], 'pos_class': pos_class, 'aspects': aspects, 'sentiments': sentiments,
             'from': froms, 'to': tos, 'dependencies': e['dependencies']})

        # Ignore sentences with single aspect or no aspect
        if len(e['aspect_sentiment']) and len(set(map(lambda x: x[1], e['aspect_sentiment']))) > 1:
            mixed_rolled.append(
                {'sentence': e['tokens'], 'tags': e['tags'], 'pos_class': pos_class, 'aspects': aspects, 'sentiments': sentiments,
                 'from': froms, 'to': tos, 'dependencies': e['dependencies']})

            # Unrolling
            for i, as_sent in enumerate(e['aspect_sentiment']):
                mixed_counter[as_sent[1]] += 1
                mixed_unrolled.append(
                    {'sentence': e['tokens'], 'tags': e['tags'], 'pos_class': pos_class, 'aspect': aspects[i], 'sentiment': sentiments[i],
                     'from': froms[i], 'to': tos[i], 'dependencies': e['dependencies']})


    logger.info('Total sentiment counter: %s', total_counter)
    logger.info('Multi-Aspect-Multi-Sentiment counter: %s', mixed_counter)

    return all_rolled, all_unrolled, mixed_rolled, mixed_unrolled

class ALSC_Dataset(Dataset):
    def __init__(self, data, args, word_vocab, dep_tag_vocab, pos_tag_vocab):
        self.data = data
        self.args = args
        self.word_vocab = word_vocab
        self.dep_tag_vocab = dep_tag_vocab
        self.pos_tag_vocab = pos_tag_vocab

        self.convert_features()
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        e = self.data[idx]
        items = e['dep_tag_ids'], \
            e['pos_class'], e['text_len'], e['aspect_len'], e['sentiment'],\
            e['predicted_heads'], e['aspect_position']
        if self.args.embedding_type == 'glove':
            non_bert_items = e['sentence_ids'], e['aspect_ids']
            items_tensor = non_bert_items + items
            items_tensor = tuple(torch.tensor(t) for t in items_tensor)
        else:  # bert
            if self.args.pure_bert:
                bert_items = e['input_cat_ids'], e['segment_ids']
                items_tensor = tuple(torch.tensor(t) for t in bert_items)
                items_tensor += tuple(torch.tensor(t) for t in items)
            else:
                bert_items = e['input_ids'], e['word_indexer'], e['input_aspect_ids'], e['aspect_indexer'], e['input_cat_ids'], e['segment_ids']
                # segment_id
                items_tensor = tuple(torch.tensor(t) for t in bert_items)
                items_tensor += tuple(torch.tensor(t) for t in items)
        return items_tensor
    
    def convert_features_bert(self, i):
        """
        BERT features.
        convert sentence to feature. 
        """
        cls_token = "[CLS]"
        sep_token = "[SEP]"
        pad_token = 0
        # tokenizer = self.args.tokenizer

        tokens = []
        word_indexer = []
        aspect_tokens = []
        aspect_indexer = []

        for word in self.data[i]['sentence']:
            word_tokens = self.args.tokenizer.tokenize(word)
            token_idx = len(tokens)
            tokens.extend(word_tokens)
            # word_indexer is for indexing after bert, feature back to the length of original length.
            word_indexer.append(token_idx)

        # aspect
        for word in self.data[i]['aspect']:
            word_aspect_tokens = self.args.tokenizer.tokenize(word)
            token_idx = len(aspect_tokens)
            aspect_tokens.extend(word_aspect_tokens)
            aspect_indexer.append(token_idx)

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0

        tokens = [cls_token] + tokens + [sep_token]
        aspect_tokens = [cls_token] + aspect_tokens + [sep_token]
        word_indexer = [i+1 for i in word_indexer]
        aspect_indexer = [i+1 for i in aspect_indexer]

        input_ids = self.args.tokenizer.convert_tokens_to_ids(tokens)
        input_aspect_ids = self.args.tokenizer.convert_tokens_to_ids(
            aspect_tokens)

        # check len of word_indexer equals to len of sentence.
        assert len(word_indexer) == len(self.data[i]['sentence'])
        assert len(aspect_indexer) == len(self.data[i]['aspect'])

        # THE STEP:Zero-pad up to the sequence length, save to collate_fn.

        if self.args.pure_bert:
            input_cat_ids = input_ids + input_aspect_ids[1:]
            segment_ids = [0] * len(input_ids) + [1] * len(input_aspect_ids[1:])

            self.data[i]['input_cat_ids'] = input_cat_ids
            self.data[i]['segment_ids'] = segment_ids
        else:
            input_cat_ids = input_ids + input_aspect_ids[1:]
            segment_ids = [0] * len(input_ids) + [1] * len(input_aspect_ids[1:])

            self.data[i]['input_cat_ids'] = input_cat_ids
            self.data[i]['segment_ids'] = segment_ids
            self.data[i]['input_ids'] = input_ids
            self.data[i]['word_indexer'] = word_indexer
            self.data[i]['input_aspect_ids'] = input_aspect_ids
            self.data[i]['aspect_indexer'] = aspect_indexer

    def convert_features(self):
        '''
        Convert sentence, aspects, pos_tags, dependency_tags to ids.
        '''
        for i in range(len(self.data)):
            if self.args.embedding_type == 'glove':
                self.data[i]['sentence_ids'] = [self.word_vocab['stoi'][w]
                                                for w in self.data[i]['sentence']]
                self.data[i]['aspect_ids'] = [self.word_vocab['stoi'][w]
                                              for w in self.data[i]['aspect']]
            else:  # self.args.embedding_type == 'bert'
                self.convert_features_bert(i)

            self.data[i]['text_len'] = len(self.data[i]['sentence'])
            self.data[i]['aspect_position'] = [0] * self.data[i]['text_len']
            try:  # find the index of aspect in sentence
                for j in range(self.data[i]['from'], self.data[i]['to']):
                    self.data[i]['aspect_position'][j] = 1
            except:
                for term in self.data[i]['aspect']:
                    self.data[i]['aspect_position'][self.data[i]
                                                    ['sentence'].index(term)] = 1
            # changed
            self.data[i]['dep_tag_ids'] = [self.dep_tag_vocab['stoi'][w]
                                           for w in self.data[i]['predicted_dependencies']]
            self.data[i]['pos_class'] = [self.pos_tag_vocab['stoi'][w]
                                             for w in self.data[i]['tags']]
            self.data[i]['aspect_len'] = len(self.data[i]['aspect'])


def my_collate(batch):
    '''
    Pad sentence and aspect in a batch.
    Sort the sentences based on length.
    Turn all into tensors.
    '''
    sentence_ids, aspect_ids, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_heads, aspect_positions = zip(
        *batch)  # from Dataset.__getitem__()
    text_len = torch.tensor(text_len)
    aspect_len = torch.tensor(aspect_len)
    sentiment = torch.tensor(sentiment)

    # Pad sequences.
    
    sentence_ids = pad_sequence(
        sentence_ids, batch_first=True, padding_value=0)
    aspect_ids = pad_sequence(aspect_ids, batch_first=True, padding_value=0)
    aspect_positions = pad_sequence(
        aspect_positions, batch_first=True, padding_value=0)

    dep_tag_ids = pad_sequence(dep_tag_ids, batch_first=True, padding_value=0)
    pos_class = pad_sequence(pos_class, batch_first=True, padding_value=0)

    dep_heads = pad_sequence(dep_heads, batch_first=True, padding_value=0)

    # Sort all tensors based on text len.
    _, sorted_idx = text_len.sort(descending=True)
    sentence_ids = sentence_ids[sorted_idx]
    aspect_ids = aspect_ids[sorted_idx]
    aspect_positions = aspect_positions[sorted_idx]
    dep_tag_ids = dep_tag_ids[sorted_idx]
    pos_class = pos_class[sorted_idx]
    text_len = text_len[sorted_idx]
    aspect_len = aspect_len[sorted_idx]
    sentiment = sentiment[sorted_idx]
    dep_heads = dep_heads[sorted_idx]

    return sentence_ids, aspect_ids, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_heads, aspect_positions


def my_collate_pure_bert(batch):
    '''
    Pad sentence and aspect in a batch.
    Sort the sentences based on length.
    Turn all into tensors.

    Process bert feature
    Pure Bert: cat text and aspect, cls to predict.
    Test indexing while at it?
    '''
    # sentence_ids, aspect_ids
    input_cat_ids, segment_ids, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_heads, aspect_positions = zip(
        *batch)  # from Dataset.__getitem__()

    text_len = torch.tensor(text_len)
    aspect_len = torch.tensor(aspect_len)
    sentiment = torch.tensor(sentiment)

    # Pad sequences.
    input_cat_ids = pad_sequence(
        input_cat_ids, batch_first=True, padding_value=0)
    segment_ids = pad_sequence(segment_ids, batch_first=True, padding_value=0)

    aspect_positions = pad_sequence(
        aspect_positions, batch_first=True, padding_value=0)

    dep_tag_ids = pad_sequence(dep_tag_ids, batch_first=True, padding_value=0)
    pos_class = pad_sequence(pos_class, batch_first=True, padding_value=0)

    dep_heads = pad_sequence(dep_heads, batch_first=True, padding_value=0)

    # Sort all tensors based on text len.
    _, sorted_idx = text_len.sort(descending=True)
    input_cat_ids = input_cat_ids[sorted_idx]
    segment_ids = segment_ids[sorted_idx]
    aspect_positions = aspect_positions[sorted_idx]
    dep_tag_ids = dep_tag_ids[sorted_idx]

    pos_class = pos_class[sorted_idx]
    text_len = text_len[sorted_idx]
    aspect_len = aspect_len[sorted_idx]
    sentiment = sentiment[sorted_idx]
    dep_heads = dep_heads[sorted_idx]

    return input_cat_ids, segment_ids, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_heads, aspect_positions


def my_collate_bert(batch):
    '''
    Pad sentence and aspect in a batch.
    Sort the sentences based on length.
    Turn all into tensors.

    Process bert feature
    '''
    input_ids, word_indexer, input_aspect_ids, aspect_indexer,input_cat_ids,segment_ids,  dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_heads, aspect_positions = zip(
        *batch)
    text_len = torch.tensor(text_len)
    aspect_len = torch.tensor(aspect_len)
    sentiment = torch.tensor(sentiment)

    # Pad sequences.
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    input_aspect_ids = pad_sequence(input_aspect_ids, batch_first=True, padding_value=0)
    input_cat_ids = pad_sequence(input_cat_ids, batch_first=True, padding_value=0)
    segment_ids = pad_sequence(segment_ids, batch_first=True, padding_value =0)
    # indexer are padded with 1, for ...
    word_indexer = pad_sequence(word_indexer, batch_first=True, padding_value=1)
    aspect_indexer = pad_sequence(aspect_indexer, batch_first=True, padding_value=1)

    aspect_positions = pad_sequence(
        aspect_positions, batch_first=True, padding_value=0)

    dep_tag_ids = pad_sequence(dep_tag_ids, batch_first=True, padding_value=0)
    pos_class = pad_sequence(pos_class, batch_first=True, padding_value=0)

    dep_heads = pad_sequence(dep_heads, batch_first=True, padding_value=0)

    # Sort all tensors based on text len.
    _, sorted_idx = text_len.sort(descending=True)
    input_ids = input_ids[sorted_idx]
    input_aspect_ids = input_aspect_ids[sorted_idx]
    word_indexer = word_indexer[sorted_idx]
    aspect_indexer = aspect_indexer[sorted_idx]
    input_cat_ids = input_cat_ids[sorted_idx]
    segment_ids = segment_ids[sorted_idx]
    aspect_positions = aspect_positions[sorted_idx]
    dep_tag_ids = dep_tag_ids[sorted_idx]
    pos_class = pos_class[sorted_idx]
    text_len = text_len[sorted_idx]
    aspect_len = aspect_len[sorted_idx]
    sentiment = sentiment[sorted_idx]
    dep_heads = dep_heads[sorted_idx]

    return input_ids, word_indexer, input_aspect_ids, aspect_indexer,input_cat_ids,segment_ids, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_heads, aspect_positions
