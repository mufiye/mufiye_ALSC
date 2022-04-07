import argparse
import json
import os
import tqdm

from lxml import etree
import spacy
from spacy.tokens import Doc


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_path', type=str, default=model_path,
    #                     help='Path to biaffine dependency parser.')
    parser.add_argument('--data_path', type=str, default='../datasets/SemEval-2014',
                        help='Directory of where semeval14 or twiiter data held.')
    return parser.parse_args()


'''
from ASGCN project's dependency_tree.py
'''


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

'''
找出xml文件中每一项的句子
'''


def find_sentence(file_path):
    sen_list = []
    with open(file_path, 'rb') as f:
        raw = f.read()
        root = etree.fromstring(raw)
        for sentence in root:
            sent = sentence.find('text').text
            terms = sentence.find('aspectTerms')
            if terms is None:
                continue
            if terms:
                sen_list.append(sent)
    return sen_list


'''
对句子进行语法解析，返回语法解析的结果
！！！ 未完成 ！！！
'''
def get_dependencies(sen_list):
    print('parsing dependency information...')
    '''
    循环解析每一句话的依赖关系
    此处模仿ASGCN使用了spacy，后续可以考虑使用其它的parser
    '''
    sentences = []
    for i in tqdm(range(len(sen_list))):
        doc = nlp(sen_list[i])
        sentence_info = {}
        sentence_info['token'] = []
        sentence_info['pos'] = ['0'] * len(doc)
        sentence_info['head'] = [0] * len(doc)
        sentence_info['deprel'] = []
        for token in doc:
            sentence_info['token'].append(token.text)
            sentence_info['deprel'].append(token.dep_)
        for i,token in enumerate(doc):
            sentence_info['head'][i] = sentence_info['token'].index(token.head.text)+1
            sentence_info['pos'][sentence_info['token'].index(token.head.text)] = token.head.pos_
        sentences.append(sentence_info)

    return sentences


'''
将结果综合其它信息写入到json文件中
！！！ 未完成 ！！！
'''


def syntaxInfo2json(sentences, origin_file):
    json_data = []
    extended_filename = origin_file.replace('.xml', '.json')
    with open(extended_filename, 'w') as f:
        json.dump(json_data, f)
    print('done', len(json_data))


def main():
    args = parse_args()
    data = ['Restaurants_Train_v2.xml',
            'Restaurants_Test_Gold.xml',
            'Laptop_Train_v2.xml',
            'Laptops_Test_Gold.xml']
    for data_file in data:
        sen_list = find_sentence(data_file)
        sentences = get_dependencies(sen_list)
        syntaxInfo2json(sentences, os.path.join(args.data_path, data_file))
        print("already done the {}".format(data_file))


if __name__ == "__main__":
    main()
