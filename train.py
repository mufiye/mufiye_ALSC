'''
# the file is for training
# the process:
1. 进行命令行参数的解析（argument parsing）
2. 进行model,optimizer,loss等的设置
3. 读取和加载数据集
4. 每个epoch将数据分batch放入特定的model中
  4.1 每个epoch返回训练过程的相关数据（比如loss,accuracy）
  4.2 特定的epoch还会进行eval以及将运行日志进行存储
5. 将训练得到的参数进行存储以及eval这个模型
'''
import argparse
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (BertConfig, BertForTokenClassification,
                                  BertTokenizer)
# from models.model import (Aspect_Text_GAT_ours,
#                     Pure_Bert, Aspect_Bert_GAT, Aspect_Text_GAT_only)
from trainer import train

# changed
from new_load_data import load_datasets_and_vocabs
from models.new_model import *


logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.if_cuda:
        torch.cuda.manual_seed_all(args.seed)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--inInfer', action='store_true',
                        help='inInfer but always not')
    '''
    some parameters about data and cuda
    '''
    parser.add_argument('--dataset_name', type=str, default='rest',
                        choices=['rest', 'laptop', 'twitter'],
                        help='Choose absa dataset.')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of classes of ABSA.')
    parser.add_argument('--output_dir', type=str, default='datasets',
                        help='Directory to store intermedia data.')
    parser.add_argument('--if_cuda', default=True, action='store_false',
                        help='about use cuda or not')
    parser.add_argument('--seed', type=int, default=2019,
                        help='random seed for initialization')
    
    
    
    '''
    Model parameters
    '''
    # 待修改：glove(glove已经处理完了) and bert model(question,还不知道怎么操作)
    
    # 感觉这部分不需要由命令行来决定
    # parser.add_argument('--model_path', type=str, default='/saved_models', 
    #                     help='the path to store the trained model parameters')

    # 需要修改的路径，\\与/, windows和linux的区别
    parser.add_argument('--glove_dir', type=str, default='./datasets/Glove',
                        help='Directory storing glove embeddings')
    parser.add_argument('--bert_model_dir', type=str, default='models/bert_base',
                        help='Path to pre-trained Bert model.')
    parser.add_argument('--pure_bert', action='store_true',
                        help='Cat text and aspect, [cls] to predict.')
    parser.add_argument('--gat_bert', action='store_true',
                        help='Cat text and aspect, [cls] to predict.')
    parser.add_argument('--highway', action='store_true',
                        help='Use highway embed.')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of layers of bilstm or highway or elmo.')
    
    # question：这三个是什么
    parser.add_argument('--add_non_connect',  type= bool, default=True,
                        help='Add a sepcial "non-connect" relation for aspect with no direct connection.')
    parser.add_argument('--multi_hop',  type= bool, default=True,
                        help='Multi hop non connection.')
    parser.add_argument('--max_hop', type = int, default=4,
                        help='max number of hops')
    
    parser.add_argument('--num_heads', type=int, default=6,
                        help='Number of heads for gat.')
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout rate for embedding.')
    parser.add_argument('--num_gcn_layers', type=int, default=2,
                        help='Number of GCN layers.')
    # the gcn_mem_dim is temporary unuseful
    parser.add_argument('--gcn_mem_dim', type=int, default=300,
                        help='Dimension of the W in GCN.')
    parser.add_argument('--gcn_dropout', type=float, default=0.2,
                        help='Dropout rate for GCN.')
    parser.add_argument('--embedding_type', type=str,default='glove', choices=['glove','bert'])
    parser.add_argument('--embedding_dim', type=int, default=300,
                        help='Dimension of glove embeddings')
    parser.add_argument('--dep_relation_embed_dim', type=int, default=300,
                        help='Dimension for dependency relation embeddings.')
    parser.add_argument('--hidden_size', type=int, default=100,
                        help='Hidden size of bilstm, in early stage.')
    parser.add_argument('--final_hidden_size', type=int, default=100,
                        help='final_hidden_size.')
    parser.add_argument('--num_mlps', type=int, default=2,
                        help='Number of mlps in the last of model.')

    # about GAT
    parser.add_argument('--gat', action='store_true',
                        help='GAT')
    parser.add_argument('--gat_our', action='store_true',
                        help='GAT_our')
    parser.add_argument('--gat_attention_type', type = str, choices=['linear','dotprod','gcn'], default='dotprod',
                        help='The attention used for gat')

    # new parameter about model
    parser.add_argument('--gat_noReshape_our', action='store_true',
                        help='gat_noReshape_our')
    parser.add_argument('--pure_bert', action='store_true',
                        help='pure_bert')
    parser.add_argument('--gat_noReshape_bert', action='store_true',
                        help='gat_noReshape_bert')

    '''
    training parameters
    '''
    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for Adam.")
    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    
    # question: max_grad_norm是什么？
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps(that update the weights) to perform. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")


    return parser.parse_args()

def check_args(args):
    logger.info(vars(args))


def main():
    # setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    # Parse args
    args = parse_args()
    check_args(args)

    # Setup CUDA, GPU training
    device = torch.device('cuda' if args.if_cuda and torch.cuda.is_available() else 'cpu')    
    args.device = device
    logger.info('Device is %s', args.device)

    # Set seed
    set_seed(args)
    
    # Bert, load pretrained model and tokenizer: about BERT module(third part package)
    if args.embedding_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir)
        args.tokenizer = tokenizer
    
    # Load dataset: about dataset load python file
    train_dataset, test_dataset, word_vocab, dep_tag_vocab, pos_tag_vocab = load_datasets_and_vocabs(args)
    # Build Model: about model package
    # if args.pure_bert:
    #     model = Pure_Bert(args)
    # elif args.gat_bert:
    #     model = Aspect_Bert_GAT(args, dep_tag_vocab['len'], pos_tag_vocab['len'])  # R-GAT + Bert
    # elif args.gat_our:
    #     model = Aspect_Text_GAT_ours(args, dep_tag_vocab['len'], pos_tag_vocab['len']) # R-GAT with reshaped tree
    # else:
    #     model = Aspect_Text_GAT_only(args, dep_tag_vocab['len'], pos_tag_vocab['len'])  # original GAT with reshaped tree
    if args.gat_noReshape_our:
        model = No_Reshaped_GAT_ours(args, dep_tag_vocab['len'])
    elif args.pure_bert:
        model = Pure_Bert(args)
    model.to(args.device)

    # Train: about trainer python file
    _, _,  all_eval_results, best_model,model_path = train(args, train_dataset, model, test_dataset)

    # saving best model
    logger.info("save the best model to %s", model_path)
    if not os.path.exists('./saved_models/state_dict'):
        os.mkdir('./saved_models/state_dict')
    if not os.path.exists('./saved_models/state_dict/best_model'):
        os.mkdir('./saved_models/state_dict/best_model')
    torch.save(best_model.state_dict(), model_path)

    if len(all_eval_results):
        best_eval_result = max(all_eval_results, key=lambda x: x['acc']) 
        for key in sorted(best_eval_result.keys()):
            logger.info("  %s = %s", key, str(best_eval_result[key]))


if __name__ == '__main__':
    main()
