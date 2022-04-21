'''
a class for model training
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os
import argparse
import random
import logging
import copy
from tqdm import tqdm, trange

from sklearn.metrics import f1_score

#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# changed
from new_load_data import my_collate, my_collate_pure_bert, my_collate_bert
from transformers import AdamW
from transformers import BertTokenizer

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


# 从batch中读取并且处理得到输入数据
# 待细读: 数据集的构成值得细读
def get_input_from_batch(args, batch):
    embedding_type = args.embedding_type
    if embedding_type == 'glove':
        inputs = {  'sentence': batch[0],
                    'aspect': batch[1], # aspect token
                    'dep_tags': batch[2], # no reshaped
                    'pos_class': batch[3],
                    'text_len': batch[4],
                    'aspect_len': batch[5],
                    'dep_heads': batch[7],
                    'aspect_position': batch[8]
                    }
        labels = batch[6]
    else: # bert
        if args.pure_bert:
            inputs = {  'input_ids': batch[0],
                        'token_type_ids': batch[1]}
            labels = batch[6]
        else:
            # input_ids, word_indexer, input_aspect_ids, aspect_indexer, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_rel_ids, dep_heads, aspect_positions
            inputs = {  'input_ids': batch[0],
                        'input_aspect_ids': batch[2],
                        'word_indexer': batch[1],
                        'aspect_indexer': batch[3],
                        'input_cat_ids': batch[4],
                        'segment_ids': batch[5],
                        'dep_tags': batch[6],
                        'pos_class': batch[7],
                        'text_len': batch[8],
                        'aspect_len': batch[9],
                        'dep_heads': batch[11],
                        'aspect_position': batch[12]
                        }
            labels = batch[10]
    return inputs, labels


def get_collate_fn(args):
    embedding_type = args.embedding_type
    if embedding_type == 'glove':
        return my_collate
    else:
        if args.pure_bert:
            return my_collate_pure_bert
        else:
            return my_collate_bert

# question: about bert optimizer
def get_bert_optimizer(args, model):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = WarmupLinearSchedule(
    #     optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    return optimizer


# 训练模型
def train(args, train_dataset, model, test_dataset):
    # the tensorboard writer
    if args.gat_noReshape_our:
        tb_writer_log_path = "gat_noReshape_our_{}_tensorboard_log".format(args.dataset_name)
    elif args.pure_bert:
        tb_writer_log_path = "pure_bert_{}_tensorboard_log".format(args.dataset_name)      
    elif args.gat_noMix_our:
        tb_writer_log_path = "gat_noMix_our_{}_tensorboard_log".format(args.dataset_name)      
    elif args.gat_noDep_our:
        tb_writer_log_path = "gat_noDep_our_{}_tensorboard_log".format(args.dataset_name)      
    else:
        tb_writer_log_path = "gat_noReshape_bert_{}_tensorboard_log".format(args.dataset_name)      
    tb_writer = SummaryWriter(tb_writer_log_path)

    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)

    # 加载数据
    collate_fn = get_collate_fn(args)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    
    # 关于优化器
    if args.embedding_type == 'bert':
        optimizer = get_bert_optimizer(args, model)
    else:
        parameters = filter(lambda param: param.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    
    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    all_eval_results = []

    # new added for saving best model
    max_test_acc = 0
    best_model = copy.deepcopy(model)
    model_path = ''

    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    for train_epoch in train_iterator:
        logger.info(" it is in train epoch %d", train_epoch+1)
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs, labels = get_input_from_batch(args, batch)
            logit = model(**inputs)
            loss = F.cross_entropy(logit, labels)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)
            
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1
                
                # 待修改：关于logging
                # 待修改：关于checkpoint，bestmodel记录文件的文件名（考虑时间和模型类型）
                # 已修改：checkpoint与best model
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    results, eval_loss = evaluate(args, test_dataset, model)
                    all_eval_results.append(results)
                    # record tensorboard data
                    for key, value in results.items():
                        tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    # tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('eval_loss', eval_loss, global_step)
                    tb_writer.add_scalar('train_loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss
                    # store the best model
                    if results['acc'] > max_test_acc:
                        max_test_acc = results['acc']
                        best_model = copy.deepcopy(model)
                        if args.gat_noReshape_our:
                            model_path = './saved_models/state_dict/best_model/gat_noReshape_our_{}_acc_{:.4f}_f1_{:.4f}' \
                                            .format(args.dataset_name, results['acc'], results['f1'])
                        elif args.pure_bert:
                            model_path = './saved_models/state_dict/best_model/pure_bert_{}_acc_{:.4f}_f1_{:.4f}' \
                                            .format(args.dataset_name, results['acc'], results['f1'])
                        elif args.gat_noMix_our:
                            model_path = './saved_models/state_dict/best_model/gat_noMix_our_{}_acc_{:.4f}_f1_{:.4f}' \
                                            .format(args.dataset_name, results['acc'], results['f1'])
                        elif args.gat_noDep_our:
                            model_path = './saved_models/state_dict/best_model/gat_noDep_our_{}_acc_{:.4f}_f1_{:.4f}' \
                                            .format(args.dataset_name, results['acc'], results['f1'])
                        else:
                            model_path = './saved_models/state_dict/best_model/gat_noReshape_bert_{}_acc_{:.4f}_f1_{:.4f}' \
                                            .format(args.dataset_name, results['acc'], results['f1'])
                        
        # check point
        if (train_epoch+1) % 5 == 0:
            if args.gat_noReshape_our:
                checkpoint_model_path = './saved_models/state_dict/checkPoint/gat_noReshape_our_{}_checkPoint_{}' \
                                .format(args.dataset_name, train_epoch+1)
            elif args.pure_bert:
                checkpoint_model_path = './saved_models/state_dict/checkPoint/pure_bert_{}_checkPoint_{}' \
                                .format(args.dataset_name, train_epoch+1)
            elif args.gat_noMix_our:
                checkpoint_model_path = './saved_models/state_dict/checkPoint/gat_noMix_our_{}_checkPoint_{}' \
                                .format(args.dataset_name, train_epoch+1)
            elif args.gat_noDep_our:
                checkpoint_model_path = './saved_models/state_dict/checkPoint/gat_noDep_our_{}_checkPoint_{}' \
                                .format(args.dataset_name, train_epoch+1)
            else:
                checkpoint_model_path = './saved_models/state_dict/checkPoint/gat_noReshape_bert_{}_checkPoint_{}' \
                                .format(args.dataset_name, train_epoch+1)
            if not os.path.exists('./saved_models/state_dict'):
                os.mkdir('./saved_models/state_dict')
            if not os.path.exists('./saved_models/state_dict/checkPoint'):
                os.mkdir('./saved_models/state_dict/checkPoint')
            torch.save(model, checkpoint_model_path)
    logger.info("evaluate train dataset")
    trainData_results, trainData_eval_loss = evaluate(args, train_dataset, best_model)
    tb_writer.close()
    return global_step, tr_loss/global_step, all_eval_results, best_model, model_path

       
# 评估模型的性能
def evaluate(args, eval_dataset, model):
    results = {}

    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(eval_dataset)
    collate_fn = get_collate_fn(args)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)
    
    # Eval
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs, labels = get_input_from_batch(args, batch)
            
            logits = model(**inputs)
            tmp_eval_loss = F.cross_entropy(logits, labels)

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, labels.detach().cpu().numpy(), axis=0)
    
    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)

    result = compute_metrics(preds, out_label_ids)
    results.update(result)

    if args.gat_noReshape_our:
        eval_results_fileName = 'eval_results_gat_noReshape_our_{}.txt'.format(args.dataset_name)
    elif args.pure_bert:
        eval_results_fileName = 'eval_results_pure_bert_{}.txt'.format(args.dataset_name)
    elif args.gat_noMix_our:
        eval_results_fileName = 'eval_results_gat_noMix_our_{}.txt'.format(args.dataset_name)
    elif args.gat_noDep_our:
        eval_results_fileName = 'eval_results_gat_noDep_our_{}.txt'.format(args.dataset_name)
    else:
        eval_results_fileName = 'eval_results_gat_noReshape_bert_{}.txt'.format(args.dataset_name)
    output_eval_file = os.path.join('./saved_models', eval_results_fileName)
    with open(output_eval_file, 'a+') as writer:
        logger.info('***** Eval results *****')
        logger.info("  eval loss: %s", str(eval_loss))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("  %s = %s\n" % (key, str(result[key])))
            writer.write('\n')
        writer.write('\n')
    
    return results, eval_loss


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "acc": acc,
        "f1": f1
    }


def compute_metrics(preds, labels):
    return acc_and_f1(preds, labels)