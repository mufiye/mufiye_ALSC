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
from models.model import (Aspect_Text_GAT_ours,
                    Pure_Bert, Aspect_Bert_GAT, Aspect_Text_GAT_only)

logger = logging.getLogger(__name__)

class Inferer:
    def __init__(self, opt):
        self.opt = opt
        # 根据model class以及opt参数设置model
        self.model = 

        # switch model to evaluation mode
        self.model.eval()
        torch.autograd.set_grad_enabled(False)
    
    def evaluate(self, text, aspect):
        # 1. 语法解析
        # 2. 将数据变为train过程中输入数据的样式
        data = {

        }
        return
    
'''
refer from https://github.com/songyouwei/ABSA-PyTorch/blob/master/infer_example.py
'''

def main():
    # setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    # 待补全: model_classes, 指模型名称及其对应的class
    model_classes = {
        'a' : A,
    }

    # 待补全: Option对象, 待补全
    class Option(object): pass
    opt=Option()
    opt.model_name='gat_our'
    opt.dataset = 'rest'
    opt.state_dict_path = 'state_dict/gat_our_rest'
    opt.num_classes=3
    opt.if_cuda=False
    opt.seed=2019
    opt.num_layers=2
    opt.add_non_connect=True
    opt.multi_hop=True
    opt.max_hop=4
    opt.num_heads=6
    opt.num_gcn_layers=1


if __name__ == '__main__':
    main()

