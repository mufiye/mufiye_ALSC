# mufiye_ALSC
graduation project in NLP domain

## 依赖包
### algorithm
1. 用于解析xml文件：lxml包，命令为conda install -c anaconda lxml
2. 用于解析和生成json文件：json包，命令为conda install -c jmcmurray json
3. 用于解析依赖关系：Spacy，命令为pip install -U pip setuptools wheel; pip install spacy
4. 用于f1_score function: sklearn包，命令为pip install -U scikit-learn
5. 用于分词：nltk，conda install nltk
6. 用于提供BERT预训练模型：transformers，pip install transformers
7. 用于训练数据可视化：tensorboard（pytorch1.2之后就自带了），pip install tb-nightly 
### showing system
1. Django: pip install Django==3.2.12
## dataset
计划使用别人已经语法解析好的dataset json file
### json文件构成
### dataset(pytorch对象)
collate_fn函数决定batch构成
#### 非BERT
* sentence：句子词向量
* aspect：方面词向量
* dep_tags：reshape的dependency关系的属性
* pos_class：每个词的语法信息
* text_len：句子长度
* aspect_len：方面词长度
* dep_rels：reshape前的dependency关系的属性
* dep_heads：predicted_heads（也就是reshape前的树）
* aspect_position：aspect的位置掩码向量
* dep_dirs：reshape的dependency关系的方向
#### pure BERT
见load_data.py的convert_features_bert函数
* input_ids：[CLS]+tokens+[SEP]+aspect+[SEP]
* token_type_ids：掩码用来区分tokens和aspect
#### BERT
前六项见load_data.py的convert_features_bert函数
* input_ids
* input_aspect_ids
* word_indexer：因为bert要做比较特别的分词，indexer记录下标是为了之后的还原
* input_cat_ids
* segment_ids
* dep_tags
* pos_class
* text_len
* aspect_len
* dep_rels
* dep_heads
* aspect_position
* dep_dirs
## idea
1. R-GAT（考虑语法解析结果中的relationship的graph attention network）
2. 对于一个剪裁的entire tree

## plan
### 2022.3.6
1. 准备好数据集
2. 生成对应的word vector
3. 搭建好项目算法部分的大致框架

### 2022.3.7
1. 继续搭建项目算法部分的大致框架
2. 彻底了解GAT, R-GAT, R-GAT with BERT这些模型

### 2022.3.10
1. read and write the code about loading dataset

### 2022.3.11
1. 完善整体框架（load_data.py, trainer.py, train.py）
2. function: get_rolled_and_unrolled_data
3. the package and function of BERT
4. load_data.py的my_collate函数
### 2022.3.13(原本，已废弃，待重新制定计划)
1. 完善整体框架1（model package 3个python file）
2. 仔细阅读与加深对代码的理解
3. 完善整体框架2（infer.py）（前提是充分完成上面的2——仔细阅读与加深对代码的理解）
4. 关于tensorBoardX，这涉及train过程中的logging和checkpoint的存储，也就是要补齐trainer.py中train函数的logging和存储checkpoint（前提条件是充分理解train函数这部分代码）
5. 存储dependency tree reshape后的结果
6. 尝试自己的重构dependency tree的方式（前提条件是充分理解model和dataset构成）
7. 对数据集json文件以及dataset对象进行重构（前提条件是充分理解数据集json文件和dataset对象的构成）
8. 尝试可视化树形结构（以及思考如何在网页中可视化树形结构）（关于echart）

### 2022.3.27
1. 详细解读R-GAT的模型代码(完成)

### 2022.3.28
1. 解读trainer.py文件中的代码
2. 详细解读其余部分代码（数据集的部分）

### 2022.4.4
1. add saving model parameters parts

### 2022.4.5
1. 尝试让train.py运行起来
### 2022.4.6
1. 开始写infer.py
### 2022.4.7
1. 尝试训练BERT模型(要找到bert pretrained model并下载)
2. 为每个模型的训练写一个shell脚本
3. 尝试使用tensorboard（命令为tensorboard --logdir=runs\Apr07_21-02-09_g0016）
4. 规范化改写代码、远端模型训练流程
### 2022.4.8
1. 写infer.py
### 其它待办
1. 完善整体框架1（model package 3个python file）
2. 存储dependency tree reshape后的结果
3. 完善整体框架2（infer.py）（前提是仔细阅读与加深对代码的理解）
4. 尝试自己的重构dependency tree的方式（前提条件是充分理解model和dataset构成）
5. 尝试可视化树形结构（以及思考如何在网页中可视化树形结构）（关于echart）
6. 构建Django网站后端
7. 构建网页前端（使用bootstrap）

## 遇到的问题以及解决方案
### NLTK(2022.4.5)
#### Question
Resource punkt not found. Please use the NLTK Downloader to obtain the resource:
>>> import nltk
>>> nltk.download('punkt')‘
#### Solution

### 关于编写代码和训练规范(2022.4.6)
#### Question
关于编写代码和训练规范
#### Solution
在本地端更改代码(注意不要把大的文件commit)，在训练端git clone，要写好每个脚本，尽可能下载state_dict

### 关于编写infer.py函数(2022.4.10)
#### Question1
数据集的处理流程（在load_data.py和train.py文件中）
#### Answer1
1. train_dataset, test_dataset, word_vocab, dep_tag_vocab, pos_tag_vocab = load_datasets_and_vocabs(args)
2. get_dataset()函数获取原始的json格式的数据集
3. get_rolled_and_unrolled_data()
4. load_and_cache_vocabs_vectors()
5. ALSC_Dataset()创建了数据集类

#### Question2
将text和aspect处理为json格式的数据
#### Answer2
已解决
#### Question3
使model正常运行
#### Answer3
保证各项参数都定义了

#### Question4
GPU训练的模型如何用cpu跑起来
#### Solution4
类似这样torch.load(os.path.join(self.ckptdir,'best_ckpt.pt'),map_location=torch.device(device))，加一个map_location参数

#### Question5
模型需要的是三维的input，但是我不进行填充只是二维的数据，如何将数据变为三维的？
#### Solution5