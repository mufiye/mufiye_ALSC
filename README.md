# mufiye_ALSC
## 依赖包
Python 3.7 

### algorithm
1. Pytorch 1.10.1, Cuda 11.3, Cudann 8.2.0
2. 用于解析xml文件：lxml包，命令为conda install -c anaconda lxml
3. 用于解析和生成json文件：json包，命令为conda install -c jmcmurray json
4. 用于解析依赖关系：Spacy，命令为pip install -U pip setuptools wheel; pip install spacy
5. 用于f1_score function: sklearn包，命令为pip install -U scikit-learn
6. 用于分词：nltk，conda install nltk
7. 用于提供BERT预训练模型：transformers，pip install transformers
8. 用于训练数据可视化：tensorboard（pytorch1.2之后就自带了），pip install tb-nightly 
### showing system
1. Django: pip install Django==3.2.12
## 数据集

## 相关运行命令
### 启动训练脚本
可以从根目录下的shell文件中寻找相应命令
```python
python train.py --gat_bert --embedding_type bert --dataset_name laptop --dropout 0.3 --num_heads 7 --hidden_size 200 --learning_rate 5e-5
```
### 启动Demo System
```
python manage.py runserver
```
## Reference Project:
1. [ABSA-pytorch](https://github.com/songyouwei/ABSA-PyTorch)
2. [ASGCN](https://github.com/GeneZC/ASGCN)
3. [Covolution_over_Dependency_Tree](https://github.com/BDBC-KG-NLP/Covolution_over_Dependency_Tree_EMNLP2019)
4. [DualGCN-ABSA](https://github.com/CCChenhao997/DualGCN-ABSA)
5. [InterGCN-ABSA](https://github.com/BinLiang-NLP/InterGCN-ABSA)
6. [RGAT-ABSA](https://github.com/shenwzh3/RGAT-ABSA)
