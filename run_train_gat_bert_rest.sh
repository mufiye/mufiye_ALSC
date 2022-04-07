#!/bin/sh
# the shell is for chaosuan cloud and gat bert model

module load anaconda/2021.05
source activate mufiye_NLP
#R-GAT+BERT in restaurant
python train.py --gat_bert --embedding_type bert --dropout 0.3 --hidden_size 200 --learning_rate 5e-5 