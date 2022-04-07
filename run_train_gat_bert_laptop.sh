#!/bin/sh
# the shell is for chaosuan cloud and gat bert model

module load anaconda/2021.05
source activate mufiye_NLP

# GAT-bert model and Laptop dataset
python train.py --gat_bert --embedding_type bert --dataset_name laptop --dropout 0.3 --num_heads 7 --hidden_size 200 --learning_rate 5e-5