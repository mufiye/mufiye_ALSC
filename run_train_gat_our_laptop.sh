#!/bin/sh
# the shell is for chaosuan cloud and gat our model

module load anaconda/2021.05
source activate mufiye_NLP

# R-GAT in laptop
python train.py --gat_our --dataset_name laptop --highway --num_heads 9 --per_gpu_train_batch_size 32 --dropout 0.7 --num_layers 3 --hidden_size 400 --final_hidden_size 400
