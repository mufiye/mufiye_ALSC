#!/bin/sh
# the shell is for chaosuan cloud and gat our model

module load anaconda/2021.05
source activate mufiye_NLP

# R-GAT in twitter
python train.py --gat_our --dataset_name twitter --highway --num_heads 9 --per_gpu_train_batch_size 8 --dropout 0.6 --num_mlps 1 --final_hidden_size 400 