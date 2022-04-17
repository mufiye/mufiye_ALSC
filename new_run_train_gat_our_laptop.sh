#!/bin/sh
# the shell is for chaosuan cloud and gat our model

module load anaconda/2021.05
source activate mufiye_NLP

# R-GAT in laptop
python train.py --num_train_epochs 100 --gat_noReshape_our --dataset_name laptop --highway --per_gpu_train_batch_size 32 --dropout 0.7 --hidden_size 100 --final_hidden_size 100