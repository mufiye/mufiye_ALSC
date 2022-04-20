#!/bin/sh
# the shell is for chaosuan cloud and gat our model

module load anaconda/2021.05
source activate mufiye_NLP

# rest
python train.py --num_train_epochs 100 --gat_noTogether_our --dataset_name rest --highway --dropout 0.8 --per_gpu_train_batch_size 32 --num_mlps 2 --hidden_size 100 --final_hidden_size 100