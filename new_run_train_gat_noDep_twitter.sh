#!/bin/sh
# the shell is for chaosuan cloud and gat our model

module load anaconda/2021.05
source activate mufiye_NLP

# twitter
python train.py --num_train_epochs 100 --gat_noDep_our --dataset_name twitter --highway --per_gpu_train_batch_size 32 --dropout 0.6 --num_mlps 2 --final_hidden_size 100