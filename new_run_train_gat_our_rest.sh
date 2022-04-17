#!/bin/sh
# the shell is for chaosuan cloud and gat our model

module load anaconda/2021.05
source activate mufiye_NLP

# GAT-our model and Restaurant dataset
python train.py --num_train_epochs --gat_noReshape_our --dataset_name rest --highway --dropout 0.8 --per_gpu_train_batch_size 16 --num_layers 2 --num_mlps 2 --hidden_size 300 --final_hidden_size 300