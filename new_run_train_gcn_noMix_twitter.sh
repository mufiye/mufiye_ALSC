#!/bin/sh
# the shell is for chaosuan cloud and gat our model

module load anaconda/2021.05
source activate mufiye_NLP

# twitter
# --per_gpu_train_batch_size 8 f1会高些
python train.py --num_train_epochs 100 --gat_noMix_our --dataset_name twitter --highway \
                --per_gpu_train_batch_size 32 --dropout 0.6 \
                --gcn_dropout 0.0  \
                --hidden_size 200 --final_hidden_size 200 --num_mlps 3