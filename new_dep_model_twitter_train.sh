#!/bin/sh
# the shell is for chaosuan cloud and gat our model

module load anaconda/2021.05
source activate mufiye_NLP

# twitter train
python train.py --num_train_epochs 100 --gat_noReshape_our --dataset_name twitter --highway \
                --per_gpu_train_batch_size 8 \
                --hidden_size 100 --final_hidden_size 100 \
                --num_mlps 2 \
                --num_heads 9 \
                --dropout 0.5 \
                --gcn_dropout 0.0