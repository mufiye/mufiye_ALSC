#!/bin/sh
# the shell is for chaosuan cloud and gat our model

module load anaconda/2021.05
source activate mufiye_NLP

# rest
python train.py --num_train_epochs 100 --gat_noReshape_our --dataset_name rest --highway \
                --seed 2019 \
                --per_gpu_train_batch_size 32 \
                --hidden_size 100 --final_hidden_size 100 \
                --num_mlps 2 \
                --dropout 0.8 \
                --gcn_dropout 0.2