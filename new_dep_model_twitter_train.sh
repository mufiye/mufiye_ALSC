#!/bin/sh
# the shell is for chaosuan cloud and gat our model

module load anaconda/2021.05
source activate mufiye_NLP

# twitter train
python train.py --num_train_epochs 100 --gat_noReshape_our --dataset_name twitter --highway \
                --per_gpu_train_batch_size 32 \
                --hidden_size 400 --final_hidden_size 400 \
                --gcn_mem_dim 200 \
                --num_mlps 2 \
                --dropout 0.3 \
                --gcn_dropout 0.2 \


# python train.py --num_train_epochs 100 --gat_noReshape_our --dataset_name twitter --highway --per_gpu_train_batch_size 32 --hidden_size 400 --final_hidden_size 400 --gcn_mem_dim 200 --num_mlps 2 --dropout 0.3 --gcn_dropout 0.2
                