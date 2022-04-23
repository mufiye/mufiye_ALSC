#!/bin/sh
# the shell is for chaosuan cloud and gat our model

module load anaconda/2021.05
source activate mufiye_NLP

# laptop train
python train.py --num_train_epochs 100 --gat_noReshape_our --dataset_name laptop --highway \
                --per_gpu_train_batch_size 32 \
                --hidden_size 100 --final_hidden_size 100 \
                --num_layers 2 --num_mlps 3 \
                --num_heads 9 \
                --dropout 0.3 \
                --gcn_dropout 0.2

# python train.py --num_train_epochs 100 --gat_noReshape_our --dataset_name laptop --highway \
#                 --per_gpu_train_batch_size 32 \
#                 --hidden_size 100 --final_hidden_size 100 \
#                 --num_layers 2 --num_mlps 3 \
#                 --dropout 0.3 \
#                 --gcn_dropout 0.1

# python train.py --num_train_epochs 100 --gat_noReshape_our --dataset_name laptop --highway \
#                 --per_gpu_train_batch_size 32 \
#                 --hidden_size 100 --final_hidden_size 100 \
#                 --num_layers 2 --num_mlps 3 \
#                 --dropout 0.3 \
#                 --gcn_dropout 0.2

# python train.py --num_train_epochs 100 --gat_noReshape_our --dataset_name laptop --highway \
#                 --per_gpu_train_batch_size 32 \
#                 --hidden_size 100 --final_hidden_size 100 \
#                 --num_layers 2 --num_mlps 3 \
#                 --dropout 0.3 \
#                 --gcn_dropout 0.3

# python train.py --num_train_epochs 100 --gat_noReshape_our --dataset_name laptop --highway \
#                 --per_gpu_train_batch_size 32 \
#                 --hidden_size 100 --final_hidden_size 100 \
#                 --num_layers 2 --num_mlps 3 \
#                 --dropout 0.3 \
#                 --gcn_dropout 0.4

# python train.py --num_train_epochs 100 --gat_noReshape_our --dataset_name laptop --highway \
#                 --per_gpu_train_batch_size 32 \
#                 --hidden_size 100 --final_hidden_size 100 \
#                 --num_layers 2 --num_mlps 3 \
#                 --dropout 0.3 \
#                 --gcn_dropout 0.5

# python train.py --num_train_epochs 100 --gat_noReshape_our --dataset_name laptop --highway \
#                 --per_gpu_train_batch_size 32 \
#                 --hidden_size 100 --final_hidden_size 100 \
#                 --num_layers 2 --num_mlps 3 \
#                 --dropout 0.3 \
#                 --gcn_dropout 0.6

# python train.py --num_train_epochs 100 --gat_noReshape_our --dataset_name laptop --highway \
#                 --per_gpu_train_batch_size 32 \
#                 --hidden_size 100 --final_hidden_size 100 \
#                 --num_layers 2 --num_mlps 3 \
#                 --dropout 0.3 \
#                 --gcn_dropout 0.7

# python train.py --num_train_epochs 100 --gat_noReshape_our --dataset_name laptop --highway \
#                 --per_gpu_train_batch_size 32 \
#                 --hidden_size 100 --final_hidden_size 100 \
#                 --num_layers 2 --num_mlps 3 \
#                 --dropout 0.3 \
#                 --gcn_dropout 0.8

# python train.py --num_train_epochs 100 --gat_noReshape_our --dataset_name laptop --highway \
#                 --per_gpu_train_batch_size 32 \
#                 --hidden_size 100 --final_hidden_size 100 \
#                 --num_layers 2 --num_mlps 3 \
#                 --dropout 0.3 \
#                 --gcn_dropout 0.9
