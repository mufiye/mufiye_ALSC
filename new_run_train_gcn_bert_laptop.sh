#!/bin/sh
# the shell is for chaosuan cloud and gat bert model

module load anaconda/2021.05
source activate mufiye_NLP

# GAT-bert model and Laptop dataset
python train.py --gat_bert --embedding_type bert --dataset_name laptop \
                --num_train_epochs 50 \
                --learning_rate 5e-5 \
                --per_gpu_train_batch_size 32 \
                --num_gcn_layers 2 \
                --num_layers 2 \
                --final_hidden_size 256 \
                --num_mlps 3 \
                --gcn_dropout 0.2 \
                --dropout 0.3 \
                --seed 0

# python train.py --gat_bert --embedding_type bert --dataset_name laptop \
#                 --num_train_epochs 50 \
#                 --learning_rate 5e-5 \
#                 --per_gpu_train_batch_size 32 \
#                 --num_gcn_layers 2 \
#                 --num_layers 2 \
#                 --final_hidden_size 256 \
#                 --num_mlps 3 \
#                 --gcn_dropout 0.8 \
#                 --dropout 0.1

# python train.py --gat_bert --embedding_type bert --dataset_name laptop \
#                 --num_train_epochs 50 \
#                 --learning_rate 5e-5 \
#                 --per_gpu_train_batch_size 32 \
#                 --num_gcn_layers 2 \
#                 --num_layers 2 \
#                 --final_hidden_size 256 \
#                 --num_mlps 3 \
#                 --gcn_dropout 0.8 \
#                 --dropout 0.2

# python train.py --gat_bert --embedding_type bert --dataset_name laptop \
#                 --num_train_epochs 50 \
#                 --learning_rate 5e-5 \
#                 --per_gpu_train_batch_size 32 \
#                 --num_gcn_layers 2 \
#                 --num_layers 2 \
#                 --final_hidden_size 256 \
#                 --num_mlps 3 \
#                 --gcn_dropout 0.8 \
#                 --dropout 0.3

# python train.py --gat_bert --embedding_type bert --dataset_name laptop \
#                 --num_train_epochs 50 \
#                 --learning_rate 5e-5 \
#                 --per_gpu_train_batch_size 32 \
#                 --num_gcn_layers 2 \
#                 --num_layers 2 \
#                 --final_hidden_size 256 \
#                 --num_mlps 3 \
#                 --gcn_dropout 0.8 \
#                 --dropout 0.4

# python train.py --gat_bert --embedding_type bert --dataset_name laptop \
#                 --num_train_epochs 50 \
#                 --learning_rate 5e-5 \
#                 --per_gpu_train_batch_size 32 \
#                 --num_gcn_layers 2 \
#                 --num_layers 2 \
#                 --final_hidden_size 256 \
#                 --num_mlps 3 \
#                 --gcn_dropout 0.8 \
#                 --dropout 0.5

# python train.py --gat_bert --embedding_type bert --dataset_name laptop \
#                 --num_train_epochs 50 \
#                 --learning_rate 5e-5 \
#                 --per_gpu_train_batch_size 32 \
#                 --num_gcn_layers 2 \
#                 --num_layers 2 \
#                 --final_hidden_size 256 \
#                 --num_mlps 3 \
#                 --gcn_dropout 0.8 \
#                 --dropout 0.6

# python train.py --gat_bert --embedding_type bert --dataset_name laptop \
#                 --num_train_epochs 50 \
#                 --learning_rate 5e-5 \
#                 --per_gpu_train_batch_size 32 \
#                 --num_gcn_layers 2 \
#                 --num_layers 2 \
#                 --final_hidden_size 256 \
#                 --num_mlps 3 \
#                 --gcn_dropout 0.8 \
#                 --dropout 0.7

# python train.py --gat_bert --embedding_type bert --dataset_name laptop \
#                 --num_train_epochs 50 \
#                 --learning_rate 5e-5 \
#                 --per_gpu_train_batch_size 32 \
#                 --num_gcn_layers 2 \
#                 --num_layers 2 \
#                 --final_hidden_size 256 \
#                 --num_mlps 3 \
#                 --gcn_dropout 0.8 \
#                 --dropout 0.8

# python train.py --gat_bert --embedding_type bert --dataset_name laptop \
#                 --num_train_epochs 50 \
#                 --learning_rate 5e-5 \
#                 --per_gpu_train_batch_size 32 \
#                 --num_gcn_layers 2 \
#                 --num_layers 2 \
#                 --final_hidden_size 256 \
#                 --num_mlps 3 \
#                 --gcn_dropout 0.8 \
#                 --dropout 0.9