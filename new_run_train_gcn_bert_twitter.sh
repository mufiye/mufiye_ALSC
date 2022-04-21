#!/bin/sh
# the shell is for chaosuan cloud and gat bert model

module load anaconda/2021.05
source activate mufiye_NLP

#R-GAT+BERT in twitter
python train.py --gat_bert --embedding_type bert --dataset_name twitter \
                --num_train_epochs 50 \
                --learning_rate 5e-5 \
                --per_gpu_train_batch_size 16 \
                --num_gcn_layers 2 \
                --num_layers 1 \
                --final_hidden_size 512 \
                --num_mlps 3 \
                --dropout 0.2 \
                --gcn_dropout 0.0 