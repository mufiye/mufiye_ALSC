# 2022.4.22
# bert-base model

# rest
python train.py --gat_bert --embedding_type bert \
                --seed 2019 \
                --num_train_epochs 50 \
                --learning_rate 5e-5 \
                --per_gpu_train_batch_size 32 \
                --num_gcn_layers 2 \
                --final_hidden_size 256 \
                --num_mlps 2 \
                --dropout 0.3 \
                --gcn_dropout 0.3

# laptop
python train.py --gat_bert --embedding_type bert --dataset_name laptop \
                --num_train_epochs 50 \
                --learning_rate 5e-5 \
                --per_gpu_train_batch_size 32 \
                --num_gcn_layers 2 \
                --final_hidden_size 256 \
                --num_mlps 3 \
                --dropout 0.0 \
                --gcn_dropout 0.5

# twitter
python train.py --gat_bert --embedding_type bert --dataset_name twitter \
                --num_train_epochs 50 \
                --learning_rate 5e-5 \
                --per_gpu_train_batch_size 16 \
                --num_gcn_layers 2 \
                --final_hidden_size 256 \
                --num_mlps 2 \
                --dropout 0.0 \
                --gcn_dropout 0.2