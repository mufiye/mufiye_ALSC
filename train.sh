#!/bin/sh
# the shell script in linux operating system for training

# GAT-our model and Restaurant dataset
python train.py --gat_our --dataset_name rest --highway --num_heads 7 --dropout 0.8 --per_gpu_train_batch_size 16 --num_layers 2 --num_mlps 2 --hidden_size 300 --final_hidden_size 300 

# some reference
# python train.py --gat_bert --embedding_type bert --output_dir data/output-gcn --dropout 0.3 --hidden_size 200 --learning_rate 5e-5 #R-GAT+BERT in restaurant
# python train.py --gat_bert --embedding_type bert --dataset_name laptop --output_dir data/output-gcn-laptop --dropout 0.3 --num_heads 7 --hidden_size 200 --learning_rate 5e-5 #R-GAT+BERT in laptop
# python train.py --gat_bert --embedding_type bert --dataset_name twitter --output_dir data/output-gcn-twitter --dropout 0.2  --hidden_size 200 --learning_rate 5e-5 #R-GAT+BERT in twitter
# python train.py --gat_our --highway --num_heads 7 --dropout 0.8 # R-GAT in restaurant
# python train.py --gat_our --dataset_name laptop --output_dir data/output-gcn-laptop --highway --num_heads 9 --per_gpu_train_batch_size 32 --dropout 0.7 --num_layers 3 --hidden_size 400 --final_hidden_size 400 # R-GAT in laptop
# python train.py --gat_our --dataset_name twitter --output_dir data/output-gcn-twitter --highway --num_heads 9 --per_gpu_train_batch_size 8 --dropout 0.6 --num_mlps 1 --final_hidden_size 400 # R-GAT in twitter