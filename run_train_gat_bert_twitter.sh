#!/bin/sh
# the shell is for chaosuan cloud and gat bert model

module load anaconda/2021.05
source activate mufiye_NLP

#R-GAT+BERT in twitter
python train.py --gat_bert --embedding_type bert --dataset_name twitter --output_dir data/output-gcn-twitter --dropout 0.2  --hidden_size 200 --learning_rate 5e-5 