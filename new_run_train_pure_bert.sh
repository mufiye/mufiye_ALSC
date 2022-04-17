#!/bin/sh
# the shell is for chaosuan cloud and gat our model

module load anaconda/2021.05
source activate mufiye_NLP

python train.py --pure_bert --embedding_type bert --learning_rate 5e-5 --dataset_name twitter
python train.py --pure_bert --embedding_type bert --learning_rate 5e-5 --dataset_name rest
python train.py --pure_bert --embedding_type bert --learning_rate 5e-5 --dataset_name laptop