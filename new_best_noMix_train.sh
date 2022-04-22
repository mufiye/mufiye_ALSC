# 2022.4.22
# bert-base model

# rest
python train.py --num_train_epochs 100 --gat_noMix_our --dataset_name rest --highway \
                --seed 2019 \
                --per_gpu_train_batch_size 32 \
                --hidden_size 100 --final_hidden_size 100 \
                --num_mlps 2 \
                --dropout 0.8 \
                --gcn_dropout 0.2

# laptop, gcn_dropout为0.2也差不多
python train.py --num_train_epochs 100 --gat_noMix_our --dataset_name laptop --highway \
                --per_gpu_train_batch_size 32 \
                --hidden_size 100 --final_hidden_size 100 \
                --num_layers 2 --num_mlps 3 \
                --dropout 0.7 \
                --gcn_dropout 0.8

# twitter
python train.py --num_train_epochs 100 --gat_noMix_our --dataset_name twitter --highway \
                --per_gpu_train_batch_size 32 \
                --hidden_size 100 --final_hidden_size 100 \
                --num_mlps 2 \
                --dropout 0.5 \
                --gcn_dropout 0.0