#!/usr/bin/env bash

python chinese_train.py --gcn_conv_type 0 --model_type 4 --dgl_type pool --weight_define 1 --dgl_layer 1 --seed_num 1 --window_size 3 --model_name dgl1_p4_pool_w1_size3
python chinese_train.py --gcn_conv_type 0 --model_type 4 --dgl_type pool --weight_define 1 --dgl_layer 1 --seed_num 100 --window_size 3 --model_name dgl1_p4_pool_w1_size3

python newtest.py --type_id 3 --model_name dgl1_p4_pool_w1_size3 --seed_start 0 --seed_end 2

python chinese_train.py --gcn_conv_type 0 --model_type 4 --dgl_type pool --weight_define 3 --dgl_layer 1 --seed_num 1 --window_size 3 --model_name dgl1_p4_pool_w3_size3
python chinese_train.py --gcn_conv_type 0 --model_type 4 --dgl_type pool --weight_define 3 --dgl_layer 1 --seed_num 100 --window_size 3 --model_name dgl1_p4_pool_w3_size3

python newtest.py --type_id 3 --model_name dgl1_p4_pool_w3_size3 --seed_start 0 --seed_end 2