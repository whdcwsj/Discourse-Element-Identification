#!/usr/bin/env bash

python chinese_train.py --gcn_conv_type 0 --model_type 4 --dgl_type pool --weight_define 3 --dgl_layer 1 --seed_num 1 --window_size 1 --model_name dgl1_p4_pool_w3_size1
python chinese_train.py --gcn_conv_type 0 --model_type 4 --dgl_type pool --weight_define 3 --dgl_layer 1 --seed_num 100 --window_size 1 --model_name dgl1_p4_pool_w3_size1

python newtest.py --type_id 3 --model_name dgl1_p4_pool_w3_size1 --seed_start 0 --seed_end 2

python chinese_train.py --gcn_conv_type 0 --model_type 4 --dgl_type pool --weight_define 1 --dgl_layer 1 --seed_num 1 --window_size 1 --add_self_loop 1 --model_name dgl1_p4_pool_w1_size1_loop
python chinese_train.py --gcn_conv_type 0 --model_type 4 --dgl_type pool --weight_define 1 --dgl_layer 1 --seed_num 100 --window_size 1 --add_self_loop 1 --model_name dgl1_p4_pool_w1_size1_loop

python newtest.py --type_id 3 --model_name dgl1_p4_pool_w1_size1_loop --seed_start 0 --seed_end 2