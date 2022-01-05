#!/usr/bin/env bash

python chinese_train.py --gcn_conv_type 0 --model_type 4 --dgl_type pool --weight_define 3 --dgl_layer 1 --seed_num 200 --window_size 1  --model_name  dgl1_p4_pool_w3_size1
python chinese_train.py --gcn_conv_type 0 --model_type 4 --dgl_type pool --weight_define 3 --dgl_layer 1 --seed_num 300 --window_size 1  --model_name  dgl1_p4_pool_w3_size1
python chinese_train.py --gcn_conv_type 0 --model_type 4 --dgl_type pool --weight_define 3 --dgl_layer 1 --seed_num 400 --window_size 1  --model_name  dgl1_p4_pool_w3_size1
python chinese_train.py --gcn_conv_type 0 --model_type 4 --dgl_type pool --weight_define 3 --dgl_layer 1 --seed_num 500 --window_size 1  --model_name  dgl1_p4_pool_w3_size1

python newtest.py --type_id 3 --model_name  dgl1_p4_pool_w3_size1 --seed_start 0 --seed_end 6