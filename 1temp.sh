#!/usr/bin/env bash

python chinese_train.py --model_type 4 --dgl_type lstm --weight_define 1 --dgl_layer 1 --seed_num 1 --window_size 1 --model_name dgl1_p4_lstm_w1_size1
python chinese_train.py --model_type 4 --dgl_type lstm --weight_define 1 --dgl_layer 1 --seed_num 100 --window_size 1 --model_name dgl1_p4_lstm_w1_size1
python newtest.py --type_id 3 --model_name dgl1_p4_lstm_w1_size1
python chinese_train.py --model_type 4 --dgl_type lstm --weight_define 2 --dgl_layer 1 --seed_num 1 --window_size 1 --model_name dgl1_p4_lstm_w2_size1
python chinese_train.py --model_type 4 --dgl_type lstm --weight_define 2 --dgl_layer 1 --seed_num 100 --window_size 1 --model_name dgl1_p4_lstm_w2_size1
python newtest.py --type_id 3 --model_name dgl1_p4_lstm_w2_size1