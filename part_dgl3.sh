#!/usr/bin/env bash

python chinese_train.py --gcn_conv_type 0 --model_type 4 --dgl_type lstm --weight_define 3 --dgl_layer 2 --add_self_loop 1 --seed_num 200 --window_size 3 --model_name dgl2_p4_lstm_w3_size1_loop
python chinese_train.py --gcn_conv_type 0 --model_type 4 --dgl_type lstm --weight_define 3 --dgl_layer 2 --add_self_loop 1 --seed_num 300 --window_size 3 --model_name dgl2_p4_lstm_w3_size1_loop
python chinese_train.py --gcn_conv_type 0 --model_type 4 --dgl_type lstm --weight_define 3 --dgl_layer 2 --add_self_loop 1 --seed_num 400 --window_size 3 --model_name dgl2_p4_lstm_w3_size1_loop
python chinese_train.py --gcn_conv_type 0 --model_type 4 --dgl_type lstm --weight_define 3 --dgl_layer 2 --add_self_loop 1 --seed_num 500 --window_size 3 --model_name dgl2_p4_lstm_w3_size1_loop

python newtest.py --type_id 3 --model_name dgl2_p4_lstm_w3_size1_loop --seed_start 0 --seed_end 6