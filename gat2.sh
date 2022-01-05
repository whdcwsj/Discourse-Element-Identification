#!/usr/bin/env bash

python chinese_train.py --gcn_conv_type 1 --model_type 3 --dgl_type lstm --weight_define 3 --dgl_layer 1 --seed_num 1 --num_head 3 --model_name gat1_p3_lstm_w3_head3
python chinese_train.py --gcn_conv_type 1 --model_type 3 --dgl_type lstm --weight_define 3 --dgl_layer 1 --seed_num 100 --num_head 3 --model_name gat1_p3_lstm_w3_head3
python chinese_train.py --gcn_conv_type 1 --model_type 3 --dgl_type lstm --weight_define 3 --dgl_layer 1 --seed_num 200 --num_head 3 --model_name gat1_p3_lstm_w3_head3
python chinese_train.py --gcn_conv_type 1 --model_type 3 --dgl_type lstm --weight_define 3 --dgl_layer 1 --seed_num 300 --num_head 3 --model_name gat1_p3_lstm_w3_head3
python chinese_train.py --gcn_conv_type 1 --model_type 3 --dgl_type lstm --weight_define 3 --dgl_layer 1 --seed_num 400 --num_head 3 --model_name gat1_p3_lstm_w3_head3
python chinese_train.py --gcn_conv_type 1 --model_type 3 --dgl_type lstm --weight_define 3 --dgl_layer 1 --seed_num 500 --num_head 3 --model_name gat1_p3_lstm_w3_head3
python chinese_train.py --gcn_conv_type 1 --model_type 3 --dgl_type lstm --weight_define 3 --dgl_layer 1 --seed_num 600 --num_head 3 --model_name gat1_p3_lstm_w3_head3
python chinese_train.py --gcn_conv_type 1 --model_type 3 --dgl_type lstm --weight_define 3 --dgl_layer 1 --seed_num 700 --num_head 3 --model_name gat1_p3_lstm_w3_head3
python chinese_train.py --gcn_conv_type 1 --model_type 3 --dgl_type lstm --weight_define 3 --dgl_layer 1 --seed_num 800 --num_head 3 --model_name gat1_p3_lstm_w3_head3
python chinese_train.py --gcn_conv_type 1 --model_type 3 --dgl_type lstm --weight_define 3 --dgl_layer 1 --seed_num 900 --num_head 3 --model_name gat1_p3_lstm_w3_head3
python chinese_train.py --gcn_conv_type 1 --model_type 3 --dgl_type lstm --weight_define 3 --dgl_layer 1 --seed_num 1000 --num_head 3 --model_name gat1_p3_lstm_w3_head3

python newtest.py --type_id 3 --model_name gat1_p3_lstm_w3_head3 --seed_start 0 --seed_end 11