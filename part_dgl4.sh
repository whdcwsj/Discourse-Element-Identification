#!/usr/bin/env bash

sleep 16h
python chinese_train.py --gcn_conv_type 0 --model_type 4 --dgl_type lstm --weight_define 3 --dgl_layer 1 --seed_num 1 --window_size 2  --model_name  dgl1_p4_lstm_w3_size2
python chinese_train.py --gcn_conv_type 0 --model_type 4 --dgl_type lstm --weight_define 3 --dgl_layer 1 --seed_num 100 --window_size 2  --model_name  dgl1_p4_lstm_w3_size2

python newtest.py --type_id 3 --model_name  dgl1_p4_lstm_w3_size2 --seed_start 0 --seed_end 2