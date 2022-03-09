#!/usr/bin/env bash

python newtrain_en_ft.py  --dgl_type lstm --weight_define 1 --dgl_layer 1 --window_size 2 --seed_num 1 --model_name dgl1_p4_lstm_w1_size2
python newtrain_en_ft.py  --dgl_type lstm --weight_define 1 --dgl_layer 1 --window_size 2 --seed_num 100 --model_name dgl1_p4_lstm_w1_size2
python newtrain_en_ft.py  --dgl_type lstm --weight_define 1 --dgl_layer 1 --window_size 2 --seed_num 200 --model_name dgl1_p4_lstm_w1_size2
python newtrain_en_ft.py  --dgl_type lstm --weight_define 1 --dgl_layer 1 --window_size 2 --seed_num 300 --model_name dgl1_p4_lstm_w1_size2
python newtrain_en_ft.py  --dgl_type lstm --weight_define 1 --dgl_layer 1 --window_size 2 --seed_num 400 --model_name dgl1_p4_lstm_w1_size2