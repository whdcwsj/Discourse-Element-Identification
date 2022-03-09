#!/usr/bin/env bash

python newtrain_en_ft.py  --dgl_type lstm --weight_define 3 --dgl_layer 1 --window_size 2 --seed_num 500 --model_name dgl1_p4_lstm_w3_size2
python newtrain_en_ft.py  --dgl_type lstm --weight_define 3 --dgl_layer 1 --window_size 2 --seed_num 600 --model_name dgl1_p4_lstm_w3_size2
python newtrain_en_ft.py  --dgl_type lstm --weight_define 3 --dgl_layer 1 --window_size 2 --seed_num 700 --model_name dgl1_p4_lstm_w3_size2
python newtrain_en_ft.py  --dgl_type lstm --weight_define 3 --dgl_layer 1 --window_size 2 --seed_num 800 --model_name dgl1_p4_lstm_w3_size2
python newtrain_en_ft.py  --dgl_type lstm --weight_define 3 --dgl_layer 1 --window_size 2 --seed_num 900 --model_name dgl1_p4_lstm_w3_size2
python newtrain_en_ft.py  --dgl_type lstm --weight_define 3 --dgl_layer 1 --window_size 2 --seed_num 1000 --model_name dgl1_p4_lstm_w3_size2
