#!/usr/bin/env bash

python newtrain_en.py  --dgl_type lstm --weight_define 3 --dgl_layer 1 --window_size 1 --add_self_loop 1 --seed_num 1 --model_name dgl1_p4_lstm_w3_size1_loop
python newtrain_en.py  --dgl_type lstm --weight_define 3 --dgl_layer 1 --window_size 1 --add_self_loop 1 --seed_num 100 --model_name dgl1_p4_lstm_w3_size1_loop
python newtrain_en.py  --dgl_type lstm --weight_define 3 --dgl_layer 1 --window_size 1 --add_self_loop 1 --seed_num 200 --model_name dgl1_p4_lstm_w3_size1_loop
python newtrain_en.py  --dgl_type lstm --weight_define 3 --dgl_layer 1 --window_size 1 --add_self_loop 1 --seed_num 300 --model_name dgl1_p4_lstm_w3_size1_loop
python newtrain_en.py  --dgl_type lstm --weight_define 3 --dgl_layer 1 --window_size 1 --add_self_loop 1 --seed_num 400 --model_name dgl1_p4_lstm_w3_size1_loop