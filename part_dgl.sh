#!/usr/bin/env bash

python newtrain_en_ft.py  --dgl_type pool --weight_define 1 --dgl_layer 1 --add_self_loop 1 --window_size 2 --seed_num 1 --model_name new_dgl1_p4_pool_w1_size2_loop
python newtrain_en_ft.py  --dgl_type pool --weight_define 1 --dgl_layer 1 --add_self_loop 1 --window_size 2 --seed_num 100 --model_name new_dgl1_p4_pool_w1_size2_loop
python newtrain_en_ft.py  --dgl_type pool --weight_define 1 --dgl_layer 1 --add_self_loop 1 --window_size 2 --seed_num 200 --model_name new_dgl1_p4_pool_w1_size2_loop
python newtrain_en_ft.py  --dgl_type pool --weight_define 1 --dgl_layer 1 --add_self_loop 1 --window_size 2 --seed_num 300 --model_name new_dgl1_p4_pool_w1_size2_loop