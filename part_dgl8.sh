#!/usr/bin/env bash

python newtrain_en.py  --dgl_type pool --weight_define 1 --dgl_layer 1 --window_size 1 --seed_num 500 --model_name dgl1_p4_pool_w1_size1
python newtrain_en.py  --dgl_type pool --weight_define 1 --dgl_layer 1 --window_size 1 --seed_num 600 --model_name dgl1_p4_pool_w1_size1
python newtrain_en.py  --dgl_type pool --weight_define 1 --dgl_layer 1 --window_size 1 --seed_num 700 --model_name dgl1_p4_pool_w1_size1
python newtrain_en.py  --dgl_type pool --weight_define 1 --dgl_layer 1 --window_size 1 --seed_num 800 --model_name dgl1_p4_pool_w1_size1
python newtrain_en.py  --dgl_type pool --weight_define 1 --dgl_layer 1 --window_size 1 --seed_num 900 --model_name dgl1_p4_pool_w1_size1
python newtrain_en.py  --dgl_type pool --weight_define 1 --dgl_layer 1 --window_size 1 --seed_num 1000 --model_name dgl1_p4_pool_w1_size1