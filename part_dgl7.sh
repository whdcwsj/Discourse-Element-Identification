#!/usr/bin/env bash

python newtrain_en.py  --dgl_type pool --weight_define 1 --dgl_layer 1 --window_size 1 --seed_num 1 --model_name dgl1_p4_pool_w1_size1
python newtrain_en.py  --dgl_type pool --weight_define 1 --dgl_layer 1 --window_size 1 --seed_num 100 --model_name dgl1_p4_pool_w1_size1
python newtrain_en.py  --dgl_type pool --weight_define 1 --dgl_layer 1 --window_size 1 --seed_num 200 --model_name dgl1_p4_pool_w1_size1
python newtrain_en.py  --dgl_type pool --weight_define 1 --dgl_layer 1 --window_size 1 --seed_num 300 --model_name dgl1_p4_pool_w1_size1
python newtrain_en.py  --dgl_type pool --weight_define 1 --dgl_layer 1 --window_size 1 --seed_num 400 --model_name dgl1_p4_pool_w1_size1