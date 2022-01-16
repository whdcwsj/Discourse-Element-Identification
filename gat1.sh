#!/usr/bin/env bash

python chinese_train.py --gcn_conv_type 1 --model_type 3 --weight_define 1 --dgl_layer 3 --add_self_loop 1 --seed_num 1 --num_head 3 --model_name gat3_p3_w1_head3_loop
python chinese_train.py --gcn_conv_type 1 --model_type 3 --weight_define 1 --dgl_layer 3 --add_self_loop 1 --seed_num 100 --num_head 3 --model_name gat3_p3_w1_head3_loop
python chinese_train.py --gcn_conv_type 1 --model_type 3 --weight_define 1 --dgl_layer 3 --add_self_loop 1 --seed_num 200 --num_head 3 --model_name gat3_p3_w1_head3_loop
python chinese_train.py --gcn_conv_type 1 --model_type 3 --weight_define 1 --dgl_layer 3 --add_self_loop 1 --seed_num 300 --num_head 3 --model_name gat3_p3_w1_head3_loop
python chinese_train.py --gcn_conv_type 1 --model_type 3 --weight_define 1 --dgl_layer 3 --add_self_loop 1 --seed_num 400 --num_head 3 --model_name gat3_p3_w1_head3_loop
python chinese_train.py --gcn_conv_type 1 --model_type 3 --weight_define 1 --dgl_layer 3 --add_self_loop 1 --seed_num 500 --num_head 3 --model_name gat3_p3_w1_head3_loop
python chinese_train.py --gcn_conv_type 1 --model_type 3 --weight_define 1 --dgl_layer 3 --add_self_loop 1 --seed_num 600 --num_head 3 --model_name gat3_p3_w1_head3_loop
python chinese_train.py --gcn_conv_type 1 --model_type 3 --weight_define 1 --dgl_layer 3 --add_self_loop 1 --seed_num 700 --num_head 3 --model_name gat3_p3_w1_head3_loop
python chinese_train.py --gcn_conv_type 1 --model_type 3 --weight_define 1 --dgl_layer 3 --add_self_loop 1 --seed_num 800 --num_head 3 --model_name gat3_p3_w1_head3_loop
python chinese_train.py --gcn_conv_type 1 --model_type 3 --weight_define 1 --dgl_layer 3 --add_self_loop 1 --seed_num 900 --num_head 3 --model_name gat3_p3_w1_head3_loop
python chinese_train.py --gcn_conv_type 1 --model_type 3 --weight_define 1 --dgl_layer 3 --add_self_loop 1 --seed_num 1000 --num_head 3 --model_name gat3_p3_w1_head3_loop

python newtest.py --type_id 3 --model_name gat3_p3_w1_head3_loop --seed_start 0 --seed_end 11