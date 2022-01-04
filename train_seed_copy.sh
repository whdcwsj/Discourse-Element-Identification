#!/usr/bin/env bash

python3 newtrain.py --seed_num 1  --weight_id 2 --model_package_name newbaseline_stress_pid
python3 newtrain.py --seed_num 100 --weight_id 2 --model_package_name newbaseline_stress_pid
python3 newtrain.py --seed_num 200 --weight_id 2 --model_package_name newbaseline_stress_pid
python3 newtrain.py --seed_num 300 --weight_id 2 --model_package_name newbaseline_stress_pid
python3 newtrain.py --seed_num 400 --weight_id 2 --model_package_name newbaseline_stress_pid
python3 newtrain.py --seed_num 500 --weight_id 2 --model_package_name newbaseline_stress_pid
python3 newtrain.py --seed_num 600 --weight_id 2 --model_package_name newbaseline_stress_pid
python3 newtrain.py --seed_num 700 --weight_id 2 --model_package_name newbaseline_stress_pid
python3 newtrain.py --seed_num 800 --weight_id 2 --model_package_name newbaseline_stress_pid
python3 newtrain.py --seed_num 900 --weight_id 2 --model_package_name newbaseline_stress_pid
python3 newtrain.py --seed_num 1000 --weight_id 2 --model_package_name newbaseline_stress_pid
python3 newtest.py --type_id 0 --seed_start 0 --seed_end 11 --model_name newbaseline_stress_pid