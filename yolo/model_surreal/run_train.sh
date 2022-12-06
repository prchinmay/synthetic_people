#!/usr/bin/env bash

# SPECIFY ENVIRONMENT
#cd /home/2014-0353_generaleye/Huy/YOLOv3
#source activate venv/conda_env_test

# GO TO THE MODEL
cd /home/2014-0353_generaleye/Chinmay/data/yolo_output/Model_surreal
python3 finetune.py >/home/2014-0353_generaleye/Chinmay/data/yolo_output/Model_surreal/linux_logs/lr_04_25k.log