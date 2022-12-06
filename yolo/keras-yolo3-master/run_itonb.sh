#!/usr/bin/env bash
cd /home/2014-0353_generaleye/Huy/YOLOv3
source activate venv/conda_env0
cd keras-yolo3-master
python3 train.py >/home/2014-0353_generaleye/Huy/YOLOv3/keras-yolo3-master/train.log
