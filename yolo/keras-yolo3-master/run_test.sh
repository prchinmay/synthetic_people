#!/usr/bin/env bash
cd /home/2014-0353_generaleye/Huy/YOLOv3
source venv/my_env/bin/activate
cd keras-yolo3-master
python3 test.py >/home/2014-0353_generaleye/Huy/YOLOv3/keras-yolo3-master/testing.log
