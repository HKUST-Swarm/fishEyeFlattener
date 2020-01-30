#!/bin/bash
rosrun camera_model Calibration --camera-name mycamera --input $1 -p IMG -e png -w 8 -h 12 --size 80 --camera-model myfisheye --opencv false

