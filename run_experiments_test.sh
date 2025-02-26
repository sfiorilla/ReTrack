#!/bin/bash
set -e
echo "Current directory: $(pwd)"
rm -f ./TrackEval/data/trackers/mot_challenge/MOT17-test/last_execution/data/*
rm -f ./TrackEval/data/trackers/mot_challenge/MOT17-test/last_execution/pedestrian_*
cd ./BoT-SORT
echo "Current directory: $(pwd)"
python3 tools/track.py --eval train --path ./datasets/MOT17 --default-parameters --benchmark MOT17 -f yolox/exps/example/mot/yolox_x_mix_det.py -c pretrained/bytetrack_x_mot17.pth.tar  --with-reid --with-reid --fp16 --fuse --conf 0.01 --nms 0.7 --match_thresh 0.8 --min_box_area 100 
#Remember to adapt txt_path and save_path before launching interpolation
python3 tools/interpolation_adjusted.py 

