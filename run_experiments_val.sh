#!/bin/bash
set -e
echo "Hi, The Valid test starts!"
echo "Current directory: $(pwd)"
rm -f ./TrackEval/data/trackers/mot_challenge/MOT17-train/last_execution/data/*
rm -f ./TrackEval/data/trackers/mot_challenge/MOT17-train/last_execution/pedestrian_*
cd ./BoT-SORT
echo "Current directory: $(pwd)"

seqlist=("02" "04" "05" "09" "10" "11" "13")
valframes=("--start_frame_no 302" "--start_frame_no 527" "--start_frame_no 420" "--start_frame_no 264" "--start_frame_no 329" "--start_frame_no 452" "--start_frame_no 377")

for index in "${!seqlist[@]}"; do
    seq="${seqlist[$index]}"
    frame="${valframes[$index]}"
    echo "Running Seq"$seq" in validation mode"
    #TEST: python3 tools/demo.py image --path ./datasets/MOT17/train/MOT17-"$seq"-SDP/img1 -f yolox/exps/example/mot/yolox_x_mix_det.py -c pretrained/bytetrack_x_mot17.pth.tar --with-reid --fuse-score --fp16 --fuse --save_result
    #VALID: 
    python3 tools/demo.py image --path ./datasets/MOT17/train/MOT17-"$seq"-SDP/img1 -f yolox/exps/example/mot/yolox_x_ablation.py -c pretrained/bytetrack_ablation.pth.tar --with-reid --fuse-score --fp16 $frame --fuse --save_result

done

python3 tools/interpolation_adjusted.py 
cd ../TrackEval/
echo "Metrics evaluation"
python scripts/run_mot_challenge.py --BENCHMARK MOT17 --SPLIT_TO_EVAL train --TRACKERS_TO_EVAL last_execution --METRICS HOTA CLEAR Identity VACE --USE_PARALLEL False --NUM_PARALLEL_CORES 1  --PLOT_CURVES False
echo "The test is completed."

