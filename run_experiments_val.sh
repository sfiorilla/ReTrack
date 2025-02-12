#!/bin/bash
echo "Hi, The Valid test starts!"
cd ./BoT-SORT
echo "Current directory: $(pwd)"
rm -f ./TrackEval/data/trackers/mot_challenge/MOT17-train/last_execution/data/*

list=("02" "04" "05" "09" "10" "11" "13")
for i in "${list[@]}"
do
    echo "Running Seq$i"
    python3 tools/demo.py image --path ./datasets/MOT17/valid/MOT17-"$i"-DPM/img1 -f yolox/exps/example/mot/yolox_x_mix_det.py -c pretrained/bytetrack_x_mot17.pth.tar --with-reid --fuse-score --fp16 --fuse --save_result
done

cd ../TrackEval/
echo "Metrics evaluation"
python scripts/run_mot_challenge.py --BENCHMARK MOT17 --SPLIT_TO_EVAL train --TRACKERS_TO_EVAL last_execution --METRICS HOTA CLEAR Identity VACE --USE_PARALLEL False --NUM_PARALLEL_CORES 1  --PLOT_CURVES False
echo "The test is completed."




