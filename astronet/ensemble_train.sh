#!/bin/bash

set -e

for i in {1..10}
do
    echo "Training model ${i}"
    python astronet/train.py \
        --model=AstroCNNModel \
        --config_name=extended \
        --train_files='/mnt/tess/astronet/tfrecords-15-train/*' \
        --eval_files='/mnt/tess/astronet/tfrecords-15-val/*' \
        --train_steps=20000 \
        --model_dir="/mnt/tess/astronet/checkpoints/extended_15_run_1/${i}"
done

