#!/bin/bash

set -e

for i in {1..10}
do
    echo "Training model ${i}"
    python astronet/train.py \
        --model=AstroCNNModel \
        --config_name=local_global_new_tuned \
        --train_files='/mnt/tess/astronet/tfrecords-14-train/*' \
        --eval_files='/mnt/tess/astronet/tfrecords-14-val/*' \
        --train_steps=20000 \
        --model_dir="/mnt/tess/astronet/checkpoints/local_global_new_14/${i}"
done

