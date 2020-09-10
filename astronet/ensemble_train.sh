#!/bin/bash

for i in {1..10}
do
    echo "Training model ${i}"
    python astronet/train.py \
        --model=AstroCNNModel \
        --config_name=local_global_new_tuned \
        --train_files='/mnt/tess/astronet/tfrecords-13/test-0000[0-6]*' \
        --eval_files='/mnt/tess/astronet/tfrecords-13/test-0000[7-9]*' \
        --train_steps=20000 \
        --model_dir="/mnt/tess/astronet/checkpoints/local_global_new_13/${i}"
done

