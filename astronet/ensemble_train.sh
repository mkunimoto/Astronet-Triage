#!/bin/bash

for i in {1..10}
do
    echo "Training model ${i}"
    python astronet/train.py \
        --model=AstroCNNModel \
        --config_name=local_global_new_tuned \
        --train_files='astronet/tfrecords-10/test-0000[0-5]*' \
        --eval_files='astronet/tfrecords-10/test-0000[6-6]*' \
        --train_steps=20000 \
        --model_dir="astronet/checkpoints/local_global_new_10/${i}"
done

