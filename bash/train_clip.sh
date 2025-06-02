#!/bin/bash
python vlm_eval/clip_train.py \
    --num_epochs 1 \
    --data_seeds 115 \
    --data_name MS_COCO \
    --method NONE \
    --batch_size 128 \
    --learning_rate 5e-7 \
    --save_model \
    --save_model_path ./fine_tuned_clip_models/NONE/

