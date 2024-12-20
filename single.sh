#!/bin/bash

# Job Parameters
PARTITION="gpu" # "portal"
CPUS=8
GPUS=1
MEMORY=35GB
TIME="4:00:00"

# Training Parameters
TASK_NAME="button-press-v2" # ("button-press-v2" "door-close-v2" "door-open-v2" "window-open-v2" "stick-push-v2" "lever-pull-v2")
REWARD_FN="temporal_ot"
SEED=123
NUM_DEMOS=2
CAMERA_NAME="d" # d for default (defined in env_utils.CAMERA)
DISCOUNT_FACTOR=0.9

# Logging Parameters
WANDB_MODE="disabled"
VIDEO_PERIOD=400 
EVAL_PERIOD=10000

python main.py \
    --env_name ${TASK_NAME} \
    --reward_fn ${REWARD_FN} \
    --obs_type "features" \
    --seed ${SEED} \
    --gamma ${DISCOUNT_FACTOR} \
    --num_demos ${NUM_DEMOS} \
    --camera_name ${CAMERA_NAME} \
    --wandb_mode ${WANDB_MODE} \
    --eval_step_period ${EVAL_PERIOD} \
    --video_episode_period ${VIDEO_PERIOD}