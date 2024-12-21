#!/bin/bash

# Training Parameters
TASK_NAME="button-press-v2" # ("button-press-v2" "door-close-v2" "door-open-v2" "window-open-v2" "stick-push-v2" "lever-pull-v2")
REWARD_FN="log_coverage"
MASK_K=10
SEED=123
NUM_DEMOS=2
CAMERA_NAME="d" # d for default (defined in env_utils.CAMERA)
DISCOUNT_FACTOR=0.9
TAU=10

# Logging Parameters
WANDB_MODE="online"
VIDEO_PERIOD=400 
EVAL_PERIOD=10000
MODEL_PERIOD=100000

python main.py \
    env_name=${TASK_NAME} \
    reward_fn=${REWARD_FN} \
    obs_type="features" \
    seed=${SEED} \
    discount_factor=${DISCOUNT_FACTOR} \
    num_demos=${NUM_DEMOS} \
    camera_name=${CAMERA_NAME} \
    wandb_mode=${WANDB_MODE} \
    mask_k=${MASK_K} \
    tau=${TAU} \
    eval_period=${EVAL_PERIOD} \
    model_period=${MODEL_PERIOD} \
    video_period=${VIDEO_PERIOD}
    