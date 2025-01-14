#!/bin/bash

# Training Parameters
TASK_NAME="door-close-v2" # ("button-press-v2" "door-close-v2"  "window-open-v2" "stick-push-v2" "lever-pull-v2")
REWARD_FN="dtw" 
SEED="r"

NUM_DEMOS=1
MISMATCHED=true
NUM_FRAMES="d" # d for default (if it's defined, it will search under mistmatched/subsampled_{NUM_FRAMES})
CAMERA_NAME="d" # d for default (defined in env_utils.CAMERA)

DISCOUNT_FACTOR=0.9
MASK_K=10
TAU=10
THRESHOLD=0.9 # only used by the baseline "threshold", which track the progress based on the threshold
INCLUDE_TIMESTEP=true
TRACK_PROGRESS=false
ADS=false

# Logging Parameters
WANDB_MODE="online"
VIDEO_PERIOD=400 
EVAL_PERIOD=10000
MODEL_PERIOD=100000

python main.py \
    env_name=${TASK_NAME} \
    reward_fn=${REWARD_FN} \
    track_progress=${TRACK_PROGRESS} \
    ads=${ADS} \
    mismatched=${MISMATCHED} \
    num_frames=${NUM_FRAMES} \
    obs_type="features" \
    seed=${SEED} \
    discount_factor=${DISCOUNT_FACTOR} \
    num_demos=${NUM_DEMOS} \
    camera_name=${CAMERA_NAME} \
    wandb_mode=${WANDB_MODE} \
    mask_k=${MASK_K} \
    tau=${TAU} \
    threshold=${THRESHOLD} \
    include_timestep=${INCLUDE_TIMESTEP} \
    eval_period=${EVAL_PERIOD} \
    model_period=${MODEL_PERIOD} \
    video_period=${VIDEO_PERIOD} \
    
