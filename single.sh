#!/bin/bash

# Training Parameters
TASK_NAME="door-open-v2" # ("button-press-v2" "door-close-v2"  "window-open-v2" "stick-push-v2" "lever-pull-v2")
REWARD_FN="coverage" 
SEED="r"

USE_CKPT=true

NUM_DEMOS=1
MISMATCHED=true
NUM_FRAMES="d" # d for default
CAMERA_NAME="d" # d for default (defined in env_utils.CAMERA)
# Parameters for random mismatched demos
RANDOM_MISMATCHED=false 
NUM_SECS=5  # Only used if RANDOM_MISMATCHED=true
MISMATCHED_LEVEL=3  # Only used if RANDOM_MISMATCHED=true
RANDOM_MISMATCHED_RUN_NUM=0 # Only used if RANDOM_MISMATCHED=true

DISCOUNT_FACTOR=0.9
MASK_K=10
TAU=10
THRESHOLD=0.9 # only used by the baseline "threshold", which track the progress based on the threshold

INCLUDE_TIMESTEP=true
TRACK_PROGRESS=false
ADS=false

TRAIN_STEPS=500000

# Logging Parameters
WANDB_MODE="online"
VIDEO_PERIOD=400 
EVAL_PERIOD=10000
MODEL_PERIOD=100000

python main.py \
    train_steps=${TRAIN_STEPS} \
    env_name=${TASK_NAME} \
    reward_fn=${REWARD_FN} \
    track_progress=${TRACK_PROGRESS} \
    ads=${ADS} \
    mismatched=${MISMATCHED} \
    num_frames=${NUM_FRAMES} \
    random_mismatched=${RANDOM_MISMATCHED} \
    num_secs=${NUM_SECS} \
    mismatched_level=${MISMATCHED_LEVEL} \
    random_mismatched_run_num=${RANDOM_MISMATCHED_RUN_NUM} \
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
    use_ckpt=${USE_CKPT} \
    eval_period=${EVAL_PERIOD} \
    model_period=${MODEL_PERIOD} \
    video_period=${VIDEO_PERIOD} \
    
