#!/bin/bash

# Training Parameters
# Ablation tasks: door-open-v2, window-open-v2, lever-pull-v2
TASK_NAME="lever-pull-v2" # ("button-press-v2" "door-close-v2"  "window-open-v2" "stick-push-v2" "lever-pull-v2")
REWARD_FN="coverage" 
SEED=r

# USE_CKPT=false
USE_CKPT=true

NUM_DEMOS=4
MISMATCHED=true
NUM_FRAMES="d" # d for default
CAMERA_NAME="d" # d for default (defined in env_utils.CAMERA)
# Parameters for random mismatched demos
RANDOM_MISMATCHED=false 
NUM_SECS=5  # Only used if RANDOM_MISMATCHED=true
MISMATCHED_LEVEL=3  # Only used if RANDOM_MISMATCHED=true
SPEED_TYPE='slow' # Only used if RANDOM_MISMATCHED=true, options are 'slow', 'fast', 'mixed'
RANDOM_MISMATCHED_RUN_NUM=0  # Only used if RANDOM_MISMATCHED=true
# Parameters for multi video ablation
MULTI_VIDEO_ABLATION=false
NUM_RANDOM_SPEED_VIDEOS=4

DISCOUNT_FACTOR=0.9
MASK_K=2
TAU=1
THRESHOLD=0.9 # only used by the baseline "threshold", which track the progress based on the threshold

INCLUDE_TIMESTEP=true
TRACK_PROGRESS=false
ADS=false
USE_MAX_REWARD=true  # Default is true

TRAIN_STEPS=500000
# TRAIN_STEPS=1000000

# Logging Parameters
WANDB_MODE="online"
VIDEO_PERIOD=1200 
EVAL_PERIOD=10000
MODEL_PERIOD=100000

python main.py \
    train_steps=${TRAIN_STEPS} \
    env_name=${TASK_NAME} \
    reward_fn=${REWARD_FN} \
    track_progress=${TRACK_PROGRESS} \
    ads=${ADS} \
    use_max_reward=${USE_MAX_REWARD} \
    mismatched=${MISMATCHED} \
    num_frames=${NUM_FRAMES} \
    random_mismatched=${RANDOM_MISMATCHED} \
    num_secs=${NUM_SECS} \
    mismatched_level=${MISMATCHED_LEVEL} \
    speed_type=${SPEED_TYPE} \
    random_mismatched_run_num=${RANDOM_MISMATCHED_RUN_NUM} \
    multi_video_ablation=${MULTI_VIDEO_ABLATION} \
    num_random_speed_videos=${NUM_RANDOM_SPEED_VIDEOS} \
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
    video_period=${VIDEO_PERIOD}
