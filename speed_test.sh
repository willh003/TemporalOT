#!/bin/bash

# Job Parameters
PARTITION="gpu"
CPUS=2
GPUS=1
MEMORY=35GB
TIME="20:00:00"

# Training Parameters
# All tasks in order: ("button-press-v2" "door-close-v2" "door-open-v2" "window-open-v2" "lever-pull-v2" "hand-insert-v2" "push-v2" "basketball-v2" "stick-push-v2" "door-lock-v2")
TASK_NAME=button-press-v2  #"window-open-v2" 
REWARD_FN=("coverage" "temporal_ot" "ot" "liv_text" "threshold" "dtw" ) # ("threshold" "ot" "temporal_ot" "dtw" "coverage")
SEED=44 # "r" indicates a random seed

USE_CKPT=false

NUM_DEMOS=1
MISMATCHED=true
NUM_FRAMES="d" # d for default (if it's defined, it will search under mistmatched/subsampled_{NUM_FRAMES})
CAMERA_NAME="d" # d for default (defined in env_utils.CAMERA)
# Parameters for random mismatched demos
RANDOM_MISMATCHED=false 
NUM_SECS=5  # Only used if RANDOM_MISMATCHED=true
MISMATCHED_LEVEL=3 # Only used if RANDOM_MISMATCHED=true
SPEED_TYPE='fast' # Only used if RANDOM_MISMATCHED=true, options are 'slow', 'fast', 'mixed'
RANDOM_MISMATCHED_RUN_NUM=2 # Only used if RANDOM_MISMATCHED=true

OBS_TYPE='features' # pixels for image based, features for ground truth state based=
DISCOUNT_FACTOR=0.9 # (0.9 0.99)
MASK_K=2
TAU=1
THRESHOLD=0.9 # only used by the baseline "threshold", which track the progress based on the threshold

INCLUDE_TIMESTEP=true 
TRACK_PROGRESS=false
ADS=false

TRAIN_STEPS=20000

# Logging Parameters
WANDB_MODE="online"
VIDEO_PERIOD=10000 
EVAL_PERIOD=10000
MODEL_PERIOD=100000
WANDB_TAGS="['speed_test']"

for reward_fn_i in "${REWARD_FN[@]}"; do
python main.py \
    env_name=${TASK_NAME} \
    reward_fn=${reward_fn_i} \
    use_ckpt=${USE_CKPT} \
    obs_type="features" \
    seed=${SEED} \
    discount_factor=${DISCOUNT_FACTOR} \
    track_progress=${TRACK_PROGRESS} \
    ads=${ADS} \
    mismatched=${MISMATCHED} \
    num_frames=${NUM_FRAMES} \
    random_mismatched=${RANDOM_MISMATCHED} \
    num_secs=${NUM_SECS} \
    mismatched_level=${MISMATCHED_LEVEL} \
    speed_type=${SPEED_TYPE} \
    random_mismatched_run_num=${RANDOM_MISMATCHED_RUN_NUM} \
    obs_type=${OBS_TYPE} \
    num_demos=${NUM_DEMOS} \
    camera_name=${CAMERA_NAME} \
    mask_k=${MASK_K} \
    tau=${TAU} \
    include_timestep=${INCLUDE_TIMESTEP} \
    train_steps=${TRAIN_STEPS} \
    eval_period=${EVAL_PERIOD} \
    model_period=${MODEL_PERIOD} \
    video_period=${VIDEO_PERIOD} \
    wandb_mode=${WANDB_MODE} \
    wandb_tags=${WANDB_TAGS}
done