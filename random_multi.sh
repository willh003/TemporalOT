#!/bin/bash

# Job Parameters
PARTITION="gpu"
CPUS=2
GPUS=1
MEMORY=35GB
TIME="8:00:00"

# Training Parameters
TASK_NAME=("door-open-v2") 
REWARD_FN=("coverage") 
# Fast: door open
SEED_MAPPING=(
    "1_0:217"
    "1_1:42"
    "1_2:392"
    "3_0:706"
    "3_1:615"
    "3_2:213"
    "5_0:779"
    "5_1:872"
    "5_2:298"
)
# Fast: window open
# SEED_MAPPING=(
#     "1_0:550"
#     "1_1:461"
#     "1_2:883"
#     "3_0:942"
#     "3_1:739"
#     "3_2:468"
#     "5_0:921"
#     "5_1:37"
#     "5_2:192"
# )
# Fast: lever pull
# SEED_MAPPING=(
#     "1_0:258"
#     "1_1:634"
#     "1_2:590"
#     "3_0:21"
#     "3_1:110"
#     "3_2:168"
#     "5_0:230"
#     "5_1:844"
#     "5_2:471"
# )

########################################
# Slow: door open
# SEED_MAPPING=(
#     "1_0:305"
#     "1_1:521"
#     "1_2:673"
#     "3_0:85"
#     "3_1:864"
#     "3_2:638"
#     "5_0:693"
#     "5_1:275"
#     "5_2:374"
# )
# Slow: window open 
# SEED_MAPPING=(
#     "1_0:319"
#     "1_1:607"
#     "1_2:82"
#     "3_0:226"
#     "3_1:853"
#     "3_2:601"
#     "5_0:223"
#     "5_1:641"
#     "5_2:839"
# )
# Slow: lever pull 
# SEED_MAPPING=(
#     "1_0:611"
#     "1_1:689"
#     "1_2:351"
#     "3_0:884"
#     "3_1:858"
#     "3_2:23"
#     "5_0:116"
#     "5_1:349"
#     "5_2:834"
# )


USE_CKPT=true

NUM_DEMOS=1
MISMATCHED=false
NUM_FRAMES="d" 
CAMERA_NAME="d" 
# Parameters for random mismatched demos
RANDOM_MISMATCHED=true 
NUM_SECS=5  
MISMATCHED_LEVELS=(1 3) # Define the mismatched levels to iterate over
RANDOM_MISMATCHED_RUN_NUMS=(0 1 2) # Define the run numbers to iterate over
SPEED_TYPE='fast' 

DISCOUNT_FACTOR=0.9 
MASK_K=2
TAU=1
THRESHOLD=0.9 

INCLUDE_TIMESTEP=true
TRACK_PROGRESS=false
ADS=false

TRAIN_STEPS=500000

# Logging Parameters
WANDB_MODE="online"
VIDEO_PERIOD=1200 
EVAL_PERIOD=10000
MODEL_PERIOD=100000

# Function to get the seed for a given combination of mismatched_level and run_num
get_seed() {
    local level=$1
    local run_num=$2
    for mapping in "${SEED_MAPPING[@]}"; do
        IFS=':' read -r key value <<< "$mapping"
        if [ "$key" == "${level}_${run_num}" ]; then
            echo "$value"
            return
        fi
    done
    echo "" # Return empty if no mapping is found
}

# Loop through tasks, rewards, mismatched levels, run numbers
for task_name_i in "${TASK_NAME[@]}"; do
    for reward_fn_i in "${REWARD_FN[@]}"; do
        for mismatched_level_i in "${MISMATCHED_LEVELS[@]}"; do
            for random_mismatched_run_num_i in "${RANDOM_MISMATCHED_RUN_NUMS[@]}"; do
                seed=$(get_seed "$mismatched_level_i" "$random_mismatched_run_num_i")
                if [ -z "$seed" ]; then
                    echo "No seed found for mismatched_level=${mismatched_level_i}, random_mismatched_run_num=${random_mismatched_run_num_i}, skipping..."
                    continue
                fi
                sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=train-${task_name_i}-${tau}
#SBATCH --partition=${PARTITION}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --mem=${MEMORY}
#SBATCH --time=${TIME}
#SBATCH --output=dump/train_${task_name_i}_${reward_fn_i}_%j.out
#SBATCH --error=dump/train_${task_name_i}_${reward_fn_i}_%j.err

# Capture the Slurm job ID
job_id=\$SLURM_JOB_ID

echo "Running training for task: ${task_name_i} with mismatched_level=${mismatched_level_i}, run_num=${random_mismatched_run_num_i}, seed=${seed}, job ID: \$job_id"
python main.py \
    env_name=${task_name_i} \
    reward_fn=${reward_fn_i} \
    use_ckpt=${USE_CKPT} \
    obs_type="features" \
    seed=${seed} \
    discount_factor=${DISCOUNT_FACTOR} \
    track_progress=${TRACK_PROGRESS} \
    ads=${ADS} \
    mismatched=${MISMATCHED} \
    num_frames=${NUM_FRAMES} \
    random_mismatched=${RANDOM_MISMATCHED} \
    num_secs=${NUM_SECS} \
    mismatched_level=${mismatched_level_i} \
    speed_type=${SPEED_TYPE} \
    random_mismatched_run_num=${random_mismatched_run_num_i} \
    obs_type="features" \
    num_demos=${NUM_DEMOS} \
    camera_name=${CAMERA_NAME} \
    mask_k=${MASK_K} \
    tau=${TAU} \
    include_timestep=${INCLUDE_TIMESTEP} \
    train_steps=${TRAIN_STEPS} \
    eval_period=${EVAL_PERIOD} \
    model_period=${MODEL_PERIOD} \
    video_period=${VIDEO_PERIOD} \
    wandb_mode=${WANDB_MODE}
EOF
                sleep 1.1 # Ensure a unique timestamp for each run
            done
        done
    done
done
