#!/bin/bash

# Job Parameters
PARTITION="gpu"
CPUS=2
GPUS=1
MEMORY=35GB
TIME="8:00:00"

# Training Parameters
TASK_NAME=("button-press-v2" "door-open-v2") # "lever-pull-v2" "window-open-v2"  "push-v2" )  # "door-open-v2" )
TAU=1
REWARD_FN=("coverage")  # "final_frame" "temporal_ot" 
SEED=("r" "r") #"r" "r") #117 67 89)
NUM_DEMOS=2
CAMERA_NAME="d" # d for default (defined in env_utils.CAMERA)
DISCOUNT_FACTOR=0.9 # (0.9 0.99)
MASK_K=10
INCLUDE_TIMESTEP=true
TRACK_PROGRESS=false
TRAIN_STEPS=500000

# Logging Parameters
WANDB_MODE="online"
VIDEO_PERIOD=1200 
EVAL_PERIOD=10000
MODEL_PERIOD=100000

# Loop through TASK_NAME and TAU
for task_name_i in "${TASK_NAME[@]}"; do
    for reward_fn_i in "${REWARD_FN[@]}"; do
        for seed_i in "${SEED[@]}"; do
            sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=train-${task_name}-${tau}
#SBATCH --partition=${PARTITION}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --mem=${MEMORY}
#SBATCH --time=${TIME}
#SBATCH --output=dump/train_${task_name}_${tau}_%j.out
#SBATCH --error=dump/train_${task_name}_${tau}_%j.err

# Capture the Slurm job ID
job_id=\$SLURM_JOB_ID

echo "Running training for task: ${task_name_i} with seed: ${seed_i}, job ID: \$job_id"
python main.py \
    env_name=${task_name_i} \
    reward_fn=${reward_fn_i} \
    obs_type="features" \
    seed=${seed_i} \
    discount_factor=${DISCOUNT_FACTOR} \
    track_progress=${TRACK_PROGRESS} \
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
