#!/bin/bash

# Job Parameters
PARTITION="gpu"
CPUS=8
GPUS=1
MEMORY=35GB
TIME="4:30:00"

# Training Parameters
TASK_NAME=("door-close-v2" "button-press-v2" "window-open-v2") # "door-open-v2" )
TAU=1
REWARD_FN="coverage"
SEED=("r" "r" "r" "r" "r") #117 67 89)
NUM_DEMOS=2
CAMERA_NAME="d" # d for default (defined in env_utils.CAMERA)
DISCOUNT_FACTOR=0.9 # (0.9 0.99)
MASK_K=10
INCLUDE_TIMESTEP=true

# Logging Parameters
WANDB_MODE="online"

# Loop through TASK_NAME and TAU
for task_name_i in "${TASK_NAME[@]}"; do
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

echo "Running training for task: ${TASK_NAME} with seed: ${seed_i}, job ID: \$job_id"
python main.py \
    env_name=${task_name_i} \
    reward_fn=${REWARD_FN} \
    obs_type="features" \
    seed=${seed_i} \
    discount_factor=${DISCOUNT_FACTOR} \
    num_demos=${NUM_DEMOS} \
    camera_name=${CAMERA_NAME} \
    mask_k=${MASK_K} \
    tau=${TAU} \
    include_timestep=${INCLUDE_TIMESTEP}
    wandb_mode=${WANDB_MODE}
EOF
        sleep 1.1 # Ensure a unique timestamp for each run
    done
done
