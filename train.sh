#!/bin/bash

# Job Parameters
PARTITION="portal" # "gpu"
CPUS=8
GPUS=1
MEMORY=35GB
TIME="4:00:00"

# Training Parameters
TASK_NAME=("door-lock-v2", "hand-insert-v2", "push-v2") # ("button-press-v2" "door-close-v2" "door-open-v2" "window-open-v2" "stick-push-v2" "lever-pull-v2")
SEED=123
NUM_DEMOS=2
CAMERA_NAME="d" # d for default (defined in env_utils.CAMERA)
WANDB_MODE="online"
DISCOUNT_FACTOR=0.9

#REWARD_MODEL="temporal_ot"
#VISUAL_ENCODER="resnet50"
#COST_FN="diagonal_cosine"
#WANDB_MODE="online"
#LOG_FREQ=50000 # Log every LOG_FREQ steps

for task_name in "${TASK_NAME[@]}"; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=${PARTITION}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --mem=${MEMORY}
#SBATCH --time=${TIME}
#SBATCH --output=dump/train_%j.out
#SBATCH --error=dump/train_%j.err

# Capture the Slurm job ID
job_id=\$SLURM_JOB_ID

echo "Running training for task: ${task_name} with job ID: \$job_id"
python main.py \
    --env_name ${task_name} \
    --obs_type "features" \
    --seed ${SEED} \
    --gamma ${DISCOUNT_FACTOR} \
    --num_demos ${NUM_DEMOS} \
    --camera_name ${CAMERA_NAME} \
    --wandb_mode ${WANDB_MODE} \
    --job_id \$job_id
EOF
    sleep 1.1 # Ensure a unique timestamp for each run
done
