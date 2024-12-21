#!/bin/bash

# Job Parameters
PARTITION="gpu" # "portal"
CPUS=8
GPUS=1
MEMORY=35GB
TIME="4:30:00"

# Training Parameters
TASK_NAME=("button-press-v2" "door-open-v2" "window-open-v2") #"stick-push-v2" "lever-pull-v2" "door-close-v2")
TAU=(10 60)
REWARD_FN="coverage"
SEED=123
NUM_DEMOS=2
CAMERA_NAME="d" # d for default (defined in env_utils.CAMERA)
DISCOUNT_FACTOR=0.9
MASK_K=10

# Logging Parameters
WANDB_MODE="online"
VIDEO_PERIOD=400 
EVAL_PERIOD=10000

for task_name in "${TASK_NAME[@]}"; do
    for tau in "${TAU[@]}"; do
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
    --reward_fn ${REWARD_FN} \
    --obs_type "features" \
    --seed ${SEED} \
    --gamma ${DISCOUNT_FACTOR} \
    --num_demos ${NUM_DEMOS} \
    --camera_name ${CAMERA_NAME} \
    --wandb_mode ${WANDB_MODE} \
    --eval_step_period ${EVAL_PERIOD} \
    --video_episode_period ${VIDEO_PERIOD} \
    --mask_k ${MASK_K}
    --tau ${tau}
    --job_id \$job_id
EOF
        sleep 1.1 # Ensure a unique timestamp for each run
    done
done
