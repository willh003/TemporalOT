#!/bin/bash

# Job Parameters
PARTITION="portal" # "gpu" 
CPUS=8
GPUS=1
MEMORY=35GB
TIME="4:00:00"

# Training Parameters
TASK_NAME=("button-press-v2" "door-close-v2" "door-lock-v2" "hammer-v2" "box-close-v2" "assembly-v2")
DISCOUNT_FACTOR=0.9
SEED=123
NUM_DEMOS=2

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

echo "python main.py --env_name ${task_name} --obs_type \"features\" --seed ${SEED} --gamma ${DISCOUNT_FACTOR} --num_demos ${NUM_DEMOS}"

python main.py \
    --env_name ${task_name} \
    --obs_type "features" \
    --seed ${SEED} \
    --gamma ${DISCOUNT_FACTOR} \
    --num_demos ${NUM_DEMOS}
EOF
sleep 1.1 # make sure the new wandb folder is different (seconds is the identifier)
done
