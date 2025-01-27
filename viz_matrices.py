from models import ResNet, DDPGAgent
from utils import load_gif_frames
from demo import get_demo_gif_path
from seq_matching import load_matching_fn

import torch
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda'

random_mismatched_info = {
    'mismatch_level': f'1outof5_mismatched',
    'run_num': 1,
    'speed_type': "slow"
}

demo_path = get_demo_gif_path("metaworld", 
                              task_name="door-open-v2", 
                              camera_name="corner3", 
                              demo_num=0, 
                              num_frames="d", 
                              mismatched=False, random_mismatched_info=random_mismatched_info)
print(f"Loading demo from {demo_path}")

run_folder_path = "/share/portal/hw575/TemporalOT/train_logs/2025-01-23-20-29-39-358623_envt=door-open-v2_rm=coverage_bf1bfe"

demo_gif = load_gif_frames(demo_path, "torch")

cost_encoder = ResNet().to(device)
_ = cost_encoder.eval()
with torch.no_grad():
    demos = [cost_encoder(demo_gif.to(device))]

# get the custom reward function
matching_fn_cfg = {
    "tau": 1,
    "ent_reg": 0.01,
    "mask_k": 2,
    "sdtw_smoothing": 5,
    "track_progress": False,
    "threshold": 0.9
}

reward_fn = load_matching_fn("coverage", matching_fn_cfg)

action_shape=np.random.randint(1, 10, size=(10, 3)).shape
obs_shape=np.random.randint(1, 10, size=(20, 5)).shape

agent = DDPGAgent(reward_fn=reward_fn,
                    obs_shape=obs_shape,
                    action_shape=action_shape,
                    device=device,
                    lr=1e-4,
                    env_horizon=125,
                    feature_dim=50,
                    hidden_dim=1024,
                    critic_target_tau=0.005,
                    stddev=0.1,
                    stddev_clip=0.3,
                    rew_scale=1, # this will be updated after first train iter
                    auto_rew_scale_factor=10,
                    context_num=3,
                    use_encoder=False)

agent.init_demos(cost_encoder, demos)

print(agent.demos[0].shape)

learner_gif_path = "/share/portal/hw575/TemporalOT/wandb/run-20250123_203038-8wu58n4q/files/media/videos/trajectory/video_138_16f793f6235ded611ce3.gif"

learner_gif = load_gif_frames(learner_gif_path, "torch")

print(type(learner_gif))
print(learner_gif.size())

rewards, info = agent.rewarder(learner_gif.to(device))
# print(f"Info assignment: {info['assignment']}")
# print(f"Info cost: {info['cost_matrix']}")

assignment = info['assignment']
diffs = np.zeros(assignment.shape)

for i in range(1, assignment.shape[0]):
    print(f"i-1: {i-1}, {assignment[i-1]}")
    print(f"i: {i}, {assignment[i]}")
    print(f"diff: {assignment[i] - assignment[i-1]}")
    # input("stop")
    diffs[i] = (assignment[i] - assignment[i-1]) == 0

def plot_heatmap(matrix, title, cmap):
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap=cmap, aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Reference Trajectory")
    plt.ylabel("Learner Trajectory")

    # Save the plot to a BytesIO buffer
    plt.savefig(f"main_fig_plots/{title}.png", format="png")
    plt.close()

plot_heatmap(info['cost_matrix'], "cost_matrix", "gray_r")
plot_heatmap(assignment, "assignment", "Blues")
plot_heatmap(diffs, "diffs", "Greens")
