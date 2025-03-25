from models import ResNet, DDPGAgent
from utils import load_gif_frames
from demo import get_demo_gif_path
from seq_matching import load_matching_fn

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda'

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--fig_name', type=str, required=True, help='Domain name')
args = parser.parse_args()

fig_name = args.fig_name
# Make the directory
import os
os.makedirs(f"main_fig/{fig_name}", exist_ok=True)

if fig_name == "main":
    # a slow demo
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
elif fig_name == "main_stick-push_orca":
    # a slow demo
    random_mismatched_info = {
        'mismatch_level': f'1outof5_mismatched',
        'run_num': 0,
        'speed_type': "slow"
    }

    demo_path = get_demo_gif_path("metaworld",
                                task_name="stick-push-v2",
                                camera_name="d",
                                demo_num=0,
                                num_frames="d",
                                mismatched=False, random_mismatched_info=random_mismatched_info)
elif fig_name == "film_strip":
    # Default mismatched demo
    demo_path = get_demo_gif_path("metaworld", 
                                task_name="door-open-v2", 
                                camera_name="corner3", 
                                demo_num=0, 
                                num_frames="d", 
                                mismatched=True, random_mismatched_info={})
elif "order_fail" in fig_name:
    # Default mismatched demo
    demo_path = get_demo_gif_path("metaworld",
                                task_name="lever-pull-v2",
                                camera_name="corner3",
                                demo_num=0,
                                num_frames="d",
                                mismatched=True, random_mismatched_info={})
elif "dtw_fail" in fig_name:
    # Default mismatched demo
    demo_path = get_demo_gif_path("metaworld",
                                task_name="button-press-v2",
                                camera_name="corner",
                                demo_num=0,
                                num_frames="d",
                                mismatched=True, random_mismatched_info={})
elif "pretrain_fail" in fig_name:
    # Default matched demo  
    demo_path = get_demo_gif_path("metaworld",
                                task_name="stick-push-v2",
                                camera_name="d",
                                demo_num=0,
                                num_frames="d",
                                mismatched=False, random_mismatched_info={})
                                
    
print(f"Loading demo from {demo_path}")

# run_folder_path = "/share/portal/hw575/TemporalOT/train_logs/2025-01-23-20-29-39-358623_envt=door-open-v2_rm=coverage_bf1bfe"

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

if fig_name == "main":
    reward_fn = load_matching_fn("coverage", matching_fn_cfg)
elif fig_name == "film_strip":
    reward_fn = load_matching_fn("temporal_ot", matching_fn_cfg)
elif "order_fail" in fig_name:
    if "tot" in fig_name:
        if "tot10" in fig_name:
            matching_fn_cfg["mask_k"] = 10
        elif "tot2" in fig_name:
            matching_fn_cfg["mask_k"] = 2
        reward_fn = load_matching_fn("temporal_ot", matching_fn_cfg)
    else:
        reward_fn = load_matching_fn("ot", matching_fn_cfg)
elif fig_name == "pretrain_fail_tot":
    matching_fn_cfg["mask_k"] = 10
    reward_fn = load_matching_fn("temporal_ot", matching_fn_cfg)
elif "_tot" in fig_name:
    reward_fn = load_matching_fn("temporal_ot", matching_fn_cfg)
elif "_orca" in fig_name:
    reward_fn = load_matching_fn("coverage", matching_fn_cfg)
elif "_dtw" in fig_name:
    reward_fn = load_matching_fn("dtw", matching_fn_cfg)

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

if args.fig_name == "main":
    # ORCA final learner trajectory for the run /share/portal/hw575/TemporalOT/train_logs/2025-01-23-20-29-39-358623_envt=door-open-v2_rm=coverage_bf1bfe
    learner_gif_path = "/share/portal/hw575/TemporalOT/wandb/run-20250123_203038-8wu58n4q/files/media/videos/trajectory/video_138_16f793f6235ded611ce3.gif"
elif args.fig_name == "main_stick-push_orca":
    learner_gif_path = "/share/portal/hw575/TemporalOT/wandb/run-20250124_231125-ts76umfw/files/media/videos/trajectory/video_64_aa711b056a144f7edae4.gif"
elif args.fig_name == "film_strip":
    # TOT final learner trajectory for the run https://wandb.ai/yuki-wang/temporal_ot/runs/2go5rvcy?nw=nwuserhw575
    learner_gif_path = "/share/portal/wph52/TemporalOT/wandb/run-20250124_214343-2go5rvcy/files/media/videos/trajectory/video_138_ca78d7cba45c25faba4e.gif"
elif args.fig_name == "order_fail_ot":
    learner_gif_path = "train_logs/2025-01-16-03-57-41-235100_envt=lever-pull-v2_rm=ot_6539a1/wandb/run-20250116_035742-hgarvy14/files/media/videos/trajectory/video_123_43567b1bd8456ebdf497.gif"
elif args.fig_name == "order_fail_tot10":
    learner_gif_path = "/share/portal/hw575/TemporalOT/train_logs/2025-01-15-02-11-12-637594_envt=lever-pull-v2_rm=temporal_ot_d6aff7/wandb/run-20250115_021113-u67eqn6w/files/media/videos/trajectory/video_123_46d7bf575adb05cdd45b.gif"
elif args.fig_name == "order_fail_tot2":
    learner_gif_path = "/share/portal/wph52/TemporalOT/wandb/run-20250124_214446-8u125aml/files/media/videos/trajectory/video_95_98fe100b0b0e053633c1.gif"
elif args.fig_name == "dtw_fail_orca":
    learner_gif_path = "/share/portal/hw575/TemporalOT/wandb/run-20250125_130753-bsvk4iy5/files/media/videos/trajectory/video_64_874239f0459681af62d2.gif"
elif args.fig_name == "dtw_fail_dtw":
    learner_gif_path = "/share/portal/hw575/TemporalOT/wandb/run-20250118_022208-roxhcg6l/files/media/videos/trajectory/video_138_20c0d289df8edd9d4d4d.gif"
elif args.fig_name == "pretrain_fail_orcanp":
    learner_gif_path = "/share/portal/hw575/TemporalOT/wandb/run-20250117_025931-klgz5615/files/media/videos/trajectory/video_138_e124a9614ffc006ef1b4.gif"
elif args.fig_name == "pretrain_fail_tot":
    learner_gif_path = "/share/portal/wph52/TemporalOT/train_logs/2025-01-15-17-59-07-706989_envt=stick-push-v2_rm=temporal_ot_661368/wandb/run-20250115_175908-t44g915x/files/media/videos/trajectory/video_85_673fe96c501a2c785908.gif"
elif args.fig_name == "pretrain_fail_orca":
    learner_gif_path = "/share/portal/hw575/TemporalOT/wandb/run-20250117_205206-amo0muzw/files/media/videos/trajectory/video_105_9d5b3a7dcaa74baa52ba.gif"

learner_gif = load_gif_frames(learner_gif_path, "torch")

print(type(learner_gif))
print(learner_gif.size())

rewards, info = agent.rewarder(learner_gif.to(device))
# print(f"Info assignment: {info['assignment']}")
# print(f"Info cost: {info['cost_matrix']}")

assignment = info['assignment']
diffs = np.zeros(assignment.shape)

for i in range(1, assignment.shape[0]):
    # print(f"i-1: {i-1}, {assignment[i-1]}")
    # print(f"i: {i}, {assignment[i]}")
    # print(f"diff: {assignment[i] - assignment[i-1]}")
    # input("stop")
    diffs[i] = (assignment[i] - assignment[i-1]) == 0

def plot_heatmap(matrix, title, cmap):
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap=cmap, aspect="auto")
    cbar = plt.colorbar(orientation='horizontal' if "main" in args.fig_name else 'vertical')
    cbar.ax.tick_params(labelsize=16)
    plt.title(title)
    # plt.xlabel("Reference Trajectory")
    # plt.ylabel("Learner Trajectory")
    # increase the tick font
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Save the plot to a BytesIO buffer
    plt.savefig(f"main_fig/{args.fig_name}/{title}.png", format="png")
    plt.close()

plot_heatmap(info['cost_matrix'], "cost_matrix", "gray_r")

color_map = {
    "main": "Greens",
    "main_stick-push_orca": "Greens",
    "film_strip": "Blues",
    "order_fail_ot": "Purples",
    "order_fail_tot10": "Blues",
    "order_fail_tot2": "Blues",
    "dtw_fail_orca": "Greens",
    "dtw_fail_dtw": "Reds",
    "pretrain_fail_orcanp": "Greens",
    "pretrain_fail_tot": "Blues",
    "pretrain_fail_orca": "Greens"
}
plot_heatmap(assignment, "assignment", color_map[args.fig_name])

# Copy the last column of the assignment matrix 10 times then plot them
# last_column = assignment[:, -1]
# last_column = np.expand_dims(last_column, axis=1)
# last_column = np.repeat(last_column, 10, axis=1)
# plot_heatmap(last_column, "assignment_last_column", "Greens")
# # plot_heatmap(diffs, "diffs", "Greens")
