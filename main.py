import logging
import os
import pickle
import time
from pathlib import Path

import imageio
import numpy as np
import pandas as pd
import torch
from dm_env import specs
from tqdm import tqdm

from models import ResNet, TemporalOTAgent
from utils import (eval_agent, eval_mode, get_image, get_logger, make_env,
                   make_replay_loader, make_expert_replay_loader,
                   record_demo, ReplayBufferStorage, load_gif_frames)

from utils.env_utils import CAMERA
from utils.cluster_utils import set_os_vars
from collect_expert_traj import collect_trajectories

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="basketball-v3", type=str)
    parser.add_argument("--obs_type", default="features", type=str)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--gamma", default=0.9, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_demos", default=2, type=int)
    parser.add_argument("--expl_noise", default=0.4, type=float)
    parser.add_argument("--min_expl_noise", default=0.0, type=float)
    parser.add_argument("--camera_name", default="d", type=str)

    # context embedding
    parser.add_argument("--context_num", default=3, type=int)

    # temporal mask
    parser.add_argument("--mask_k", default=10, type=int)
    parser.add_argument("--epsilon", default=0.01, type=float)
    parser.add_argument("--niter", default=100, type=int)

    # encoder
    parser.add_argument("--encoder", default="resnet", type=str)

    args = parser.parse_args()
    return args


def run(cfg):
    # random seed
    seed = cfg.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # setup logger
    env_name = cfg.env_name
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    work_dir = Path.cwd()
    t1 = time.time() 
    exp_prefix = f"cn{cfg.context_num}_mk{cfg.mask_k}_en{cfg.expl_noise}"
    exp_name = f"s{seed}_{timestamp}"
    print(f"Running {env_name}_{exp_prefix}_{exp_name}")
    log_dir = f"logs/{exp_prefix}/{env_name}"
    model_dir = f"saved_models/{exp_prefix}/{env_name}/{exp_name}"
    video_dir = f"saved_videos/{exp_prefix}/{env_name}/{exp_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    logger = get_logger(f"{log_dir}/{exp_name}.log")
    logger.info("Configs:\n" + "".join([
        f"\t{k}: {v}\n" for k, v in vars(args).items()]))

    # use pixel or state
    if cfg.obs_type == "pixels":
        frame_stack = 3
        use_encoder = True
    else:
        frame_stack = 1
        use_encoder = False

    # initialize environments
    train_env, env_horizon = make_env(name=env_name,
                                      frame_stack=frame_stack,
                                      action_repeat=2,
                                      seed=seed)
    eval_env, _ = make_env(name=env_name,
                           frame_stack=frame_stack,
                           action_repeat=2,
                           seed=seed+123)
    env_horizon = (env_horizon+1) // 2
    data_specs = [
        train_env.observation_spec()[cfg.obs_type],  # pixel: (9, 84, 84)
        train_env.action_spec(),
        specs.Array((1,), np.float32, "reward"),
        specs.Array((1,), np.float32, "discount")
    ]

    # initialize the replay buffer
    buffer_dir = work_dir / "data" / f"buffer_{env_name}_{timestamp}"
    replay_storage = ReplayBufferStorage(data_specs, buffer_dir)
    replay_loader = make_replay_loader(replay_dir=buffer_dir,
                                       max_size=150000,
                                       batch_size=512,
                                       num_workers=cfg.num_workers,
                                       save_experiences=False,
                                       nstep=3,
                                       discount=cfg.gamma)

    # replay buffer iterator
    replay_iter = None

    # initialize the agent
    obs_spec=train_env.observation_spec()
    action_spec=train_env.action_spec()
    agent = TemporalOTAgent(obs_shape=obs_spec[cfg.obs_type].shape,
                            action_shape=action_spec.shape,
                            device=device,
                            lr=1e-4,
                            env_horizon=env_horizon,
                            feature_dim=50,
                            hidden_dim=1024,
                            critic_target_tau=0.005,
                            stddev=0.1,
                            stddev_clip=0.3,
                            sinkhorn_rew_scale=1,
                            auto_rew_scale_factor=10,
                            context_num=cfg.context_num,
                            mask_k=cfg.mask_k,
                            epsilon=cfg.epsilon,
                            use_encoder=use_encoder)

    # expert demo
    # "d" is a placeholder for the default camera
    if cfg.camera_name != "d":
        camera_name = cfg.camera_name
    else:
        camera_name = CAMERA[env_name]
        
    expert_pixel = []
    for i in range(cfg.num_demos):
        # If we have not collected expert trajectories yet
        if not os.path.exists(f"create_demo/metaworld_demos/{env_name}/{env_name}_{camera_name}_{i}.gif"):
            collect_trajectories(env_name, cfg.num_demos, camera_name)

        data = load_gif_frames(f"create_demo/metaworld_demos/{env_name}/{env_name}_{camera_name}_{i}.gif", "torch")
        expert_pixel.append(data)

    # Resnet50: (88, 3, 224, 224) ==> (88, 2048, 7, 7) ==> (88, 100352) 
    cost_encoder = ResNet().to(device)
    _ = cost_encoder.eval()
    with torch.no_grad():
        demos = [cost_encoder(demo.to(device)) for demo in expert_pixel]

    agent.init_demos(cost_encoder, demos)
    logger.info(f"len(demo) = {len(demos)}, demos[0].shape = {demos[0].shape}")

    # start training
    global_episode, episode_step, episode_reward = 0, 0, 0
    time_steps, pixels = [], []
    time_step = train_env.reset()
    time_steps.append(time_step)
    success = 0
    cum_success = 0
    record_traj = False

    # run 1e6 steps with action repeat = 2
    t = 1
    total_timesteps = 500000
    pbar = tqdm(total=total_timesteps)
    res = [(0, cfg.gamma**3, 0)]
    while t <= total_timesteps:
        # end of a trajectory
        if time_step.last():
            if record_traj:
                video_fname = f"{video_dir}/{global_episode}.mp4"
                imageio.mimsave(video_fname, frames, fps=60)
            global_episode += 1
            record_traj = global_episode % 400 == 0
            pixels = np.stack(pixels, axis=0)

            ot_rewards, cost_min, cost_max = agent.ot_rewarder(pixels)
            assert cost_min >= 0

            # use first episode to normalize rewards
            if global_episode == 1:
                ot_rewards_sum = abs(ot_rewards.sum())
                agent.sinkhorn_rew_scale = agent.auto_rew_scale_factor / ot_rewards_sum
                logger.info(f"agent.sinkhorn_rew_scale = {agent.sinkhorn_rew_scale:.3f}")
                ot_rewards, cost_min, cost_max = agent.ot_rewarder(pixels)

            # add to buffer at the end of the trajectory
            for i, elt in enumerate(time_steps):
                elt = elt._replace(observation=time_steps[i].observation[cfg.obs_type])
                if i == 0:
                    elt = elt._replace(reward=float("nan"))
                else:
                    elt = elt._replace(reward=ot_rewards[i - 1])
                replay_storage.add(elt)

            # reset env
            time_steps = []
            pixels = []

            time_step = train_env.reset()
            if record_traj:
                frames = [get_image(time_step)]

            time_steps.append(time_step)
            episode_step = 0
            episode_reward = 0

        # sample action
        if t <= 2000:
            action = train_env.action_space.sample()
        else:
            with torch.no_grad(), eval_mode(agent):
                expl_noise = cfg.expl_noise - t * (
                    (cfg.expl_noise - cfg.min_expl_noise) / total_timesteps)
                action = agent.act(time_step.observation[cfg.obs_type],
                                   expl_noise=expl_noise,
                                   eval_mode=False)

        # update the agent
        if t > 6000:
            if replay_iter is None:
                replay_iter = iter(replay_loader)
            gamma = agent.update(replay_iter)

        # take env step
        time_step = train_env.step(action)
        success = time_step.observation["goal_achieved"]
        cum_success += success
        episode_reward += time_step.reward
        time_steps.append(time_step)
        pixels.append(time_step.observation["pixels_large"])
        if record_traj:
            frames.append(get_image(time_step))
        episode_step += 1

        # evaluation
        if t % 10000 == 0:
            eval_success_rate = eval_agent(agent, eval_env, cfg.obs_type)
            res.append((t, gamma.item(), expl_noise, eval_success_rate))
            logger.info(
                f"[T {t//1000}K][EP {global_episode}] "
                f"time: {(time.time() - t1)/60:.1f}, "
                f"sr: {eval_success_rate:.1f}\n"
                f"\tR: {ot_rewards.mean():.3f}, "
                f"Rmax: {ot_rewards.max():.3f}, "
                f"Rmin: {ot_rewards.min():.3f}, "
                f"expl: {expl_noise:.3f}\n"
                f"\tgamma: {gamma.item():.2f}, "
                f"done: {success:.0f}, "
                f"cum_done: {cum_success:.0f}, "
                f"cmin: {cost_min:.2f}, "
                f"cmax: {cost_max:.2f}\n"
            )

        # update step
        t += 1
        pbar.update(1)

    # save logging
    df = pd.DataFrame(res, columns=["step", "gamma", "expl_noise", "success_rate"])
    df.to_csv(f"{log_dir}/{exp_name}.csv")

    # delete buffer
    os.system(f"rm -rf ./data/buffer_{env_name}_{timestamp}")


def run_wandb():
    with wandb.init(
        project=cfg.logging.wandb_project,
        name=cfg.logging.run_name,
        tags=tags,
        sync_tensorboard=True,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        mode=cfg.logging.wandb_mode,
        monitor_gym=True,  # auto-upload the videos of agents playing the game
    ) as wandb_run:

if __name__ == "__main__":
    args = get_args()
    set_os_vars()
    run(args)
