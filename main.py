import logging
import os
import time
from pathlib import Path

import imageio
import numpy as np
import pandas as pd
import torch
from dm_env import specs
from tqdm import tqdm

from models import ResNet, DDPGAgent
from utils import (eval_agent, eval_mode, get_image, get_logger, make_env,
                   make_replay_loader, make_expert_replay_loader,
                   record_demo, ReplayBufferStorage, load_gif_frames, get_output_folder_name, get_output_path)

from seq_matching import load_matching_fn

from demo import CAMERA, get_demo_gif_path
from utils.cluster_utils import set_os_vars

from datetime import datetime
import wandb
import uuid
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.global_hydra import GlobalHydra

def run(cfg, wandb_run=None):
    

    # random seed
    if cfg.seed == 'r':
        seed = np.random.rand() * 1000
    else:
        seed = cfg.seed

    np.random.seed(seed)
    torch.manual_seed(seed)

    # setup logger
    env_name = cfg.env_name
    device = cfg.device

    run_path = get_output_path()

    buffer_dir = os.path.join(run_path, "buffer") # used for storing replay buffer
    log_dir = os.path.join(run_path, "logs")
    eval_dir = os.path.join(run_path, "eval")
    model_dir = os.path.join(run_path, "models")
    video_dir = os.path.join(run_path, "videos")

    os.makedirs(buffer_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    logger = get_logger(f"{log_dir}/log.log")

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
                           seed=seed)
    env_horizon = (env_horizon+1) // 2
    data_specs = [
        train_env.observation_spec()[cfg.obs_type],  # pixel: (9, 84, 84)
        train_env.action_spec(),
        specs.Array((1,), np.float32, "reward"),
        specs.Array((1,), np.float32, "discount")
    ]

    # initialize the replay buffer
    replay_storage = ReplayBufferStorage(data_specs, Path(buffer_dir))
    replay_loader = make_replay_loader(replay_dir=Path(buffer_dir),
                                       max_size=150000,
                                       batch_size=512,
                                       num_workers=cfg.num_workers,
                                       save_experiences=False,
                                       nstep=3,
                                       discount=cfg.discount_factor)

    # replay buffer iterator
    replay_iter = None

    # get the custom reward function
    matching_fn_cfg = {
        "tau": cfg.tau,
        "ent_reg": cfg.ent_reg,
        "mask_k": cfg.mask_k,
        "sdtw_smoothing": cfg.sdtw_smoothing
    }
    
    reward_fn = load_matching_fn(cfg.reward_fn, matching_fn_cfg)

    # initialize the agent
    obs_spec=train_env.observation_spec()
    action_spec=train_env.action_spec() 
    agent = DDPGAgent(reward_fn=reward_fn,
                        obs_shape=obs_spec[cfg.obs_type].shape,
                        action_shape=action_spec.shape,
                        device=device,
                        lr=1e-4,
                        env_horizon=env_horizon,
                        feature_dim=50,
                        hidden_dim=1024,
                        critic_target_tau=0.005,
                        stddev=0.1,
                        stddev_clip=0.3,
                        rew_scale=1, # this will be updated after first train iter
                        auto_rew_scale_factor=10,
                        context_num=cfg.context_num,
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
        demo_path = get_demo_gif_path("metaworld", env_name, camera_name, i, num_frames="d")

        if not os.path.exists(demo_path):
            raise Exception(f"No trajectory for {env_name}_{camera_name}_{i}. You need to create the trajectories first")

        data = load_gif_frames(demo_path, "torch")
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
    res = [(0, cfg.discount_factor**3, 0)]
    while t <= total_timesteps:
        # end of a trajectory
        if time_step.last():
            if record_traj:
                if wandb_run is not None:
                    wandb_run.log({"trajectory/video":
                        wandb.Video(np.stack([np.uint8(f).transpose(2, 0, 1) for f in frames]), fps=15)}
                    )
                else:
                    video_fname = f"{video_dir}/{global_episode}.mp4"
                    imageio.mimsave(video_fname, frames, fps=15)

            global_episode += 1
            record_traj = global_episode % cfg.video_period == 0 or global_episode == 1 # record the first timestep and every video_record_period
            pixels = np.stack(pixels, axis=0)

            rewards, cost_min, cost_max = agent.rewarder(pixels)
            assert cost_min >= 0

            # use first episode to normalize rewards
            if global_episode == 1:
                rewards_sum = abs(rewards.sum())
                agent.set_reward_scale(1 / (rewards_sum+1e-5))
                logger.info(f"agent.sinkhorn_rew_scale = {agent.get_reward_scale():.3f}")
                rewards, cost_min, cost_max = agent.rewarder(pixels)

            # add to buffer at the end of the trajectory
            for i, elt in enumerate(time_steps):
                elt = elt._replace(observation=time_steps[i].observation[cfg.obs_type])
                if i == 0:
                    elt = elt._replace(reward=float("nan"))
                else:
                    elt = elt._replace(reward=rewards[i - 1])
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

        # for the first 2000 steps, just randomly sample actions. Then start querying the agent.
        if t <= 2000:
            action = train_env.action_space.sample()
        else:
            with torch.no_grad(), eval_mode(agent):
                expl_noise = cfg.expl_noise - t * (
                    (cfg.expl_noise - cfg.min_expl_noise) / total_timesteps)
                action = agent.act(time_step.observation[cfg.obs_type],
                                   expl_noise=expl_noise,
                                   eval_mode=False)
        
        # after 6000 steps, start updating agent
        if t > 6000:
            if replay_iter is None:
                replay_iter = iter(replay_loader)
            discount_factor = agent.update(replay_iter)

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
        if t % cfg.eval_period == 0:
            eval_metrics, final_observations = eval_agent(agent, eval_env, cfg.obs_type, cfg.n_eval_episodes)
            
            # save the observations
            np.save(os.path.join(eval_dir,f"{t}.npy"), final_observations)

            metrics = {"train/step": t, 
                    "train/global_episode": global_episode,
                    "train/expl_noise": expl_noise, 
                    "train/discount_factor": discount_factor.item(),
                    "rewards/mean_reward": rewards.mean(),
                    "rewards/min_reward": rewards.min(),
                    "rewards/max_reward": rewards.max(),
                    "rewards/min_distance": cost_min,
                    "rewards/max_distance": cost_max, 
                    "rewards/reward_scale": -agent.get_reward_scale(),
                    **eval_metrics
            }

            res.append((t, discount_factor.item(), expl_noise, *(v for v in eval_metrics.values())))

            if wandb_run is not None:        
                wandb_run.log(metrics)
            else:
                logger.info(
                    str(metrics)
                )
        if t % cfg.model_period == 0:
            state_dict = agent.save_snapshot()
            torch.save(state_dict, os.path.join(model_dir, f'{t}.pt'))

        # update step
        t += 1
        pbar.update(1)

    # save logging
    df = pd.DataFrame(res, columns=["step", "gamma", "expl_noise", *(k for k in eval_metrics.keys())])
    df.to_csv(f"{log_dir}/performance.csv")

    # delete buffer
    os.system(f"rm -rf {buffer_dir}")


def run_wandb(cfg):
    run_name = get_output_folder_name()
    tags = [cfg.env_name, cfg.reward_fn]

    with wandb.init(
        project="temporal_ot",
        name=run_name,
        tags=tags,
        sync_tensorboard=True,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        mode=cfg.wandb_mode,
        monitor_gym=True,  # auto-upload the videos of agents playing the game
    ) as wandb_run:
    
        run(cfg, wandb_run)


@hydra.main(config_path="configs", config_name="train_config")
def main(cfg: DictConfig):

    if cfg.wandb_mode == "disabled":
        run(cfg)
    else:
        run_wandb(cfg)

if __name__=="__main__":
    set_os_vars()
    GlobalHydra.instance().clear()

    # Generate short uuid to ensure no collisions on run paths
    # This is important because buffers are stored in the run, so collisions will cause weird errors
    OmegaConf.register_new_resolver("uuid", lambda: str(uuid.uuid4())[:6])
    
    main()
