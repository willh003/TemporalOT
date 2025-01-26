from gymnasium import Env, spaces
import numpy as np
from stable_baselines3 import PPO, SAC
import torch as th
from s3dg import S3D
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList
import os
import cv2
from PIL import Image, ImageSequence
import metaworld
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_HIDDEN, ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from gymnasium.spaces import Box

from callbacks import WandbCallback, MetaWorldVideoRecorderCallback
from make_vec_env import make_vec_env
from subproc_vec_env import SubprocVecEnv as CQSubprocVecEnv
import utils_local

import wandb
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.global_hydra import GlobalHydra
from loguru import logger

from temporal_env_utils import make_env
from utils_local import set_os_vars
from env_utils import make_env, RGBArrayAsObservationWrapper, ActionDTypeWrapper, ActionRepeatWrapper, FrameStackWrapper, ExtendedTimeStepWrapper
import random
from meta_vars import CAMERA


RB_PATH = "/home/aw588/git_annshin/roboclip_local/"

MAP_GIF_NAME = {
    "button-press-v2": "button-press-v2_corner_0",
    "button-press-v2-goal-hidden": "button-human",
    "door-close-v2": "door-close-v2_corner_0",
    "door-open-v2": "door-open-v2_corner3_0",
    "window-open-v2": "window-open-v2_corner3_0",
    "lever-pull-v2": "lever-pull-v2_corner3_0",
    "hand-insert-v2": "hand-insert-v2_corner3_0",
    "push-v2": "push-v2_corner3_0",
    "basketball-v2": "basketball-v2_corner_0",
    "stick-push-v2": "stick-push-v2_corner_0",
    "door-lock-v2": "door-lock-v2_corner_0",
}


class MetaWorldSparseEnv(Env):
    def __init__(
        self,
        env_id: str,
        episode_length: int = 128,
        render_mode: str = "rgb_array",
        video_path: str = None,
        normalize_similarity: bool = False,
        rank: int = 0,
        seed: int = 0,
        **kwargs,
    ):
        # Initialize MetaWorld environment
        # env_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[env_id]
        self.env_id = env_id
        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[f"{env_id}-goal-observable"]
        self.seed = seed
        self.env = env_cls(seed=self.seed)
        self.env._render_camera_name = CAMERA[self.env_id]
        self.env._freeze_rand_vec = False
        self.render_mode = render_mode
        
        # Set up spaces
        # self.observation_space = self.env.observation_space
        # self.action_space = self.env.action_space
        # Convert gym spaces to gymnasium spaces
        self.observation_space = Box(
            low=self.env.observation_space.low,
            high=self.env.observation_space.high,
            dtype=self.env.observation_space.dtype
        )
        self.action_space = Box(
            low=self.env.action_space.low,
            high=self.env.action_space.high,
            dtype=self.env.action_space.dtype
        )
        
        self.episode_length = episode_length
        self.num_steps = 0
        self.normalize_similarity = normalize_similarity

        # Initialize video processing components
        self.past_observations = []
        self.window_length = 16
        self.net = S3D(f'{RB_PATH}/s3d_dict.npy', 512)
        self.net.load_state_dict(th.load(f'{RB_PATH}/s3d_howto100m.pth'))
        self.net = self.net.eval()

        # Set up device
        if th.cuda.is_available():
            self.device = th.device("cuda")
            self.device_type = "cuda"
        else:
            self.device = th.device("cpu")
            self.device_type = "cpu"
        
        self.net = self.net.to(self.device)

        # Load reference video if provided
        if video_path:
            frames = self.read_video(video_path)
            frames = self.preprocess_video(frames)
            video = th.from_numpy(frames)
            if self.device_type == "cuda":
                video = video.float().to(self.device)
            else:
                video = video.float()
            video_output = self.net(video)
            self.target_embedding = video_output['video_embedding']
            self.target_embedding = self.target_embedding.to(self.device)
        else:
            raise ValueError("video_path must be provided for video-based reward")
        
        # # add wrappers
        # note: commented, because we just need goal_achieved which is info["success"]
        # self.env = RGBArrayAsObservationWrapper(self.env,
        #                                 max_path_length=None,
        #                                 camera_name="d",
        #                                 include_timestep=False)
        # self.env = ActionDTypeWrapper(self.env, np.float32)
        # self.env = ActionRepeatWrapper(self.env, 2)
        # self.env = FrameStackWrapper(self.env, 1)
        # self.env = ExtendedTimeStepWrapper(self.env)

        # set random seed
        self.env.seed(self.seed)
        self.env.action_space.seed(seed=self.seed)

    def read_video(self, video_path):
        gif = Image.open(video_path)
        frames = []
        
        for frame in ImageSequence.Iterator(gif):
            rgb_frame = frame.convert("RGB")
            resized_frame = rgb_frame.resize((250, 250), Image.LANCZOS)
            np_frame = np.array(resized_frame)
            bgr_frame = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)
            frames.append(bgr_frame)
        
        return frames
    
    def preprocess_video(self, frames, shorten=True):
        center = 240, 320
        h, w = (250, 250)
        x = int(center[1] - w/2)
        y = int(center[0] - h/2)
        frames = np.array([frame[y:y+h, x:x+w] for frame in frames])
        frames = frames[None, :,:,:,:]
        frames = frames.transpose(0, 4, 1, 2, 3)
        if shorten:
            frames = frames[:, :,::4,:,:]
        return frames

    def render(self):
        return self.env.render(camera_name=self.env._render_camera_name)

    def step(self, action):
        obs, _, terminated, info = self.env.step(action)
        self.past_observations.append(self.render())
        self.num_steps += 1

        if terminated or self.num_steps >= self.episode_length:
            frames = self.preprocess_video(self.past_observations)
            video = th.from_numpy(frames)
            if self.device_type == "cuda":
                video = video.float().to(self.device)
            else:
                video = video.float()
            video_output = self.net(video)
            video_embedding = video_output['video_embedding']
            similarity_matrix = th.matmul(self.target_embedding, video_embedding.t())
            if self.normalize_similarity:
                similarity_matrix = similarity_matrix / th.norm(similarity_matrix, p='fro')
            if self.device_type == "cuda":
                reward = similarity_matrix.cpu().detach().numpy()[0][0]
            else:
                reward = similarity_matrix.detach().numpy()[0][0]
        else:
            reward = 0.0

        terminated = self.num_steps >= self.episode_length
        truncated = False
        
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.past_observations = []
        self.num_steps = 0
        obs = self.env.reset()
        return obs, {}

def make_env_fn(cfg, **kwargs):
    """Create environment factory function"""
    print(f"video_path: {f'{RB_PATH}/metaworld_demos/{cfg.env.task_name}/{cfg.demo_folder}/{MAP_GIF_NAME[cfg.env.task_name]}.gif'}")
    seed = random.randint(0, 1000000)
    print(f"seed: {seed}")
    def _init():
        env = MetaWorldSparseEnv(
            env_id=cfg.env.task_name,
            episode_length=cfg.env.episode_length,
            video_path=f"{RB_PATH}/metaworld_demos/{cfg.env.task_name}/{cfg.demo_folder}/{MAP_GIF_NAME[cfg.env.task_name]}.gif",
            seed=seed,
            **kwargs
        )
        return env
    return _init

@hydra.main(config_path="configs", config_name="train_config")
def main(cfg: DictConfig):
    set_os_vars()

    # Setup paths and logging
    cfg.logging.run_path = os.getcwd()
    logger.info(f"Started run with run_name={cfg.logging.run_path}")
    
    # Create directories
    os.makedirs(os.path.join(cfg.logging.run_path, "eval"), exist_ok=True)
    wandb_dir = f"{cfg.logging.run_path}/wandb"
    os.makedirs(wandb_dir, exist_ok=True)

    # Create vectorized environment
    make_env_kwargs = utils_local.get_make_env_kwargs(cfg)
    envs = make_vec_env(
        make_env_fn(cfg),
        n_envs=cfg.compute.n_cpu_workers,
        seed=cfg.seed,
        vec_env_cls=CQSubprocVecEnv,
        vec_env_kwargs=dict(render_dim=(cfg.env.render_dim[0], cfg.env.render_dim[1], 3)),
    )
    eval_env = make_vec_env(
        make_env_fn(cfg),
        n_envs=1,
        seed=cfg.seed,
        vec_env_cls=CQSubprocVecEnv,
        vec_env_kwargs=dict(render_dim=(cfg.env.render_dim[0], cfg.env.render_dim[1], 3)),
    )

    # Initialize model
    if cfg.algo == "ppo":
        model = PPO("MlpPolicy", envs, verbose=1, tensorboard_log=cfg.logging.run_path, 
                   n_steps=cfg.n_steps, batch_size=cfg.n_steps*cfg.compute.n_cpu_workers)
    elif cfg.algo == "sac":
        model = SAC("MlpPolicy", envs, verbose=1, tensorboard_log=cfg.logging.run_path)

    # Setup wandb and callbacks
    with wandb.init(
        entity="yuki-wang",
        project=cfg.logging.wandb_project,
        name=cfg.logging.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=wandb_dir,
        mode=cfg.logging.wandb_mode,
        tags=cfg.logging.wandb_tags,
        sync_tensorboard=True,
        monitor_gym=True,
    ) as wandb_run:
        checkpoint_dir = os.path.join(cfg.logging.run_path, "checkpoint")
        
        callbacks = [
            WandbCallback(
                model_save_path=checkpoint_dir,
                model_save_freq=cfg.logging.model_save_freq // cfg.compute.n_cpu_workers,
            ),
            MetaWorldVideoRecorderCallback(
                eval_env=eval_env,
                rollout_save_path=os.path.join(cfg.logging.run_path, "eval"),
                render_freq=cfg.logging.video_save_freq // cfg.compute.n_cpu_workers, # 2000
                n_eval_episodes=1,
                deterministic=True,
                verbose=1,
                env_id=cfg.env.task_name
            )
        ]

        # Train model
        model.learn(
            total_timesteps=cfg.total_timesteps,
            callback=CallbackList(callbacks),
        )

        # Save final model
        model.save(os.path.join(checkpoint_dir, "final_model"))
        logger.info("Training complete")
        wandb_run.finish()

if __name__ == "__main__":
    GlobalHydra.instance().clear()
    main()