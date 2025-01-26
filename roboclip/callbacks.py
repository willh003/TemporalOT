import os
import time
import json
from typing import Any, Dict, Optional
import imageio
import gymnasium
import torch as th
import numpy as np
from numpy import array
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import (
    CheckpointCallback as SB3CheckpointCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video
from stable_baselines3.common.logger import Image as LogImage  # To avoid conflict with PIL.Image
from wandb.integration.sb3 import WandbCallback as SB3WandbCallback
from stable_baselines3.common.base_class import BaseAlgorithm

from PIL import Image, ImageDraw, ImageFont
from numbers import Number

from loguru import logger
from einops import rearrange

from seq_reward.seq_utils import get_matching_fn, load_reference_seq, load_images_from_reference_seq, seq_matching_viz
from seq_reward.cost_fns import euclidean_distance_advanced, euclidean_distance_advanced_arms_only

from constants import TASK_SEQ_DICT
from utils_local import calc_iqm
from math_utils import interquartile_mean_and_ci, mean_and_std

from meta_vars import CAMERA
    

def plot_info_on_frame(pil_image, info, font_size=20):
    """
    Parameters:
        pil_image: PIL.Image
            The image to plot the info on
        info: Dict
            The information to plot on the image
        font_size: int
            The size of the font to use for the text
    
    Effects:
        pil_image is modified to include the info
    """
    # TODO: this is a hard-coded path
    font = ImageFont.truetype("/share/portal/hw575/vlmrm/src/vlmrm/cli/arial.ttf", font_size)
    draw = ImageDraw.Draw(pil_image)

    x = font_size  # X position of the text
    y = pil_image.height - font_size  # Beginning of the y position of the text
    
    i = 0
    for k in info:
        # TODO: This is pretty ugly
        if not any([text in k for text in ["TimeLimit", "render_array", "geom_xpos"]]):
            reward_text = f"{k}:{info[k]}"
            # Plot the text from bottom to top
            text_position = (x, y - (font_size + 10)*(i+1))
            draw.text(text_position, reward_text, fill=(255, 255, 255), font=font)
        i += 1


class VideoRecorderCallback(BaseCallback):
    def __init__(
        self,
        eval_env: gymnasium.Env,
        rollout_save_path: str,
        render_freq: int,
        render_dim: tuple = (480, 480, 3),
        n_eval_episodes: int = 1,
        deterministic: bool = True,
        use_geom_xpos: bool = True,
        task_name: str = "",
        threshold: float = 0.5,
        success_fn_cfg: dict = {},
        matching_fn_cfg: dict = {}, 
        calc_visual_reward: bool = False,
        verbose=0
    ):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to
        TensorBoard

        Pararmeters
            eval_env: A gym environment from which the trajectory is recorded
                Assumes that there's only 1 environment
            rollout_save_path: The path to save the rollouts (states and rewards)
            render_freq: Render the agent's trajectory every eval_freq call of the callback.
            n_eval_episodes: Number of episodes to render
            deterministic: Whether to use deterministic or stochastic policy
            goal_seq_name: The name of the reference sequence to compare with (This defines the unifying metric that all approaches attempting to solve the same task gets compared against)
            seq_name: The name of the reference sequence to compare with
                You only need to set this if you want to calculate the OT reward
            matching_fn_cfg: The configuration for the matching function
        """
        super().__init__(verbose)
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._render_dim = render_dim
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic
        self._rollout_save_path = rollout_save_path  # Save the state of the environment
        self._use_geom_xpos = use_geom_xpos
        self._threshold = threshold
        self._calc_visual_reward = calc_visual_reward

        if task_name != "":
            self._goal_ref_seq = load_reference_seq(task_name=task_name, seq_name="key_frames", use_geom_xpos=self._use_geom_xpos)
            logger.info(f"[VideoRecorderCallback] Loaded reference sequence. task_name={task_name}, seq_name=key_frames, use_geom_xpos={self._use_geom_xpos}, shape={self._goal_ref_seq.shape}")

            self.set_ground_truth_goal_matching_fn(task_name, use_geom_xpos)
            self.set_success_fn(success_fn_cfg)

            # self._calc_gt_reward = True

            self._success_json_save_path = os.path.join(self._rollout_save_path, "success_results.json")
            self._success_results = {}
        # else:
        #     self._calc_gt_reward = False
        self._calc_gt_reward = True

        if matching_fn_cfg != {}:
            # The reference sequence that is used to calculate the sequence matching reward
            self._seq_matching_ref_seq = load_reference_seq(task_name=task_name, seq_name=matching_fn_cfg["seq_name"], use_geom_xpos=self._use_geom_xpos)
            # For the frames, we remove the initial frame which matches the initial position
            self._seq_matching_ref_seq_frames = load_images_from_reference_seq(task_name=task_name, seq_name=matching_fn_cfg["seq_name"])[1:]
            # TODO: For now, we can only visualize this when the reference frame is defined via a gif
            self._plot_matching_visualization = len(self._seq_matching_ref_seq_frames) > 0

            self._calc_matching_reward = True
            self._scale = matching_fn_cfg['scale']
            self._matching_fn, self._matching_fn_name = get_matching_fn(matching_fn_cfg, matching_fn_cfg["cost_fn"])

            self._reward_vmin = matching_fn_cfg.get("reward_vmin", -1)
            self._reward_vmax = matching_fn_cfg.get("reward_vmax", 0)

            logger.info(f"[VideoRecorderCallback] Loaded reference sequence for seq level matching. task_name={task_name}, seq_name={matching_fn_cfg['seq_name']}, use_geom_xpos={self._use_geom_xpos}, shape={self._seq_matching_ref_seq.shape}, image_frames_shape={self._seq_matching_ref_seq_frames.shape}")
        else:
            self._calc_matching_reward = False
        

    def set_ground_truth_goal_matching_fn(self, task_name: str, use_geom_xpos: bool):
        """Set the ground-truth goal matching function based on the goal_seq_name.

        This will be unifying metric that we measure the performance of different methods against.

        The function will return an reward array of size (n_timesteps,) where each element is the reward for the corresponding timestep.
        """
        is_goal_reaching_task = TASK_SEQ_DICT[task_name]["task_type"].lower() == "goal_reaching"

        if is_goal_reaching_task:
            logger.info(f"Goal Reaching Task. The ground-truth reward will be calculated based on the final joint state only. Task name = {task_name}")

            assert len(self._goal_ref_seq) == 1, f"Expected only 1 reference sequence, got {len(self._goal_ref_seq)}"
            
            axis_to_norm = (1,2) if use_geom_xpos else 1

            self._gt_goal_matching_fn = lambda rollout: np.exp(-np.linalg.norm(rollout - self._goal_ref_seq, axis=axis_to_norm))
        else:
            def seq_matching_fn(ref, rollout, threshold):
                """
                Calculate the reward based on the sequence matching to the goal_ref_seq

                Parameters:
                    rollout: np.array (rollout_length, ...)
                        The rollout sequence to calculate the reward
                """
                # Calculate reward from the rollout to self.goal_ref_seq
                reward_matrix = np.exp(-euclidean_distance_advanced(rollout, ref))

                # Detect when a stage is completed (the rollout is close to the goal_ref_seq) (under self._threshold)
                stage_completed = 0
                stage_completed_matrix = np.zeros(reward_matrix.shape) # 1 if the stage is completed, 0 otherwise
                current_stage_matrix = np.zeros(reward_matrix.shape) # 1 if the current stage, 0 otherwise
                
                for i in range(len(reward_matrix)):  # Iterate through the timestep
                    current_stage_matrix[i, stage_completed] = 1
                    if reward_matrix[i][stage_completed] > threshold and stage_completed < len(ref) - 1:
                        stage_completed += 1
                    stage_completed_matrix[i, :stage_completed] = 1

                # Find the highest reward to each reference sequence
                highest_reward = np.max(reward_matrix, axis=0)

                # Reward (shape: (rollout)) at each timestep is
                #   Stage completion reward + Reward at the current stage
                reward = np.sum(stage_completed_matrix * highest_reward + current_stage_matrix * reward_matrix, axis=1)/len(ref)

                return reward
            
            self._gt_goal_matching_fn = lambda rollout: seq_matching_fn(self._goal_ref_seq, rollout, self._threshold)


    def set_success_fn(self, success_fn_cfg):
        """
        Whether the entire body is above an threshold (0.5)
        Whether the arm is above an threshold (0.55)

        Binary success: whether at any point has the key poses have been hit
            # of the key poses that have been hit (in the right order)
        The percentage of time that it's holding the key pose
            For each key pose, we find the time interval that each key poses hold
        """
        def success_fn(obs_seq, ref_seq, threshold):
            """
            Calculate the binary success based on the rollout and the reference sequence

            Parameters:
                rollout: np.array (rollout_length, ...)
                    The rollout sequence to calculate the reward

            Return:
                pct_stage_completed: float
                    The percentage of stages that are completed
                pct_timesteps_completing_the_stages: float
                    The percentage of timesteps that are completing the stages
            """
            # Calculate reward from the rollout to self.goal_ref_seq
            reward_matrix = np.exp(-euclidean_distance_advanced(obs_seq, ref_seq))

            # Detect when a stage is completed (the rollout is close to the goal_ref_seq) (under self._threshold)
            current_stage = 0
            stage_completed = 0
            # Track the number of steps where a stage is being completed
            #   Offset by 1 to play nicely with the stage_completed
            n_steps_completing_each_stage = [0] * (len(ref_seq) + 1)

            for i in range(len(reward_matrix)):  # Iterate through the timestep
                if reward_matrix[i][current_stage] > threshold and stage_completed < len(ref_seq):
                    stage_completed += 1
                    current_stage = min(current_stage + 1, len(ref_seq)-1)
                    n_steps_completing_each_stage[stage_completed] += 1
                elif current_stage == len(ref_seq)-1 and reward_matrix[i][current_stage] > threshold:
                    # We are at the last stage
                    n_steps_completing_each_stage[stage_completed] += 1
                elif current_stage > 0 and reward_matrix[i][current_stage-1] > threshold:
                    # Once at least 1 stage is counted, if it's still above the threshold for the current stage, we will add to the count
                    n_steps_completing_each_stage[stage_completed] += 1

            pct_stage_completed = stage_completed/len(ref_seq)

            # The last pose is never reached
            if n_steps_completing_each_stage[-1] == 0:
                # We don't count any of the previous stage's steps
                pct_timesteps_completing_the_stages = 0
            else:
                pct_timesteps_completing_the_stages = np.sum(n_steps_completing_each_stage)/len(ref_seq)

            return pct_stage_completed, pct_timesteps_completing_the_stages
        
        self._success_fn_based_on_all_pos = lambda obs_seq, ref_seq=self._goal_ref_seq, threshold=success_fn_cfg["threshold_for_all_pos"]: success_fn(obs_seq, ref_seq, threshold)

        self._success_fn_based_on_only_arm_pos = lambda obs_seq, ref_seq=self._goal_ref_seq, threshold=success_fn_cfg["threshold_for_arm_pos"]: success_fn(obs_seq[:, 12:], ref_seq[:, 12:], threshold)


    def add_success_results(self, curr_timestep, timestep_success_dict):
        """
        Add the success results to the success_results dictionary
        """
        self._success_results[curr_timestep] = timestep_success_dict

        with open(self._success_json_save_path, "w") as f:
            json.dump(self._success_results, f, indent=4)

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            # Saving for only one env (the first env)
            #   Because we are using this to plot
            raw_screens = []
            screens = []
            infos = []
            # Saving for each env
            states = []
            rewards = []
            geom_xposes = [[] for _ in range(self._n_eval_episodes)]

            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in
                the captured `screens` list

                :param _locals: A dictionary containing all local variables of the
                 callback's scope
                :param _globals: A dictionary containing all global variables of the
                 callback's scope
                """
                env_i = _locals['i']

                if env_i == 0:
                    screen = self._eval_env.render()

                    image_int = np.uint8(screen)[:self._render_dim[0], :self._render_dim[1], :]

                    raw_screens.append(Image.fromarray(image_int))
                    screens.append(Image.fromarray(image_int))  # The frames here will get plotted with info later
                    infos.append(_locals.get('info', {}))

                    states.append(_locals["observations"][:, :22])
                    rewards.append(_locals["rewards"])

                geom_xpos = _locals.get('info', {})["geom_xpos"]

                # Normalize the joint states based on the torso (index 1)
                geom_xpos = geom_xpos - geom_xpos[1]
                geom_xposes[env_i].append(geom_xpos)
            
            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )

            # Save the raw_screens locally
            imageio.mimsave(os.path.join(self._rollout_save_path, f"{self.num_timesteps}_rollouts.gif"), raw_screens, duration=1/30, loop=0)

            states = np.array(states)  # size: (rollout_length, n_eval_episodes, 22)
            rewards = np.array(rewards) # size: (rollout_length, n_eval_episodes)
            geom_xposes = np.array(geom_xposes) # (n_eval_episodes, rollout_length, 18, 3)

            if self._calc_gt_reward:
                # Calculate the goal matching reward
                if self._use_geom_xpos:            
                    full_pos_success_rate_list = []
                    full_pos_pct_success_timesteps_list = []
                    arm_pos_success_rate_list = []
                    arm_pos_pct_success_timesteps_list = []

                    for env_i in range(self._n_eval_episodes):
                        # Don't need to do anything here, geom_xpos is getting normalized in the grab_screens function
                        geom_xposes_to_process = geom_xposes[env_i]

                        full_pos_success_rate, full_pos_pct_success_timesteps = self._success_fn_based_on_all_pos(geom_xposes_to_process)
                        arm_pos_success_rate, arm_pos_pct_success_timesteps = self._success_fn_based_on_only_arm_pos(geom_xposes_to_process)

                        full_pos_success_rate_list.append(full_pos_success_rate)
                        full_pos_pct_success_timesteps_list.append(full_pos_pct_success_timesteps)
                        arm_pos_success_rate_list.append(arm_pos_success_rate)
                        arm_pos_pct_success_timesteps_list.append(arm_pos_pct_success_timesteps)

                    full_pos_success_rate_iqm, full_pos_success_rate_std = calc_iqm(full_pos_success_rate_list)
                    full_pos_pct_success_timesteps_iqm, full_pos_pct_success_timesteps_std = calc_iqm(full_pos_pct_success_timesteps_list)
                    arm_pos_success_rate_iqm, arm_pos_success_rate_std = calc_iqm(arm_pos_success_rate_list)
                    arm_pos_pct_success_timesteps_iqm, arm_pos_pct_success_timesteps_std = calc_iqm(arm_pos_pct_success_timesteps_list)

                    # Save the success results locally
                    self.add_success_results(self.num_timesteps, {
                        "full_pos_success_rate": full_pos_success_rate_list,
                        "full_pos_success_rate_iqm": float(full_pos_success_rate_iqm),
                        "full_pos_success_rate_std": float(full_pos_success_rate_std),
                        "full_pos_pct_success_timesteps": full_pos_pct_success_timesteps_list,
                        "full_pos_pct_success_timesteps_iqm": float(full_pos_pct_success_timesteps_iqm),
                        "full_pos_pct_success_timesteps_std": float(full_pos_pct_success_timesteps_std),
                        "arm_pos_success_rate": arm_pos_success_rate_list,
                        "arm_pos_success_rate_iqm": float(arm_pos_success_rate_iqm),
                        "arm_pos_success_rate_std": float(arm_pos_success_rate_std),
                        "arm_pos_pct_success_timesteps": arm_pos_pct_success_timesteps_list,
                        "arm_pos_pct_success_timesteps_iqm": float(arm_pos_pct_success_timesteps_iqm),
                        "arm_pos_pct_success_timesteps_std": float(arm_pos_pct_success_timesteps_std)
                    })
                    
                    self.logger.record("eval/full_pos_success", 
                                        full_pos_success_rate_iqm, 
                                        exclude=("stdout", "log", "json", "csv"))
                    
                    self.logger.record("eval/full_pos_pct_success_timesteps", 
                                        full_pos_pct_success_timesteps_iqm, 
                                        exclude=("stdout", "log", "json", "csv"))
                    
                    self.logger.record("eval/arm_pos_success",
                                        arm_pos_success_rate_iqm,
                                        exclude=("stdout", "log", "json", "csv"))
                    
                    self.logger.record("eval/arm_pos_pct_success_timesteps",
                                        arm_pos_pct_success_timesteps_iqm,
                                        exclude=("stdout", "log", "json", "csv"))
                else:
                    raise NotImplementedError(f"Ground truth reward calculation for self._use_geom_xpos={self._use_geom_xpos} is False")

                # Show the success rate for the 1st env's rollout
                reward_matrix = np.exp(-euclidean_distance_advanced(geom_xposes[0], self._goal_ref_seq))
                arm_reward_matrix = np.exp(-euclidean_distance_advanced_arms_only(geom_xposes[0], self._goal_ref_seq))
                if self._calc_matching_reward:
                    reward_matrix_using_seq_req = np.exp(-euclidean_distance_advanced(geom_xposes[0], self._seq_matching_ref_seq))
                    arm_reward_matrix_using_seq_req = np.exp(-euclidean_distance_advanced_arms_only(geom_xposes[0], self._seq_matching_ref_seq))
                for i in range(len(infos)):
                    # Plot the reward (exp of the negative distance) based on the ground-truth goal reference sequence
                    infos[i]["rf_r"] = str([f"{reward_matrix[i][j]:.2f}" for j in range(len(self._goal_ref_seq))]) + " | " +  str([f"{arm_reward_matrix[i][j]:.2f}" for j in range(len(self._goal_ref_seq))])
                    # Success Rate based on the entire body + based on only the arm
                    infos[i]["[all, arm] success"] = f"{full_pos_success_rate_list[0]:.2f}, {arm_pos_success_rate_list[0]:.2f}"
                    if self._calc_matching_reward:
                        # Plot the reward (exp of the negative distance) based on the reference sequence USED FOR SEQUENCE MATCHING
                        infos[i]["seqrf_r"] = str([f"{reward_matrix_using_seq_req[i][j]:.2f}" for j in range(len(self._seq_matching_ref_seq))]) + " | " + str([f"{arm_reward_matrix_using_seq_req[i][j]:.2f}" for j in range(len(self._seq_matching_ref_seq))])
            
            screen_arrays = [np.array(screen) for screen in screens]
            frames = th.from_numpy(np.stack(screen_arrays)).float().cuda(0).permute(0,3,1,2) / 255.0
            # # frames = th.from_numpy(np.array(screens)).float().cuda(0).permute(0,3,1,2) / 255.0
            
            if self._calc_visual_reward:
                logger.info("Evaluating rollout for recorder callback")
                self.model.reward_model.requires_grad_(False)
                vlm_rewards = self.model._compute_joint_rewards(
                            model=self.model.reward_model,
                            transform=self.model.image_transform,
                            frames=frames,
                            ref_joint_states=self.model._ref_joint_states,
                            batch_size=self.model.reward_model_config["reward_batch_size"],
                            ).detach().cpu().numpy()

                # To write the values on the rollout frames
                for i in range(len(infos)):
                    infos[i]["vlm_joint_match_r"] = f"{vlm_rewards[i]:.4f}"

                self.logger.record("rollout/avg_vlm_total_reward", 
                                np.mean(vlm_rewards + rewards), 
                                exclude=("stdout", "log", "json", "csv"))

            # TODO: We can potentially also do VLM reward calculation
            if self._calc_matching_reward:
                if self._use_geom_xpos:
                    # Don't need to do anything here, geom_xpos is getting normalized in the grab_screens function
                    matching_reward, matching_reward_info = self._matching_fn(geom_xposes[0], self._seq_matching_ref_seq)
                else:
                    matching_reward, matching_reward_info = self._matching_fn(np.array(states[:, 0])[:, :22], self._seq_matching_ref_seq)

                self.logger.record("rollout/avg_matching_reward", 
                                np.mean(matching_reward)/self._scale, 
                                exclude=("stdout", "log", "json", "csv"))

                # Add the matching_reward to the infos so that we can plot it
                for i in range(len(infos)):
                    infos[i]["matching_reward"] = f"{matching_reward[i]:.2f}"

                # Save the matching_rewards locally    
                with open(os.path.join(self._rollout_save_path, f"{self.num_timesteps}_rollouts_matching_rewards.npy"), "wb") as f:
                    np.save(f, np.array(matching_reward))

                if self._plot_matching_visualization:
                    # TODO: For now, we can only visualize this when the reference frame is defined via a gif
                    matching_reward_viz_save_path = os.path.join(self._rollout_save_path, f"{self.num_timesteps}_matching_fn_viz.png")

                    # Subsample the frames. Otherwise, the visualization will be too long
                    if len(raw_screens) > 20:
                        obs_seq_skip_step = int(0.1 * len(raw_screens))
                        raw_screens_used_to_plot = np.array([raw_screens[i] for i in range(obs_seq_skip_step, len(raw_screens), obs_seq_skip_step)])
                    else:
                        raw_screens_used_to_plot = np.array(raw_screens)
                        
                    if len(self._seq_matching_ref_seq_frames) > 8:
                        ref_seq_skip_step = max(int(0.1 * len(self._seq_matching_ref_seq_frames)), 2)
                        ref_seqs_used_to_plot = np.array([self._seq_matching_ref_seq_frames[i] for i in range(ref_seq_skip_step, len(self._seq_matching_ref_seq_frames), ref_seq_skip_step)])
                    else:
                        ref_seqs_used_to_plot = self._seq_matching_ref_seq_frames
                    seq_matching_viz(
                        matching_fn_name=self._matching_fn_name,
                        obs_seq=raw_screens_used_to_plot,
                        ref_seq=ref_seqs_used_to_plot,
                        matching_reward=matching_reward,
                        info=matching_reward_info,
                        reward_vmin=self._reward_vmin,
                        reward_vmax=self._reward_vmax,
                        path_to_save_fig=matching_reward_viz_save_path,
                        rolcol_size=2
                    )

                    # Log the image to wandb
                    img = Image.open(matching_reward_viz_save_path)
                    self.logger.record(
                        "trajectory/matching_fn_viz",
                        LogImage(np.array(img), dataformats="HWC"),
                        exclude=("stdout", "log", "json", "csv"),
                    )

            # Plot info on the frames  
            for i in range(len(screens)):
                plot_info_on_frame(screens[i], infos[i])

            screens = [np.uint8(s).transpose(2, 0, 1) for s in screens]

            # Log to wandb
            self.logger.record(
                "trajectory/video",
                Video(th.ByteTensor(array([screens])), fps=40),
                exclude=("stdout", "log", "json", "csv"),
            )

            # Save the rollouts locally    
            with open(os.path.join(self._rollout_save_path, f"{self.num_timesteps}_rollouts_states.npy"), "wb") as f:
                np.save(f, np.array(states))

            with open(os.path.join(self._rollout_save_path, f"{self.num_timesteps}_rollouts_geom_xpos_states.npy"), "wb") as f:
                np.save(f, np.array(geom_xposes))
            
            with open(os.path.join(self._rollout_save_path, f"{self.num_timesteps}_rollouts_rewards.npy"), "wb") as f:
                np.save(f, np.array(rewards))

        return True


class WandbCallback(SB3WandbCallback):
    def __init__(
        self,
        model_save_path: str,
        model_save_freq: int,
        **kwargs,
    ):
        super().__init__(
            model_save_path=model_save_path,
            model_save_freq=model_save_freq,
            **kwargs,
        )

    def save_model(self) -> None:
        model_path = os.path.join(
        self.model_save_path, f"model_{self.model.num_timesteps}_steps.zip"
        )
        self.model.save(model_path)


class MetaWorldVideoRecorderCallback(BaseCallback):
    """
    Records a video of an agent's trajectory for a MetaWorld environment
    and logs it to TensorBoard (and/or local disk) at a specified frequency.
    """
    def __init__(
        self,
        eval_env: gymnasium.Env,
        rollout_save_path: str,
        render_freq: int,
        n_eval_episodes: int = 1,
        deterministic: bool = True,
        verbose: int = 0,
        env_id: str = None
    ):
        """
        :param eval_env: A single MetaWorld environment instance for evaluation.
        :param rollout_save_path: Directory in which to save the recorded GIFs.
        :param render_freq: Evaluate and record a video every `render_freq` calls of the callback.
        :param n_eval_episodes: Number of episodes to record each time we do a video evaluation.
        :param deterministic: Whether to use a deterministic or stochastic policy for evaluation.
        :param verbose: Verbosity level.
        :param env_id: The environment ID to use for rendering.
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.render_freq = render_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.rollout_save_path = rollout_save_path
        self.env_id = env_id
        # Ensure the save path exists
        os.makedirs(self.rollout_save_path, exist_ok=True)

    def _on_step(self) -> bool:
        """
        Called at every environment step. We check if it's time to evaluate and, if so,
        run evaluation and record results.
        """
        if self.n_calls % self.render_freq == 0:
            self._run_evaluation()
        return True

    def _run_evaluation(self):
        """
        Runs a meta-world evaluation for `n_eval_episodes`, collecting:
          - states,
          - rewards,
          - success metrics (like final_success_rate),
          - frames for video logging.

        Then logs to the SB3 logger and saves data as .npy / .gif.
        """
        if self.verbose > 0:
            print(f"[MetaWorldEvalCallback] Running evaluation at step={self.n_calls}...")

        all_states = []
        all_frames = []
        all_successes = []
        # store raw frames from the *first* few episodes for visualization
        all_frames_for_video = []

        # Accumulate final successes (e.g. did we achieve the goal by the end of the episode?)
        # Or you can track partial successes each step. Example below: "goal_achieved" per step
        final_successes_count = 0

        for ep_i in range(self.n_eval_episodes):
            print(f"[MetaWorldEvalCallback] Running episode {ep_i} of {self.n_eval_episodes}")
            ep_states = []
            ep_frames = []
            ep_successes = 0

            obs = self.eval_env.reset()
            done, truncated = False, False

            # step_i = 0
            while not (done or truncated):
                # print(f"[MetaWorldEvalCallback] Running step {step_i} | {ep_i} of {self.n_eval_episodes}")
                # step_i += 1

                frame = self.eval_env.render()
                if frame is not None:
                    ep_frames.append(frame)

                # save the observations from the first 10 eval episodes
                if ep_i < 10:
                    ep_states.append(obs)

                # Convert obs to a tensor if needed or use the stable-baselines policy directly
                with th.no_grad():
                    action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, _, done, info = self.eval_env.step(action)

                ep_states.append(obs)
                # print(f"[MetaWorldEvalCallback] Step {step_i} | {info} | {type(info)} | {len(info)}")
                if "success" in info[0] and info[0]["success"]:
                    ep_successes += info[0]["success"]
            
            if ep_i < 10:
                all_states.append(np.stack(ep_states))

            # We consider the final step's success as "did the agent succeed by the end of the episode?"
            # info[0].keys(): dict_keys(['success', 'near_object', 'grasp_success', 'grasp_reward', 'in_place_reward', 'obj_to_target', 'unscaled_reward', 'episode', 'TimeLimit.truncated', 'terminal_observation', 'render_array'])
            # print(f"[MetaWorldEvalCallback] Episode {ep_i} | {info} | {type(info)} | {len(info)} | {info[0].keys()}")
            final_successes_count += info[0]["success"]
            all_states.append(ep_states)
            all_successes.append(ep_successes)
            all_frames_for_video.append(ep_frames)

        # -------------
        # Compute metrics
        # -------------
        # "final_success_rate" is how many episodes ended in success
        final_success_rate = final_successes_count / self.n_eval_episodes

        # Suppose "total_successes" is the sum of success signals within an episode.
        total_successes_array = np.array(all_successes)  # shape (n_eval_episodes,)

        # Basic stats
        total_successes_mean, total_successes_std = mean_and_std(total_successes_array)
        total_successes_iqm, total_successes_lower, total_successes_upper = interquartile_mean_and_ci(total_successes_array)

        # -------------
        # Log metrics to SB3 logger
        # -------------
        self.logger.record("eval/final_success_rate", final_success_rate)
        self.logger.record("eval/total_success_mean", total_successes_mean)
        self.logger.record("eval/total_success_std", total_successes_std)
        self.logger.record("eval/total_success_iqm", total_successes_iqm)
        self.logger.record("eval/total_success_lower", total_successes_lower)
        self.logger.record("eval/total_success_upper", total_successes_upper)

        # -------------
        # Save states & rewards as .npy
        # -------------
        # We typically name them with the current global step for reference
        # step_folder = os.path.join(self.rollout_save_path, f"eval_{self.num_timesteps}")
        # os.makedirs(step_folder, exist_ok=True)

        np.save(os.path.join(self.rollout_save_path, f"{self.num_timesteps}.npy"), np.array(all_states, dtype=object))
        np.save(os.path.join(self.rollout_save_path, f"{self.num_timesteps}_return.npy"), total_successes_array)

        # Also save a quick JSON with the metrics
        eval_metrics = {
            "eval/final_success_rate": final_success_rate,
            "eval/total_success_mean": total_successes_mean,
            "eval/total_success_std": total_successes_std,
            "eval/total_success_iqm": total_successes_iqm,
            "eval/total_success_lower": total_successes_lower,
            "eval/total_success_upper": total_successes_upper,
        }
        with open(os.path.join(self.rollout_save_path, f"eval_metrics_{self.num_timesteps}.json"), "w") as f:
            json.dump(eval_metrics, f, indent=4)
        


        # -------------
        # Create & log a short video (GIF) to SB3 logger
        # -------------
        # We can just take the first episode's frames for the GIF to keep it short
        # or you can combine frames from multiple episodes if desired.
        frames_to_gif = all_frames_for_video[0] if len(all_frames_for_video) > 0 else []

        # Save to local .gif
        gif_path = os.path.join(self.rollout_save_path, f"rollout_{self.num_timesteps}.gif")
        if len(frames_to_gif) > 0:
            # Convert frames to PIL and save as GIF
            imageio.mimsave(gif_path, [Image.fromarray(np.uint8(f)) for f in frames_to_gif],
                            duration=1/30, loop=0)

            # Convert frames to SB3's Video format: (T, C, H, W)
            tensors = []
            for frm in frames_to_gif:
                # Expect shape (H,W,3). Convert to (3,H,W)
                frm_t = th.ByteTensor(np.uint8(frm).transpose(2, 0, 1))
                tensors.append(frm_t)
            video_tensor = th.stack(tensors, dim=0)  # (T, C, H, W)

            # Record the video in the SB3 logger
            self.logger.record(
                "eval/video",
                Video(video_tensor.unsqueeze(0), fps=30),  # add batch dimension
                exclude=("stdout", "log", "json", "csv"),
            )

        if self.verbose > 0:
            print(f"[MetaWorldEvalCallback] Finished evaluation at step={self.n_calls}")
            print(f"  final_success_rate={final_success_rate:.3f}, mean successes={total_successes_mean:.3f}")
