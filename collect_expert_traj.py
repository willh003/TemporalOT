# An example of generating expert demos by MetaWorld's scripted policies.
# Warning: The scripted policies may generate failure trajectories. You can write a task-dependent criterion to filter the generated trajectories.

import metaworld
import random
import metaworld.policies as policies
import cv2
import numpy as np

import pickle
import imageio
from pathlib import Path
from collections import deque
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from utils import record_demo


POLICY = {
    'hammer-v2': policies.SawyerHammerV2Policy,
    'drawer-close-v2': policies.SawyerDrawerCloseV2Policy,
    'drawer-open-v2': policies.SawyerDrawerOpenV2Policy,
    'door-open-v2': policies.SawyerDoorOpenV2Policy,
    'bin-picking-v2': policies.SawyerBinPickingV2Policy,
    'button-press-topdown-v2': policies.SawyerButtonPressTopdownV2Policy,
    'door-unlock-v2': policies.SawyerDoorUnlockV2Policy,
    'basketball-v3': policies.SawyerBasketballV2Policy,
    'plate-slide-v2': policies.SawyerPlateSlideV2Policy,
    "hand-insert-v2": policies.SawyerHandInsertV2Policy,  
    "peg-insert-side-v2": policies.SawyerPegInsertionSideV2Policy,  
    'assembly-v3': policies.SawyerAssemblyV2Policy,
    'push-wall-v2': policies.SawyerPushWallV2Policy,
    'soccer-v2': policies.SawyerSoccerV2Policy,
    'disassemble-v2': policies.SawyerDisassembleV2Policy,
    'pick-place-wall-v3': policies.SawyerPickPlaceWallV2Policy,
    'pick-place-v2': policies.SawyerPickPlaceV2Policy,
    'push-v2': policies.SawyerPushV2Policy,
    'push-wall-v2': policies.SawyerPushWallV2Policy,
    'lever-pull-v2': policies.SawyerLeverPullV2Policy,
    'stick-pull-v2': policies.SawyerStickPullV2Policy,
    'shelf-place-v2': policies.SawyerShelfPlaceV2Policy,
    'window-close-v2': policies.SawyerWindowCloseV2Policy,
    'window-open-v2': policies.SawyerWindowOpenV2Policy,
    'reach-v2': policies.SawyerReachV2Policy,
    'button-press-wall-v2': policies.SawyerButtonPressWallV2Policy,
    'box-close-v2': policies.SawyerBoxCloseV2Policy,
    'stick-push-v2': policies.SawyerStickPushV2Policy,
    'handle-pull-v2': policies.SawyerHandlePullV2Policy,
    'door-lock-v2': policies.SawyerDoorLockV2Policy,
}


CAMERA = {
    'hammer-v2': 'corner3',
    'drawer-close-v2': 'corner',
    'drawer-open-v2': 'corner',
    'door-open-v2': 'corner3',
    'bin-picking-v2': 'corner',
    'button-press-topdown-v2': 'corner',
    'door-unlock-v2': 'corner',
    'basketball-v3': 'corner',
    'plate-slide-v2': 'corner',
    'hand-insert-v2': 'corner',
    'peg-insert-side-v2': 'corner3',
    'assembly-v3': 'corner',
    'push-wall-v2': 'corner',
    'soccer-v2': 'corner',
    'disassemble-v2': 'corner',
    'pick-place-wall-v3': 'corner3',
    'pick-place-v2': 'corner3',
    'push-v2': 'corner3',
    'push-wall-v2': 'corner',
    'lever-pull-v2': 'corner4',
    'stick-pull-v2': 'corner3',
    'shelf-place-v2': 'corner',
    'window-close-v2': 'corner3',
    'window-open-v2': 'corner3',
    'reach-v2': 'corner3',
    'button-press-wall-v2': 'corner',
    'box-close-v2': 'corner3',
    'stick-push-v2': 'corner',
    'handle-pull-v2': 'corner3',
    'door-lock-v2': 'corner',
}


MAX_PATH_LENGTH = {
    'hammer-v2': 125,
    'drawer-close-v2': 125,
    'drawer-open-v2': 125,
    'door-open-v2': 125,
    'bin-picking-v2': 175,
    'button-press-topdown-v2': 125,
    'door-unlock-v2': 125,
    'basketball-v3': 175,
    'plate-slide-v2': 125,
    'hand-insert-v2': 125,
    'peg-insert-side-v2': 150,
    'assembly-v3': 175,
    'push-wall-v2': 175,
    'soccer-v2': 125,
    'disassemble-v2': 125,
    'pick-place-wall-v3': 175,
    'pick-place-v2': 125,
    'push-v2': 125,
    'push-wall-v2': 175,
    'lever-pull-v2': 175,
    'stick-pull-v2': 175,
    'shelf-place-v2': 175,
    'window-close-v2': 125,
    'window-open-v2': 125,
    'reach-v2': 125,
    'button-press-wall-v2': 125,
    'box-close-v2': 175,
    'stick-push-v2': 125,
    'handle-pull-v2': 175,
    'door-lock-v2': 125,
}


def record_video(fname, large_images):
    frames = []
    for img in large_images:
        frames.append(np.transpose(img[-3:], (1, 2, 0)))
    imageio.mimsave(f"{fname}.mp4", frames, fps=60)


num_demos = 20
env_name = "basketball-v3"
stop = False  # tune this for different task


policy = POLICY[env_name]()
env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[f"{env_name}-goal-observable"]()
env._freeze_rand_vec = False


images_list = list()
large_images_list = list()
observations_list = list()
actions_list = list()
rewards_list = list()
episode = 0
while episode < num_demos:
    print(f"Episode {episode}")
    images = list()
    large_images = list()
    observations = list()
    actions = list()
    rewards = list()
    success = 0
    image_stack = deque([], maxlen=3)
    large_image_stack = deque([], maxlen=3)
    goal_achieved = 0

    # Reset env
    observation = env.reset()  # Reset environment
    print(f"Goal: ({observation[-3]:.3f}, {observation[-2]:.3f}, {observation[-1]:.3f})")
    num_steps = MAX_PATH_LENGTH[env_name]
    for step in range(num_steps):
        # Get frames
        pixel = env.render(offscreen=True, camera_name=CAMERA[env_name])
        frame = cv2.resize(pixel.copy(), (84, 84))
        frame = np.transpose(frame, (2, 0, 1))
        image_stack.append(frame)
        while (len(image_stack) < 3):
            image_stack.append(frame)
        images.append(np.concatenate(image_stack, axis=0))
        large_frame = cv2.resize(pixel.copy(), (224, 224))
        large_frame = np.transpose(large_frame, (2, 0, 1))
        large_image_stack.append(large_frame)
        while (len(large_image_stack) < 3):
            large_image_stack.append(large_frame)
        large_images.append(np.concatenate(large_image_stack, axis=0))

        # Get action
        if stop and success:
            # action = actions[-1]
            action = np.zeros_like(actions[-1])
        else:
            action = policy.get_action(observation)
            action = np.clip(action, -1.0, 1.0)
        actions.append(action)

        # Get observation
        observation[-3:] = 0
        observations.append(observation)

        # Act in the environment
        observation, reward, done, info = env.step(action)
        rewards.append(reward)
        success = info['success']
        goal_achieved += success

    # You can write a task-dependent criterion to filter the generated trajectories!!!
    if success == 0:
        continue

    record_video(f"{env_name}_{episode}", large_images)

    # Store trajectory
    episode = episode + 1
    images_list.append(np.array(images))
    large_images_list.append(np.array(large_images))
    observations_list.append(np.array(observations))
    actions_list.append(np.array(actions))
    rewards_list.append(np.array(rewards))


# data = [images_list, observations_list, actions_list, rewards_list, large_images_list]
data = large_images_list


# check if the demos are correct
# for idx, images in large_images_list:
#     record_demo(images, "./", f"demo_origin{idx}")


# save two demos
data = [large_images_list[0], large_images_list[1]]
with open(f"data/expert_demos/{env_name}.pkl", 'wb') as f: 
    pickle.dump([data[0][::2], data[1][::2]], f)  # action repeat is 2
