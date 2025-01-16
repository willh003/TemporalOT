"""
Usage (at the top directory): python -m demo.create_random_mismatch_traj
"""

import os
import numpy as np
import json

from .constants import get_demo_dir, get_demo_gif_path
from .subsample_gifs_and_states import load_frames_and_states, save_frames_and_states
"""
File structure to store the randomly generated mismatch

Trim the demo after being successful consecutively for 10% of total timesteps

- task_name
    - random_mismatched
        - 1outof5_mismatched
            - 1outof5_mismatched_0
            - 1outof5_mismatched_1
            - 1outof5_mismatched_2
        - 2outof5_mismatched
            - ...
        - 3outof5_mismatched
            - ...
"""

MISMATCH_SPEED_LEVELS = [2, 4, 6, 8, 10]

def random_subsample_gif_and_states(input_gif_path, output_dir, mismatch_level, num_of_sections, pct_of_success_needed=0.1):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_gif_path))[0]
    frames, states = load_frames_and_states(input_gif_path)

    # Load the success vector
    success = np.load(os.path.splitext(input_gif_path)[0] + "_success.npy")
    # Find the beginning of the first N consecutive successes
    N_consecutive_success_needed = int(len(success) * pct_of_success_needed)
    start_idx = 0
    found_N_consecutive = False
    while start_idx < len(success) - N_consecutive_success_needed:
        if np.all(success[start_idx:start_idx + N_consecutive_success_needed]):
            found_N_consecutive = True
            break
        start_idx += 1

    if not found_N_consecutive:
        print(f"Could not find {N_consecutive_success_needed} consecutive successes in {input_gif_path}")
    
    frames = frames[:start_idx + N_consecutive_success_needed]
    states = states[:start_idx + N_consecutive_success_needed]

    print(f"Trimming the demo to {len(frames)} frames after {N_consecutive_success_needed} consecutive successes")

    num_frames = len(frames)
    num_frames_per_section = num_frames // num_of_sections

    # Randomly select the section to mismatch
    mismatched_section = np.random.choice(num_of_sections, mismatch_level, replace=False)
    mismatched_section.sort()

    mismatched_selected_idx = []
    mismatched_info = {}

    # Mismatch the selected sections
    for i in range(num_of_sections):
        if i in mismatched_section:
            # Randomly select a subsample speed level
            available_speed_levels = [level for level in MISMATCH_SPEED_LEVELS if level < num_frames_per_section]

            speed_level = np.random.choice(available_speed_levels)

            subsampled_indices = list(range(i * num_frames_per_section, (i + 1) * num_frames_per_section))[::speed_level]

            print(f"i={i}, speed_level={speed_level}, len={len(subsampled_indices)}, {subsampled_indices}")

            mismatched_selected_idx.extend(subsampled_indices)

            mismatched_info[i] = {
                'speed_level': int(speed_level),
                'subsampled_indices': subsampled_indices
            }
        else:
            mismatched_selected_idx.extend(list(range(i * num_frames_per_section, (i + 1) * num_frames_per_section)))

            mismatched_info[i] = {
                'speed_level': -1,
                'subsampled_indices': list(range(i * num_frames_per_section, (i + 1) * num_frames_per_section))
            }

        print(f"i={i}, current len={len(mismatched_selected_idx)}")

    if len(frames) - 1 not in mismatched_selected_idx:
        mismatched_selected_idx.append(len(frames) - 1)

    print(f"[len={len(mismatched_selected_idx)}] Saving the mismatched trajectory at {output_dir}")
    print("=====================")

    mismatched_frames = [frames[i] for i in mismatched_selected_idx]
    mismatched_states = states[mismatched_selected_idx]

    # Save the mismatched trajectory
    save_frames_and_states(mismatched_frames, os.path.join(output_dir, f"{base_name}.gif"), mismatched_states, os.path.join(output_dir, f"{base_name}_states.npy"))

    # Save the mismatched info as json
    with open(os.path.join(output_dir, f"{base_name}_mismatched_info.json"), 'w') as f:
        json.dump(mismatched_info, f, indent=4)
    


def create_random_mismatch_trajs(input_gif_path, mismatch_level, num_of_sections, num_traj):
    for i in range(num_traj):
        output_dir = get_demo_dir(env_name, task_name, camera_name, mismatched=False, random_mismatched_info=
                                  {'mismatch_level': f'{mismatch_level}outof{num_of_sections}_mismatched', 
                                   'run_num': i}) 
        
        if os.path.exists(input_gif_path):
            random_subsample_gif_and_states(input_gif_path, output_dir, mismatch_level, num_of_sections)
        else:
            print(f"{input_gif_path} does not exist")

if __name__ == "__main__":
    env_name = "metaworld"
    task_name = "stick-push-v2"
    camera_name = "d"
    mismatch_level = [1, 3, 5]
    num_of_sections = 5 # How many subsection to partition the demo into
    num_traj = 3
    pct_of_frames_to_trim = 0.1

    input_gif_path = get_demo_gif_path(env_name, task_name, camera_name, demo_num=0, num_frames="d") 
    # new gif path

    if type(mismatch_level) == list:
        for level in mismatch_level:
            create_random_mismatch_trajs(input_gif_path, level, num_of_sections, num_traj)
    elif type(mismatch_level) == int:
        create_random_mismatch_trajs(input_gif_path, mismatch_level, num_of_sections, num_traj)