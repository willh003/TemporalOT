"""
Randomly select N videos to use for ablation

Usage:
python -m demo.determine_multi_video_for_ablation

Note: We only use once so that we can generate the dictionary multi_video_ablation_dict.json. Then use in training. 
"""
import random

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json

from utils.math_utils import mean_and_se
from .eval_constants import get_demo_gif_path

N_VIDEO_PER_TASK_PER_LEVEL = 1

# Approaches
tasks_to_include = ["Door-open", "Window-open", "Lever-pull"]
approaches = ["TemporalOT", "ORCA+TOT pretrained (500k-500k)"]

video_dict = {task: [] for task in tasks_to_include}

for speed_type in ['slow', 'fast']:
    csv_file = os.path.join("eval/eval_path_csv", f"metaworld_random_{speed_type}_ablation.csv")
    df = pd.read_csv(csv_file)

    # Initialize a dictionary to store results
    """
    {
        task_name: {
            mismatch_level: {
                approach: [values]
            }
        }
    }
    """
    results = {}

    # Iterate through each task and approach
    for index, row in df.iterrows():
        task_name = row['Tasks']
        mismatch_level = row['Mismatched Level']
        run_num = str(row['Run Num'])

        if task_name in tasks_to_include:
            # Load the return from the orginal demonstration
            og_demo_path = get_demo_gif_path("metaworld", task_name.lower() + "-v2", "d", demo_num=0, mismatched=False)
            # Load the success vector
            success = np.load(os.path.splitext(og_demo_path)[0] + "_success.npy")
            expert_return = np.sum(success)
            
            # Initialize storage for the task if not already present
            if task_name not in results:
                results[task_name] = {}

            if mismatch_level not in results[task_name]:
                results[task_name][mismatch_level] = {}

            if run_num not in results[task_name][mismatch_level]:
                results[task_name][mismatch_level][run_num] = {approach: [] for approach in approaches}

            for approach in approaches:
                path = row[approach]
                if isinstance(path, str) and os.path.exists(path):
                    # Assume each folder contains a file named `results.txt` with a single float value
                    if approach == "ORCA+TOT pretrained (500k-500k)":
                        final_eval_path = os.path.join(path, "eval", "500000_return.npy")
                    else:
                        final_eval_path = os.path.join(path, "eval", "1000000_return.npy")

                    try:
                        with open(final_eval_path, 'rb') as file:
                            return_values = np.load(file) / expert_return # Normalize by the expert return
                            results[task_name][mismatch_level][run_num][approach].append(return_values)
                    except Exception as e:
                        print(f"Error reading {final_eval_path}: {e}")
                else:
                    print(f"Path {path} does not exist for {approach}")


    """
    For each task, calculate the average difference between the subsections, and split the path into 3 even groups
    """
    dict_from_mad = {
        "Low": [],
        "Medium": [],
        "High": []
    }

    dict_from_mad_with_filepath = {
        "Low": [],
        "Medium": [],
        "High": []
    }


    for tb_task_name in ["Door-open", "Window-open", "Lever-pull"]:
        task_name = tb_task_name.lower() + "-v2"

        demo_mad_list = []
        demo_mad_with_filepath_list = []

        for level in [1, 3, 5]:
            for i in range(3):
                with open(f"/share/portal/hw575/TemporalOT/create_demo/metaworld_demos/{task_name}/random_mismatched_{speed_type}/{level}outof5_mismatched/{level}outof5_mismatched_{i}/{task_name}_corner3_0_mismatched_info.json") as f:
                    info = json.load(f)
                    subsection_lens = [len(info[subsection]["subsampled_indices"]) for subsection in info.keys()]
                    subsection_prop = [l/np.sum(subsection_lens) for l in subsection_lens]

                    list_to_use = subsection_lens

                    demo_mad_list.append((tb_task_name, level, i, np.mean(np.abs(list_to_use - np.mean(list_to_use)))))
                    demo_mad_with_filepath_list.append((tb_task_name, level, i, np.mean(np.abs(list_to_use - np.mean(list_to_use)))))

        # Sort the list based on the 3rd element in each tuple (from smallest to largest)
        demo_mad_list.sort(key=lambda x: x[3])
        demo_mad_with_filepath_list.sort(key=lambda x: x[3])
        
        # Split the list into 3 even groups
        for i, result_lvl in enumerate(["Low", "Medium", "High"]):
            dict_from_mad[result_lvl].extend(demo_mad_list[i*3:(i+1)*3])
            dict_from_mad_with_filepath[result_lvl].extend(demo_mad_with_filepath_list[i*3:(i+1)*3])

    print("Dict from mad with filepath: speed_type = ", speed_type)
    print(dict_from_mad_with_filepath)

    for result_lvl in ["Low", "Medium"]:
        for task in tasks_to_include:
            # Randomly select N videos to use for ablation
            valid_videos = [dict_from_mad_with_filepath[result_lvl][i] for i in range(len(dict_from_mad_with_filepath[result_lvl])) if dict_from_mad_with_filepath[result_lvl][i][0] == task]

            # Randomly select N videos to use for ablation
            selected_videos = random.sample(valid_videos, N_VIDEO_PER_TASK_PER_LEVEL)

            video_dict[task].append((speed_type, result_lvl, selected_videos))


print("Final video dict")
print(json.dumps(video_dict, indent=4))

with open(f"/share/portal/hw575/TemporalOT/create_demo/metaworld_demos/multi_video_n={N_VIDEO_PER_TASK_PER_LEVEL * 4}_ablation_dict.json", "w") as f:
    json.dump(video_dict, f, indent=4)