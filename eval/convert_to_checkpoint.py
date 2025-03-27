"""
Typical usage:
- set exp to one of the following: mismatched, matched, random_mismatched_fast, random_mismatched_slow, multi_video
- run the script
    python eval/convert_to_checkpoint.py
"""

import pandas as pd
import os
import json

exp = "multi_diff_video_20"

if exp == "mismatched" or exp == "matched" or exp == "multi_video" or exp == "multi_diff_video_4" or exp == "multi_diff_video_10" or exp == "multi_diff_video_20":
    df = pd.read_csv(f"/share/portal/hw575/TemporalOT/eval/eval_path_csv/metaworld_{exp}.csv")

    checkpoint_path = {}

    # Iterate through each task and approach
    for index, row in df.iterrows():
        task_name = row['Tasks'].lower() + "-v2"

        if task_name not in checkpoint_path:
            checkpoint_path[task_name] = {}
        
        path = row['TemporalOT']

        if isinstance(path, str):
            checkpoint_path[task_name][int(row['Runs'])] = {
                "path": row['TemporalOT'],
                "used": "unclaimed" # unclaimed, claimed, completed
            }
        else:
            checkpoint_path[task_name][int(row['Runs'])] = {
                "path": '',
                "used": "unclaimed" # unclaimed, claimed, completed
            }

    with open(f"utils/temporalot_checkpoint_path_{exp}.json", 'w') as f:
        json.dump(checkpoint_path, f, indent=4)
elif exp == "random_mismatched_fast" or exp == "random_mismatched_slow":
    speed = "fast" if exp == "random_mismatched_fast" else "slow"

    df = pd.read_csv(f"/share/portal/hw575/TemporalOT/eval/eval_path_csv/metaworld_random_{speed}_ablation.csv")

    checkpoint_path = {}

    # Iterate through each task and approach
    for index, row in df.iterrows():
        task_name = row['Tasks'].lower() + "-v2"

        if task_name not in checkpoint_path:
            checkpoint_path[task_name] = {}

        mismatched_level = int(row['Mismatched Level'][0])

        if mismatched_level not in checkpoint_path[task_name]:
            checkpoint_path[task_name][mismatched_level] = {}
        
        path = row['TemporalOT']

        if isinstance(path, str):
            checkpoint_path[task_name][mismatched_level][int(row['Seed'])] = {
                "path": row['TemporalOT'],
                "used": "unclaimed" # unclaimed, claimed, completed
            }
        else:
            checkpoint_path[task_name][mismatched_level][int(row['Seed'])] = {
                "path": '',
                "used": "unclaimed" # unclaimed, claimed, completed
            }
    
    with open(f"utils/temporalot_checkpoint_path_random_{speed}.json", 'w') as f:
        json.dump(checkpoint_path, f, indent=4)