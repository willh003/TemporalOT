import pandas as pd
import os
import json

exp = "mismatched"

df = pd.read_csv(f"/share/portal/hw575/TemporalOT/eval/eval_path_csv/metaworld_{exp}.csv")

checkpoint_path = {}

# Iterate through each task and approach
for index, row in df.iterrows():
    task_name = row['Tasks'].lower() + "-v2"

    if task_name not in checkpoint_path:
        checkpoint_path[task_name] = {}
    
    path = row['TemporalOT']

    if isinstance(path, str) and os.path.exists(path):
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