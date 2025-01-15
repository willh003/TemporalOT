import pandas as pd
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--domain', type=str, required=True, choices=['metaworld'], help='Domain name')
parser.add_argument('-m', '--mismatched', default=False, action='store_true', help='Whether to access the mismatched experiment result')
args = parser.parse_args()

# Load the CSV file
csv_file = os.path.join("eval/eval_path_csv", f"{args.domain}_{'mismatched' if args.mismatched else 'matched'}.csv")
df = pd.read_csv(csv_file)

# Columns for approaches
approaches = ["Threshold", "RoboCLIP", "OT", "TemporalOT", "DTW", "ORCA"]

# Initialize a dictionary to store results
results = {}

# Iterate through each task and approach
for index, row in df.iterrows():
    task_key = (row['Difficulty Level'], row['Tasks'])

    # Initialize storage for the task if not already present
    if task_key not in results:
        results[task_key] = {approach: [] for approach in approaches}

    for approach in approaches:
        path = row[approach]
        if isinstance(path, str) and os.path.exists(path):
            # Assume each folder contains a file named `results.txt` with a single float value
            final_eval_path = os.path.join(path, "eval", "1000000_return.npy")

            try:
                with open(final_eval_path, 'rb') as file:
                    return_values = np.load(file)
                    results[task_key][approach].append(return_values)
            except Exception as e:
                print(f"Error reading {final_eval_path}: {e}")

# Calculate mean and std for each task and approach
aggregated_results = []
for task_key, approaches_data in results.items():
    difficulty, task = task_key
    curr_task_results = {"Difficulty Level": difficulty, "Task": task}

    for approach, values in approaches_data.items():
        if values:  # Only calculate if there are valid values
            mean_val = np.mean(values)
            std_val = np.std(values)
        else:
            mean_val = -1
            std_val = -1

        curr_task_results[approach] = f"{mean_val:.3f} ({std_val:.3f})"
    
    aggregated_results.append(curr_task_results)

# Convert aggregated results to a DataFrame
aggregated_df = pd.DataFrame(aggregated_results)

# Save the aggregated results to a new CSV
output_csv = os.path.join("eval/eval_agg_results", f"{args.domain}_{'mismatched' if args.mismatched else 'matched'}_agg_result.csv")
aggregated_df.to_csv(output_csv, index=False)

print(f"Aggregated results saved to {output_csv}")