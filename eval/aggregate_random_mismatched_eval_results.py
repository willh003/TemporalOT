"""
Typical usage: 
1. Make sure `eval_path_csv` has the updated csvs
2. In the main directory for TemporalOT, run
    python -m eval.gen_random_mismatched_eval_results
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json

from utils.math_utils import mean_and_se


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--speed_type', type=str, required=True, choices=['fast', 'slow', 'mixed'], help='Domain name')
parser.add_argument('-m', '--metric', type=str, required=False, choices=['std', 'diff', 'cv'], help='Metric to use for clustering')
args = parser.parse_args()

# Load the CSV file
speed_type = args.speed_type  # options: 'slow', 'fast', 'mixed'

csv_file = os.path.join("eval/eval_path_csv", f"metaworld_random_{speed_type}_ablation.csv")
df = pd.read_csv(csv_file)

# Approaches
tasks_to_include = ["Door-open", "Window-open", "Lever-pull"]
approaches = ["TemporalOT", "ORCA+TOT pretrained (500k-500k)"]

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
                        return_values = np.load(file)
                        results[task_name][mismatch_level][run_num][approach].append(return_values)
                except Exception as e:
                    print(f"Error reading {final_eval_path}: {e}")
            else:
                print(f"Path {path} does not exist for {approach}")


"""
For each task, calculate the average difference between the subsections, and split the path into 3 even groups
"""
dict_from_std = {
    "Low": [],
    "Medium": [],
    "High": []
}

dict_from_cv = {
    "Low": [],
    "Medium": [],
    "High": []
}

dict_from_diff = {
    "Low": [],
    "Medium": [],
    "High": []
}


for tb_task_name in ["Door-open", "Window-open", "Lever-pull"]:
    task_name = tb_task_name.lower() + "-v2"

    demo_std_list = []
    demo_cv_list = []
    demo_diff_list = []

    for level in [1, 3, 5]:
        for i in range(3):
            with open(f"/share/portal/wph52/TemporalOT/create_demo/metaworld_demos/{task_name}/random_mismatched_{speed_type}/{level}outof5_mismatched/{level}outof5_mismatched_{i}/{task_name}_corner3_0_mismatched_info.json") as f:
                info = json.load(f)
                subsection_lens = [len(info[subsection]["subsampled_indices"]) for subsection in info.keys()]
                demo_len = np.sum(subsection_lens)
                demo_std = np.std(subsection_lens)
                demo_std_list.append((tb_task_name, level, i, demo_std))
                demo_cv_list.append((tb_task_name, level, i, demo_std/np.mean(subsection_lens)))

                diff = 0
                for i in range(len(subsection_lens)-1):
                    for j in range(i+1, len(subsection_lens)):
                        diff += abs(subsection_lens[i] - subsection_lens[j])
                diff = diff / demo_len / 10.0
                demo_diff_list.append((tb_task_name, level, i, diff))

    # Sort the list based on the 3rd element in each tuple (from smallest to largest)
    demo_std_list.sort(key=lambda x: x[3])
    demo_cv_list.sort(key=lambda x: x[3])
    demo_diff_list.sort(key=lambda x: x[3])
    
    # Split the list into 3 even groups
    for i, result_lvl in enumerate(["Low", "Medium", "High"]):
        dict_from_std[result_lvl].extend(demo_std_list[i*3:(i+1)*3])
        dict_from_cv[result_lvl].extend(demo_cv_list[i*3:(i+1)*3])
        dict_from_diff[result_lvl].extend(demo_std_list[i*3:(i+1)*3])

means_plot = {approach: [] for approach in approaches}
ses_plot = {approach: [] for approach in approaches}

if args.metric == 'std':
    dict_to_use = dict_from_std
elif args.metric == 'cv':
    dict_to_use = dict_from_cv
else:
    dict_to_use = dict_from_diff

if speed_type == 'slow':
    order_for_result_lvl = ['High', 'Medium', 'Low']
else:
    order_for_result_lvl = ['Low', 'Medium', 'High']

for approach in approaches:
    for result_lvl in order_for_result_lvl:
        all_values = []
        for task_name, level, i, _ in dict_to_use[result_lvl]:
            all_values.extend(results[task_name][f"{level}outof5"][str(i)][approach])

        all_values = np.array(all_values).flatten()

        if len(all_values) > 0:
            mean_val, se_val = mean_and_se(all_values)
        else:
            mean_val, se_val = -1, -1

        means_plot[approach].append(mean_val)
        ses_plot[approach].append(se_val)


"""##################################################################################

        Plot a bar plot with mean and standard error

##################################################################################"""

from .eval_constants import APPROACH_COLOR_DICT, APPROACH_NAME_TO_PLOT

x = np.arange(3)  # the label locations
width = 0.35  # the width of the bars

plt.grid(True, linestyle='--', alpha=0.3, zorder=0)

# Plotting the bars
for i, approach in enumerate(approaches):
    plt.bar(x + (i - 1) * width, means_plot[approach], width, label=APPROACH_NAME_TO_PLOT[approach], color=APPROACH_COLOR_DICT[approach], zorder=3)
    plt.errorbar(x + (i - 1) * width, means_plot[approach], ses_plot[approach], fmt='none', ecolor='black', capsize=5, zorder=4)

# Add the mean values on top of the bars
for i, approach in enumerate(approaches):
    for j, mean_val in enumerate(means_plot[approach]):
        plt.text(j + (i - 1) * width, mean_val + 0.5, f"{mean_val:.2f}", ha='center', va='bottom', fontsize=16)

# Adding labels, title, and legend
plt.xlabel(f'Misaligned Level ({"Sped Up" if speed_type == "fast" else "Slowed Down"})', fontsize=20)
plt.ylabel('Cumulative Return', fontsize=20)
# ax.set_title('Total Results for Approaches with Mismatch Levels')

ordered_xticks = ["Low", "Medium", "High"]
if speed_type == 'slow':
    # reverse the order
    ordered_xticks = ordered_xticks[::-1]
plt.xticks(x, ordered_xticks, fontsize=16)
plt.ylim([0, 20])

plt.legend(fontsize=16, loc='upper right', ncol=2)

# Display the plot
plt.tight_layout()

# Save the plot
output_plot = os.path.join("eval/eval_agg_results", f"metaworld_reclustered={args.metric}_random_{speed_type}_mismatched_result.png")
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
plt.close()

print(f"Plot saved to {output_plot}")
