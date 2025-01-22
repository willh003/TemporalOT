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

from utils.math_utils import mean_and_se

# Load the CSV file
csv_file = os.path.join("eval/eval_path_csv", f"metaworld_random_ablation.csv")
df = pd.read_csv(csv_file)

# Approaches
tasks_to_include = ["Door-open", "Window-open", "Lever-pull"]
approaches = ["TemporalOT", "ORCA"]

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

    if task_name in tasks_to_include:
        # Initialize storage for the task if not already present
        if task_name not in results:
            results[task_name] = {}

        if mismatch_level not in results[task_name]:
            results[task_name][mismatch_level] = {approach: [] for approach in approaches}

        for approach in approaches:
            path = row[approach]
            if isinstance(path, str) and os.path.exists(path):
                # Assume each folder contains a file named `results.txt` with a single float value
                final_eval_path = os.path.join(path, "eval", "1000000_return.npy")

                try:
                    with open(final_eval_path, 'rb') as file:
                        return_values = np.load(file)
                        results[task_name][mismatch_level][approach].append(return_values)
                except Exception as e:
                    print(f"Error reading {final_eval_path}: {e}")
            else:
                print(f"Path {path} does not exist for {approach}")

"""##################################################################################

     Generate the aggregated table

##################################################################################"""

result_table = []
""" Structure the table as (where 1outof5, 3outof5, 5outof5 are the mismatch levels):

task_name, approach, 1outof5, 3outof5, 5outof5
task1, approach1, mean1_1_1 (std1_1_1), mean1_1_3 (std1_1_3), mean1_1_5 (std1_1_5)
task1, approach2, mean1_2_1 (std1_2_1), mean1_2_3 (std1_2_3), mean1_2_5 (std1_2_5)
task2, approach1, mean2_1_1 (std2_1_1), mean2_1_3 (std2_1_3), mean2_1_5 (std2_1_5)
task2, approach2, mean2_2_1 (std2_2_1), mean2_2_3 (std2_2_3), mean2_2_5 (std2_2_5)
...
total, approach1, mean_total_1 (std_total_1), mean_total_3 (std_total_3), mean_total_5 (std_total_5)
total, approach2, mean_total_1 (std_total_1), mean_total_3 (std_total_3), mean_total_5 (std_total_5)
"""
# Aggregate results into result table based on the structure above
for task_name, result_dict in results.items():
    for approach in approaches:
        curr_task_approach_result = {"Task": task_name, "Approach": approach}
        for mismatch_level, approach_dict in result_dict.items():
            if approach_dict[approach]:
                flatten_values = [value for values in approach_dict[approach] for value in values]
                
                mean_val, se_val = mean_and_se(flatten_values)
            else:
                mean_val, se_val = -1, -1
            
            curr_task_approach_result[mismatch_level] = f"{mean_val:.2f} ({se_val:.2f})"

        result_table.append(curr_task_approach_result)

means_plot = {approach: [] for approach in approaches}
ses_plot = {approach: [] for approach in approaches}

for approach in approaches:
    total_task_results = {"Task": "Total", "Approach": approach}

    for mismatch_level in ["1outof5", "3outof5", "5outof5"]:
        all_values = [value for task_values in results.values() for value in task_values[mismatch_level][approach]]

        flatten_all_values = np.concatenate(all_values)

        if all_values:
            mean_val, se_val = mean_and_se(flatten_all_values)
        else:
            mean_val, se_val = -1, -1

        means_plot[approach].append(mean_val)
        ses_plot[approach].append(se_val)

        total_task_results[mismatch_level] = f"{mean_val:.2f} ({se_val:.2f})"

    result_table.append(total_task_results)

# Convert aggregated results to a DataFrame
aggregated_df = pd.DataFrame(result_table)

# Save the aggregated results to a new CSV
output_csv = os.path.join("eval/eval_agg_results", f"metaworld_random_mismatched_agg_result.csv")
aggregated_df.to_csv(output_csv, index=False)

print(f"Aggregated results saved to {output_csv}")

"""##################################################################################

        Plot a bar plot with mean and standard error

##################################################################################"""

from .eval_constants import APPROACH_COLOR_DICT

mismatched_levels = ['1outof5', '3outof5', '5outof5']

x = np.arange(len(mismatched_levels))  # the label locations
width = 0.35  # the width of the bars

plt.grid(True, linestyle='--', alpha=0.3, zorder=0)

# Plotting the bars
plt.bar(x - width/2, means_plot['TemporalOT'], width, yerr=ses_plot['TemporalOT'], label='TemporalOT', capsize=10, zorder=3, color=APPROACH_COLOR_DICT['TemporalOT'])
plt.bar(x + width/2, means_plot['ORCA'], width, yerr=ses_plot['ORCA'], label='ORCA', capsize=10, zorder=3, color=APPROACH_COLOR_DICT['ORCA'])

# Add the mean values on top of the bars
for i, mean_val in enumerate(means_plot['TemporalOT']):
    plt.text(i - width/2, mean_val + 0.05, f"{mean_val:.2f}", ha='center', va='bottom', zorder=4, fontsize=16)
for i, mean_val in enumerate(means_plot['ORCA']):
    plt.text(i + width/2, mean_val + 0.05, f"{mean_val:.2f}", ha='center', va='bottom', zorder=4, fontsize=16)

# Adding labels, title, and legend
plt.xlabel('Mismatch Level', fontsize=20)
plt.ylabel('Cumulative Return', fontsize=20)
# ax.set_title('Total Results for Approaches with Mismatch Levels')
plt.xticks(x, ["Low", "Medium", "High"], fontsize=16)

plt.legend(fontsize=16)

# Display the plot
plt.tight_layout()

# Save the plot
output_plot = os.path.join("eval/eval_agg_results", f"metaworld_random_mismatched_agg_result.png")
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
plt.close()

print(f"Plot saved to {output_plot}")
