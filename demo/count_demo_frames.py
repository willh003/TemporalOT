import numpy as np
import json

medium_frame_indices = {
    "window-open-v2": list(range(5)) + list(range(26, 38)) + [49, 50, 51], 
    "button-press-v2": list(range(15)) + [31, 43, 44],
    "door-close-v2": list(range(15)) + [31, 43, 44],
    "door-open-v2":  list(range(15)) + [31, 32, 55, 56],
    "stick-push-v2": list(range(10, 25)) + [49, 50, 51],
    "push-v2": list(range(10, 25)) + [49, 50, 51],
    "door-lock-v2": list(range(20)) + [28,29,30] + [61, 62],
    "lever-pull-v2": list(range(5, 18)) + [31, 32, 55, 56, 60, 61],
    "hand-insert-v2": list(range(5, 18)) + [21, 22, 31, 32, 55, 56],
    "basketball-v2": list(range(7, 20)) + [25, 26, 31, 32, 55, 56]
}

# Compute the average number of frames in the medium_frame_indices
demo_len_list = [len(medium_frame_indices[task_name]) for task_name in medium_frame_indices.keys()]
print(f"Mismatched Experiments Average Number of Frames: {np.mean(demo_len_list)}")
print("==================")


mismatch_level = [1, 3, 5]
tasks = ["door-open-v2", "window-open-v2", "lever-pull-v2"]

for speed in ["fast", "slow"]:
    for level in mismatch_level:
        level_demo_len_list = []
        level_demo_std_list = []
        level_demo_cv_list = []
        level_demo_avg_diff_list = []
        for task_name in tasks:
            for i in range(3):
                with open(f"/share/portal/wph52/TemporalOT/create_demo/metaworld_demos/{task_name}/random_mismatched_{speed}/{level}outof5_mismatched/{level}outof5_mismatched_{i}/{task_name}_corner3_0_mismatched_info.json") as f:
                    info = json.load(f)
                    subsection_lens = [len(info[subsection]["subsampled_indices"]) for subsection in info.keys()]
                    demo_len = np.sum(subsection_lens)
                    demo_std = np.std(subsection_lens)
                    level_demo_len_list.append(demo_len)
                    level_demo_std_list.append(demo_std)
                    level_demo_cv_list.append(demo_std/np.mean(subsection_lens)*100)

                    diff = 0
                    for i in range(len(subsection_lens)-1):
                        for j in range(i+1, len(subsection_lens)):
                            diff += abs(subsection_lens[i] - subsection_lens[j])
                    diff = diff / np.sum(subsection_lens) / 10.0
                    level_demo_avg_diff_list.append(diff)
        
        print(f"speed: {speed}, level: {level}, Average Number of Frames: {np.mean(level_demo_len_list)}, avg std: {np.mean(level_demo_std_list)}, avg cv: {np.mean(level_demo_cv_list)}, avg diff: {np.mean(level_demo_avg_diff_list)}")
        input("stop")