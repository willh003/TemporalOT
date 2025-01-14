import numpy as np

def compute_tracking_with_threshold_reward(cost_matrix, threshold=0.9):
    """
    For each subgoal, the reward is exp(-cost)

    We track the current subgoal by comparing the current reward with the threshold. If it's above the threshold, we move to the next subgoal.
    """
    prob_matrix = np.exp(-cost_matrix)
    reward_vector = np.zeros(prob_matrix.shape[0])
    subgoal_tracking_matrix = np.zeros_like(prob_matrix)  # To use the visualization of assignment matrix from other approaches

    curr_subgoal = 0
    total_subgoals = prob_matrix.shape[1]

    for i in range(prob_matrix.shape[0]):
        # 2 components for the reward
        #   - current subgoal reward
        #   - progress reward
        # We then normalize the reward by the total number of subgoals to keep the reward in the range [0, 1]
        reward_vector[i] = (prob_matrix[i, curr_subgoal] + curr_subgoal) / total_subgoals
        subgoal_tracking_matrix[i][curr_subgoal] = 1

        if prob_matrix[i, curr_subgoal] > threshold:
            # Move to the next subgoal until reaching the last subgoal
            curr_subgoal = min(curr_subgoal + 1, prob_matrix.shape[1] - 1)

        # print(f"timestep: {i}; subgoal: {curr_subgoal}/{total_subgoals-1}; reward: {reward_vector[i]}")

    return reward_vector, {"assignment": subgoal_tracking_matrix}