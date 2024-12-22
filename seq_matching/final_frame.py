import numpy as np

def compute_final_frame_reward(cost_matrix):
    """
    Reward = -d(obs, ref[-1])
    Just distance from final reference state (ignore the sequence)
    """
    assignment = np.zeros_like(cost_matrix)
    assignment[:, -1] = 1

    final_reward = - np.sum(cost_matrix * assignment, axis=1)  # size: (train_freq,)

    return final_reward, assignment