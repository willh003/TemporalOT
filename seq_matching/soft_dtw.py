from tslearn.metrics import SoftDTW

import numpy as np

def compute_soft_dtw_reward(cost_matrix, smoothing=1) -> np.ndarray:
    assert smoothing > 0, "Currently not supporting reg == 0"

    # Calculate the cost matrix between the reference sequence and the observed sequence
    #   size: (train_freq, ref_seq_len)
    sdtw = SoftDTW(cost_matrix, gamma=smoothing)
    dist_sq = sdtw.compute()  # We don't actually use this
    a = sdtw.grad()

    normalized_a = a / np.expand_dims(np.sum(a, axis=1), 1)
    soft_dtw_cost = np.sum(cost_matrix * normalized_a, axis=1)  # size: (train_freq,)

    final_reward = - soft_dtw_cost
    return final_reward