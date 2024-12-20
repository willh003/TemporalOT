import numpy as np
import ot

def compute_ot_reward(cost_matrix, ent_reg=.01) -> np.ndarray:
    """
    Compute the Optimal Transport (OT) reward between the reference sequence and the observed sequence

    Parameters:
        obs: np.ndarray
            The observed sequence of joint states
            size: (train_freq, 22)
                For OT-based reward, train_freq == episode_length
                22 is the observation size that we want to calculate
        ref: np.ndarray
            The reference sequence of joint states
            size: (ref_seq_len, 22)
                22 is the observation size that we want to calculate
        cost_fn: function
            Options: cosine_distance, euclidean_distance
        scale: float
            The scaling factor for the OT reward
    """
    # Calculate the cost matrix between the reference sequence and the observed sequence
    #   size: (train_freq, ref_seq_len)
    
    # Calculate the OT plan between the reference sequence and the observed sequence
    obs_weight = np.ones(cost_matrix.shape[0]) / cost_matrix.shape[0]
    ref_weight = np.ones(cost_matrix.shape[1]) / cost_matrix.shape[1]

    if ent_reg == 0:
        T = ot.emd(obs_weight, ref_weight, cost_matrix)  # size: (train_freq, ref_seq_len)
    else:
        T = ot.sinkhorn(obs_weight, ref_weight, cost_matrix, reg=ent_reg, log=False)  # size: (train_freq, ref_seq_len)

    # Normalize the path so that each row sums to 1
    normalized_T = T / np.expand_dims(np.sum(T, axis=1), 1)

    # Calculate the OT cost for each timestep
    #   sum by row of (cost matrix * OT plan)
    ot_cost = np.sum(cost_matrix * normalized_T, axis=1)  # size: (train_freq,)

    final_reward = -ot_cost
    return final_reward