import numpy as np
from .utils import bordered_identity_like

def compute_even_distribution_reward(cost_matrix, mask_k: int=10):
    """
    Compute reward between obs and ref based on an assignment matrix that evenly distributes the frames from obs to ref, with an additional border on each side of size mask_k
    i.e., the first N frames from obs will be distributed to the first frame of ref, and so on, where N is len(obs) // len(ref)
    
    if mask_k == 0 and cost_matrix is square, then this is the identity
    """
    # Calculate the cost matrix between the reference sequence and the observed sequence
    assignment = bordered_identity_like(cost_matrix.shape[0], cost_matrix.shape[1], mask_k)
    normalized_assignment = assignment / np.expand_dims(np.sum(assignment, axis=1), 1)

    even_distributed_cost = np.sum(normalized_assignment * cost_matrix, axis=1)

    final_reward = - even_distributed_cost

    return final_reward, {"assignment": normalized_assignment}
