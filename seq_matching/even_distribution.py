import numpy as np

def compute_even_distribution_reward(cost_matrix):
    """
    Compute reward between obs and ref based on an assignment matrix that evenly distributes the frames from obs to ref

    i.e., the first N frames from obs will be distributed to the first frame of ref, and so on, where N is len(obs) // len(ref)
    """
    # Calculate the cost matrix between the reference sequence and the observed sequence
    assignment = identity_like(cost_matrix.shape[0], cost_matrix.shape[1])
    normalized_assignment = assignment / np.expand_dims(np.sum(assignment, axis=1), 1)

    even_distributed_cost = np.sum(normalized_assignment * cost_matrix, axis=1)

    final_reward = - even_distributed_cost

    return final_reward
    
def identity_like(N, M):
    """
    Create an identity matrix of shape (N, M), such that each column has N // M 1s
    And the remainder is distributed as evenly as possible starting from the last column
    """

    # Base number of 1s per column
    k = N // M
    # Remainder to distribute among the first (N % M) columns
    remainder = N % M
    
    # Initialize an (N, M) zero matrix
    matrix = np.zeros((N, M), dtype=int)
    
    # Fill each column with k 1s, plus 1 additional 1 for the first `remainder` columns
    current_row = 0
    for col in range(M):
        num_ones = k + 1 if M - col - 1 < remainder else k
        matrix[current_row:current_row + num_ones, col] = 1
        current_row += num_ones  # Move to the next starting row
    return matrix

