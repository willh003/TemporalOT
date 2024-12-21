import numpy as np

def bordered_identity_like(N, M, k):
    """
    Create an identity-like matrix of shape (N, M), such that each column has N // M 1s,
    the remainder is distributed as evenly as possible starting from the last column,
    and a border of width k is added on each side of the ones
    """
    k = int(k)

    # Base number of 1s per column
    base_ones = N // M
    # Remainder to distribute among the first (N % M) columns
    remainder = N % M

    # Initialize an (N, M) zero matrix
    matrix = np.zeros((N, M), dtype=np.float32)

    # Fill each column with `base_ones` 1s, plus 1 additional 1 for the first `remainder` columns
    current_row = 0
    for col in range(M):
        num_ones = base_ones + 1 if M - col - 1 < remainder else base_ones
        matrix[current_row:current_row + num_ones, col] = 1
        current_row += num_ones  # Move to the next starting row

    # Create the border by adding k ones to the left and right of each row's 1s
    bordered_matrix = np.zeros_like(matrix)

    for row in range(N):
        for col in range(M):
            if matrix[row, col] == 1:
                start_col  = max(0, col - k)
                end_col = min(N, col + k + 1)
                bordered_matrix[row, start_col:end_col] = 1

    return bordered_matrix

    
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

