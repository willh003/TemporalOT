import numpy as np
import torch
import ot
import numpy as np
from scipy.special import logsumexp

def compute_temporal_ot_reward(cost_matrix,
                                mask_k: int = 2,
                                niter: int = 100,
                                ent_reg: float = 0.01):

    # optimal weights 
    mask = bordered_identity_like(cost_matrix.shape[0], cost_matrix.shape[1], k=mask_k)
    transport_plan = mask_optimal_transport_plan(cost_matrix,
                                                mask,
                                                niter,
                                                ent_reg)
    

    ot_cost = np.sum(transport_plan * cost_matrix, axis=1)
    ot_reward = -ot_cost

    return ot_cost

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
    matrix = np.zeros((N, M), dtype=int)

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


def mask_sinkhorn(a, b, M, Mask, reg=0.01, numItermax=1000, stopThr=1e-9):
    # set a large value (1e6) for masked entry
    Mr = -M/reg*Mask + (-1e6)*(1-Mask)
    loga = np.log(a)
    logb = np.log(b)

    u = np.zeros(len(a))
    v = np.zeros(len(b))
    err = 1

    for i in range(numItermax):
        v = logb - logsumexp(Mr + u[:, None], 0)
        u = loga - logsumexp(Mr + v[None, :], 1)
        if i % 10 == 0:
            tmp_pi = np.exp(Mr + u[:, None] + v[None, :])
            err = np.linalg.norm(tmp_pi.sum(0) - b)
            if err < stopThr:
                return tmp_pi

    pi = np.exp(Mr + u[:, None] + v[None, :])
    return pi


def sinkhorn_log(a, b, M, reg=0.01, numItermax=1000, stopThr=1e-9):
    Mr = -M / reg
    loga = np.log(a)
    logb = np.log(b)

    u = np.zeros(len(a))
    v = np.zeros(len(b))
    err = 1

    for i in range(numItermax):
        v = logb - logsumexp(Mr + u[:, None], 0)
        u = loga - logsumexp(Mr + v[None, :], 1)
        if i % 10 == 0:
            tmp_pi = np.exp(Mr + u[:, None] + v[None, :])
            err = np.linalg.norm(tmp_pi.sum(0) - b)
            if err < stopThr:
                return tmp_pi

    pi = np.exp(Mr + u[:, None] + v[None, :])
    return pi


def mask_optimal_transport_plan(cost_matrix,
                                Mask,
                                niter=100,
                                ent_reg=0.01,
                                device='cuda'):
    X_pot = np.ones(cost_matrix.shape[0]) / cost_matrix.shape[0]
    Y_pot = np.ones(cost_matrix.shape[1]) / cost_matrix.shape[1]

    transport_plan = mask_sinkhorn(X_pot,
                                   Y_pot,
                                   cost_matrix,
                                   Mask,
                                   ent_reg,
                                   numItermax=niter)

    return transport_plan


def optimal_transport_plan(X,
                           Y,
                           cost_matrix,
                           method="sinkhorn_gpu",
                           niter=500,
                           use_log=False,
                           ent_reg=0.01):
    X_pot = np.ones(X.shape[0]) / X.shape[0]
    Y_pot = np.ones(Y.shape[0]) / Y.shape[0]
    c_m = cost_matrix.data.detach().cpu().numpy()
    if use_log:
        transport_plan = sinkhorn_log(X_pot,
                                      Y_pot,
                                      c_m,
                                      ent_reg,
                                      numItermax=niter)
    else:
        transport_plan = ot.sinkhorn(X_pot,
                                     Y_pot,
                                     c_m,
                                     ent_reg,
                                     numItermax=niter)
    transport_plan = torch.from_numpy(transport_plan).to(X.device)
    transport_plan.requires_grad = False
    return transport_plan.float()
