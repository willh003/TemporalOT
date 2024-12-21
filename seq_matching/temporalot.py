import numpy as np
import torch
import ot
from .utils import bordered_identity_like
from scipy.special import logsumexp

def compute_temporal_ot_reward(cost_matrix,
                                mask_k: int = 10,
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
    return ot_reward

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
