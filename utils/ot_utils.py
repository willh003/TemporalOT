import numpy as np
import torch
import ot
import numpy as np
from scipy.special import logsumexp


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


def mask_optimal_transport_plan(X,
                                Y,
                                cost_matrix,
                                Mask,
                                niter=100,
                                epsilon=0.01):
    X_pot = np.ones(X.shape[0]) / X.shape[0]
    Y_pot = np.ones(Y.shape[0]) / Y.shape[0]
    c_m = cost_matrix.data.detach().cpu().numpy()
    transport_plan = mask_sinkhorn(X_pot,
                                   Y_pot,
                                   c_m,
                                   Mask,
                                   epsilon,
                                   numItermax=niter)
    transport_plan = torch.from_numpy(transport_plan).to(X.device)
    transport_plan.requires_grad = False
    return transport_plan.float()


def optimal_transport_plan(X,
                           Y,
                           cost_matrix,
                           method="sinkhorn_gpu",
                           niter=500,
                           use_log=False,
                           epsilon=0.01):
    X_pot = np.ones(X.shape[0]) / X.shape[0]
    Y_pot = np.ones(Y.shape[0]) / Y.shape[0]
    c_m = cost_matrix.data.detach().cpu().numpy()
    if use_log:
        transport_plan = sinkhorn_log(X_pot,
                                      Y_pot,
                                      c_m,
                                      epsilon,
                                      numItermax=niter)
    else:
        transport_plan = ot.sinkhorn(X_pot,
                                     Y_pot,
                                     c_m,
                                     epsilon,
                                     numItermax=niter)
    transport_plan = torch.from_numpy(transport_plan).to(X.device)
    transport_plan.requires_grad = False
    return transport_plan.float()


def cosine_distance(x, y):
    C = torch.mm(x, y.T)
    x_norm = torch.norm(x, p=2, dim=1)
    y_norm = torch.norm(y, p=2, dim=1)
    x_n = x_norm.unsqueeze(1)
    y_n = y_norm.unsqueeze(1)
    norms = torch.mm(x_n, y_n.T)
    C = (1 - C / norms)
    return C


def euclidean_distance(x, y):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    c = torch.sqrt(torch.sum((torch.abs(x_col - y_lin))**2, 2))
    return c
