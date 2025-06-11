import math
from typing import Union
from functools import partial
from typing import Optional

import numpy as np
import torch
import ot as pot

min_var_est = 1e-8


def wasserstein_distance_raw(batch_learning, batch_target):
    """
    Calulates the two components of the 2-Wasserstein metric:
    The general formula is given by: d(P_X, P_Y) = min_{X, Y} E[|X-Y|^2]
    For multivariate gaussian distributed inputs z_X ~ MN(mu_X, cov_X) and z_Y ~ MN(mu_Y, cov_Y),
    this reduces to: d = |mu_X - mu_Y|^2 - Tr(cov_X + cov_Y - 2(cov_X * cov_Y)^(1/2))
    Fast method implemented according to following paper: https://arxiv.org/pdf/2009.14075.pdf
    Input shape: [b, n] (e.g. batch_size x num_features)
    Output shape: scalar

    first implementation from https://gist.github.com/Flunzmas/6e359b118b0730ab403753dcc2a447df
    """
    X = batch_target
    Y = batch_learning

    if X.shape != Y.shape:
        raise ValueError("Expecting equal shapes for X and Y!")

    # the linear algebra ops will need some extra precision -> convert to double
    X, Y = X.transpose(0, 1).double(), Y.transpose(0, 1).double()  # [n, b]
    mu_X, mu_Y = torch.mean(X, dim=1, keepdim=True), torch.mean(
        Y, dim=1, keepdim=True
    )  # [n, 1]
    n, b = X.shape
    fact = 1.0 if b < 2 else 1.0 / (b - 1)

    # Cov. Matrix
    E_X = X - mu_X
    E_Y = Y - mu_Y
    cov_X = torch.matmul(E_X, E_X.t()) * fact  # [n, n]
    cov_Y = torch.matmul(E_Y, E_Y.t()) * fact

    # calculate Tr((cov_X * cov_Y)^(1/2)). with the method proposed in https://arxiv.org/pdf/2009.14075.pdf
    # The eigenvalues for M are real-valued.
    C_X = E_X * math.sqrt(fact)  # [n, n], "root" of covariance
    C_Y = E_Y * math.sqrt(fact)
    M_l = torch.matmul(C_X.t(), C_Y)
    M_r = torch.matmul(C_Y.t(), C_X)
    M = torch.matmul(M_l, M_r)
    S = (
        torch.linalg.eigvals(M) + 1e-15
    )  # add small constant to avoid infinite gradients from sqrt(0)
    sq_tr_cov = S.sqrt().abs().sum()

    # plug the sqrt_trace_component into Tr(cov_X + cov_Y - 2(cov_X * cov_Y)^(1/2))
    trace_term = torch.trace(cov_X + cov_Y) - 2.0 * sq_tr_cov  # scalar

    # |mu_X - mu_Y|^2
    diff = mu_X - mu_Y  # [n, 1]
    mean_term = torch.sum(torch.mul(diff, diff))  # scalar

    # put it together
    return (trace_term + mean_term).float()


def wasserstein_distance_raw_avg(batch_learning, batch_target):
    n_func_samples = batch_target.shape[
        -1
    ]  # the raw value grows proportionally to number of sampling pts
    return wasserstein_distance_raw(batch_learning, batch_target) / n_func_samples


def wasserstein(
        x0: torch.Tensor,
        x1: torch.Tensor,
        method: Optional[str] = None,
        reg: float = 0.05,
        power: int = 2,
        **kwargs,
) -> float:
    assert power == 1 or power == 2
    # ot_fn should take (a, b, M) as arguments where a, b are marginals and
    # M is a cost matrix
    if method == "exact" or method is None:
        ot_fn = pot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(pot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)
    x1 = x1.to(x0.device, x0.dtype)
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M ** 2
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=1e7)
    if power == 2:
        ret = math.sqrt(ret)
    return ret


# Consider linear time MMD with a linear kernel:
# K(f(x), f(y)) = f(x)^Tf(y)
# h(z_i, z_j) = k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(x_j, y_i)
#             = [f(x_i) - f(y_i)]^T[f(x_j) - f(y_j)]
#
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def linear_mmd2(f_of_X, f_of_Y):
    loss = 0.0
    delta = f_of_X - f_of_Y
    loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
    return loss


# Consider linear time MMD with a polynomial kernel:
# K(f(x), f(y)) = (alpha*f(x)^Tf(y) + c)^d
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def poly_mmd2(f_of_X, f_of_Y, d=2, alpha=1.0, c=2.0):
    K_XX = alpha * (f_of_X[:-1] * f_of_X[1:]).sum(1) + c
    K_XX_mean = torch.mean(K_XX.pow(d))

    K_YY = alpha * (f_of_Y[:-1] * f_of_Y[1:]).sum(1) + c
    K_YY_mean = torch.mean(K_YY.pow(d))

    K_XY = alpha * (f_of_X[:-1] * f_of_Y[1:]).sum(1) + c
    K_XY_mean = torch.mean(K_XY.pow(d))

    K_YX = alpha * (f_of_Y[:-1] * f_of_X[1:]).sum(1) + c
    K_YX_mean = torch.mean(K_YX.pow(d))

    return K_XX_mean + K_YY_mean - K_XY_mean - K_YX_mean


def _mix_rbf_kernel(X, Y, sigma_list):
    assert X.size(0) == Y.size(0)
    m = X.size(0)

    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma ** 2)
        K += torch.exp(-gamma * exponent)

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)


def mix_rbf_mmd2(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


def mix_rbf_mmd2_and_ratio(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


################################################################################
# Helper functions to compute variances based on kernel matrices
################################################################################


def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)  # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)  # (m,)
        diag_Y = torch.diag(K_YY)  # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X  # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y  # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)  # K_{XY}^T * e

    Kt_XX_sum = Kt_XX_sums.sum()  # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()  # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()  # e^T * K_{XY} * e

    if biased:
        mmd2 = (
                (Kt_XX_sum + sum_diag_X) / (m * m)
                + (Kt_YY_sum + sum_diag_Y) / (m * m)
                - 2.0 * K_XY_sum / (m * m)
        )
    else:
        mmd2 = (
                Kt_XX_sum / (m * (m - 1))
                + Kt_YY_sum / (m * (m - 1))
                - 2.0 * K_XY_sum / (m * m)
        )

    return mmd2


def _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    mmd2, var_est = _mmd2_and_variance(
        K_XX, K_XY, K_YY, const_diagonal=const_diagonal, biased=biased
    )
    loss = mmd2 / torch.sqrt(torch.clamp(var_est, min=min_var_est))
    return loss, mmd2, var_est


def _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)  # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
        sum_diag2_X = sum_diag2_Y = m * const_diagonal ** 2
    else:
        diag_X = torch.diag(K_XX)  # (m,)
        diag_Y = torch.diag(K_YY)  # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)
        sum_diag2_X = diag_X.dot(diag_X)
        sum_diag2_Y = diag_Y.dot(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X  # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y  # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)  # K_{XY}^T * e
    K_XY_sums_1 = K_XY.sum(dim=1)  # K_{XY} * e

    Kt_XX_sum = Kt_XX_sums.sum()  # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()  # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()  # e^T * K_{XY} * e

    Kt_XX_2_sum = (K_XX ** 2).sum() - sum_diag2_X  # \| \tilde{K}_XX \|_F^2
    Kt_YY_2_sum = (K_YY ** 2).sum() - sum_diag2_Y  # \| \tilde{K}_YY \|_F^2
    K_XY_2_sum = (K_XY ** 2).sum()  # \| K_{XY} \|_F^2

    if biased:
        mmd2 = (
                (Kt_XX_sum + sum_diag_X) / (m * m)
                + (Kt_YY_sum + sum_diag_Y) / (m * m)
                - 2.0 * K_XY_sum / (m * m)
        )
    else:
        mmd2 = (
                Kt_XX_sum / (m * (m - 1))
                + Kt_YY_sum / (m * (m - 1))
                - 2.0 * K_XY_sum / (m * m)
        )

    var_est = (
            2.0
            / (m ** 2 * (m - 1.0) ** 2)
            * (
                    2 * Kt_XX_sums.dot(Kt_XX_sums)
                    - Kt_XX_2_sum
                    + 2 * Kt_YY_sums.dot(Kt_YY_sums)
                    - Kt_YY_2_sum
            )
            - (4.0 * m - 6.0)
            / (m ** 3 * (m - 1.0) ** 3)
            * (Kt_XX_sum ** 2 + Kt_YY_sum ** 2)
            + 4.0
            * (m - 2.0)
            / (m ** 3 * (m - 1.0) ** 2)
            * (K_XY_sums_1.dot(K_XY_sums_1) + K_XY_sums_0.dot(K_XY_sums_0))
            - 4.0 * (m - 3.0) / (m ** 3 * (m - 1.0) ** 2) * (K_XY_2_sum)
            - (8 * m - 12) / (m ** 5 * (m - 1)) * K_XY_sum ** 2
            + 8.0
            / (m ** 3 * (m - 1.0))
            * (
                    1.0 / m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
                    - Kt_XX_sums.dot(K_XY_sums_1)
                    - Kt_YY_sums.dot(K_XY_sums_0)
            )
    )
    return mmd2, var_est


def compute_distances(pred, true):
    """computes distances between vectors."""
    mse = torch.nn.functional.mse_loss(pred, true).item()
    me = math.sqrt(mse)
    mae = torch.mean(torch.abs(pred - true)).item()
    return mse, me, mae


def compute_distribution_distances(pred: torch.Tensor, true: Union[torch.Tensor, list]):
    """computes distances between distributions.
    pred: [batch, times, dims] tensor
    true: [batch, times, dims] tensor or list[batch[i], dims] of length times

    This handles jagged times as a list of tensors.
    """
    assert pred.shape == true.shape, "Predictions and ground truth must have the same dimensions!"
    
    NAMES = [
        "Wasserstein_RAW",
        "Wasserstein_RAW_AVG",
        "1-Wasserstein",
        "2-Wasserstein",
        "Linear_MMD",
        "Poly_MMD",
        "RBF_MMD",
        "Mean_MSE",
        "Mean_L2",
        "Mean_L1",
        "Median_MSE",
        "Median_L2",
        "Median_L1",
    ]
    is_jagged = isinstance(true, list)
    dists = []
    to_return = []
    names = []
    filtered_names = [
        name for name in NAMES if not is_jagged or not name.endswith("MMD")
    ]

    a = pred
    b = true

    w_raw = wasserstein_distance_raw(b, a)  # note order of arguments!
    w_raw_avg = wasserstein_distance_raw_avg(b, a)  # note order of arguments!
    w1 = wasserstein(a, b, power=1)
    w2 = wasserstein(a, b, power=2)
    mmd_linear = linear_mmd2(a, b).item()
    mmd_poly = poly_mmd2(a, b, d=2, alpha=1.0, c=2.0).item()
    mmd_rbf = mix_rbf_mmd2(a, b, sigma_list=[0.01, 0.1, 1, 10, 100]).item()
    mean_dists = compute_distances(torch.mean(a, dim=0), torch.mean(b, dim=0))
    median_dists = compute_distances(
        torch.median(a, dim=0)[0], torch.median(b, dim=0)[0]
    )

    dists.append(
        (w_raw, w_raw_avg, w1, w2, mmd_linear, mmd_poly, mmd_rbf, *mean_dists, *median_dists)
    )

    to_return.extend(np.array(dists).mean(axis=0))
    names.extend(filtered_names)
    metrics = dict()
    for name, to_return in zip(names, to_return):
        metrics[name] = to_return
    return metrics