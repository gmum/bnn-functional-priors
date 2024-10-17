import torch
import math
import function_stats


def mse(batch_target, batch_learning):
    return ((batch_learning - batch_target) ** 2).mean()


def wasserstein_distance_raw(batch_target, batch_learning):
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


def wasserstein_distance(batch_target, batch_learning):
    n_func_samples = batch_target.shape[
        -1
    ]  # the raw value grows proportionally to number of sampling pts
    return wasserstein_distance_raw(batch_target, batch_learning) / n_func_samples


def match_quantiles(batch_target, batch_learning):
    return torch.mean(
        (
            torch.quantile(batch_target, 0.2, dim=0)
            - torch.quantile(batch_learning, 0.2, dim=0)
        )
        ** 2
    ) + torch.mean(
        (
            torch.quantile(batch_target, 0.8, dim=0)
            - torch.quantile(batch_learning, 0.8, dim=0)
        )
        ** 2
    )


def create_sinkhorn_distance():
    """Implementation from http://www.kernel-operations.io/geomloss/index.html
    Installation by # !pip install geomloss[full]
    """
    from geomloss import SamplesLoss

    sinkhorn_loss = SamplesLoss("sinkhorn", p=2, blur=0.05)

    def sinkhorn_distance(batch_target, batch_learning):
        return sinkhorn_loss(batch_target, batch_learning)

    return sinkhorn_distance


def match_variances(batch_target, batch_learning):
    # return (torch.abs((torch.var(batch_target, dim=1) - torch.var(batch_learning, dim=1)))).mean()
    # return torch.abs(torch.sort(torch.var(batch_target, dim=1)) - torch.sort(torch.var(batch_learning, dim=1))).mean()
    return (
        torch.var(batch_target, dim=1).mean() - torch.var(batch_learning, dim=1).mean()
    ) ** 2


def wasserstein_distance_regularized(
    batch_target, batch_learning, M1=1.0, M2=1.0, M3=1.0
):
    variances_loss = (
        torch.var(batch_target, dim=1).mean() - torch.var(batch_learning, dim=1).mean()
    ) ** 2
    kurtosis_loss = (
        function_stats.compute_kurtosis(batch_target).mean()
        - function_stats.compute_kurtosis(batch_learning).mean()
    ) ** 2
    skewness_loss = (
        function_stats.compute_skewness(batch_target).mean()
        - function_stats.compute_skewness(batch_learning).mean()
    ) ** 2

    return wasserstein_distance(batch_target, batch_learning) + (
        M1 * variances_loss + M2 * kurtosis_loss + M3 * skewness_loss
    )


def wasserstein_distance_strongly_regularized(batch_target, batch_learning):
    return wasserstein_distance_regularized(
        batch_target, batch_learning, M1=1000.0, M2=1000.0, M3=1000.0
    )
