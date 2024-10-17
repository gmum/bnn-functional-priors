from typing import NamedTuple

import torch
import gpytorch

from activations.gp_activations import StepActivation, sigma
from targets import SingleHiddenLayerWideNNWithGaussianPrior
from targets_gp import GPGenerator1D, AllYouNeed1DRegression, AllYouNeedUCIRegression


class _BNNMetadata(NamedTuple):
    ma: float = 0.0
    mu: float = 0.0
    mb: float = 0.0
    mv: float = 0.0
    sa: float = 1.0
    su: float = 1.0
    sb: float = 1.0
    wv: float = 0.001
    activation: callable = lambda x: x * 0.0  # dummy activation


def get_bnn_configs(config_set_no=1, net_width=1_000):
    if config_set_no == 1:
        # Config set 1: See Priors for Infinite Networks (Neal 94'): Figure 2: left
        activation = torch.nn.Tanh()
        ma = mu = mb = mv = 0.0
        sa = su = 5.0
        sb = wv = 1.0
        data_min_x, data_max_x = -1.0, 1.0
    elif config_set_no == 2:
        # Config set 2: See Priors for Infinite Networks (Neal 94'): Figure 1
        activation = StepActivation()
        ma = mu = mb = mv = 0.0
        sa = su = sb = wv = 1.0
        data_min_x, data_max_x = -1.0, 1.0
    elif config_set_no == 3:
        # Config set 3: some random set of parameters, without any reasoning behind
        activation = sigma
        ma = mu = mb = mv = 0
        sa = su = 10.0
        sb = wv = 3.0
        data_min_x, data_max_x = -10.0, 10.0
    else:
        raise ValueError(f"Not supported config_set_no={config_set_no}!")

    generator = SingleHiddenLayerWideNNWithGaussianPrior(
        ma=ma,
        mu=mu,
        mb=mb,
        mv=mv,
        sa=sa,
        su=su,
        sb=sb,
        wv=wv,
        activation=activation,
        width=net_width,
    )
    meta = _BNNMetadata(
        ma=ma, mu=mu, mb=mb, mv=mv, sa=sa, su=su, sb=sb, wv=wv, activation=activation
    )

    return generator, meta, data_min_x, data_max_x


def get_gp_configs(config_set_no=-1, lengthscale_precision=1e-7):
    meta = _BNNMetadata()  # use the default dummy configuration
    data_min_x, data_max_x = -3.0, 3.0

    if (
        config_set_no == -111
    ):  # matching setting for banana classification in Stationary Activations ...
        data_min_x, data_max_x = -3.75, 3.75
        kernel = gpytorch.kernels.MaternKernel(
            nu=5 / 2,
            lengthscale_constraint=gpytorch.constraints.Interval(
                1.0 - lengthscale_precision / 2, 1.0 + lengthscale_precision / 2
            ),
        )

    elif config_set_no == -1:
        print("WARNING: Using Matern 5/2 kernel but with lengthscale~=0.7!")
        kernel = gpytorch.kernels.MaternKernel(nu=5 / 2)

    elif config_set_no == -2:
        kernel = gpytorch.kernels.MaternKernel(nu=3 / 2)
        print("WARNING: Using Matern 3/2 kernel but with lengthscale~=0.7!")

    elif config_set_no == -3:
        kernel = gpytorch.kernels.MaternKernel(nu=1 / 2)
        print("WARNING: Using Matern 1/2 kernel but with lengthscale~=0.7!")

    elif config_set_no == -11:
        kernel = gpytorch.kernels.MaternKernel(
            nu=5 / 2,
            lengthscale_constraint=gpytorch.constraints.Interval(
                1.0 - lengthscale_precision / 2, 1.0 + lengthscale_precision / 2
            ),
        )

    elif config_set_no == -12:
        kernel = gpytorch.kernels.MaternKernel(
            nu=3 / 2,
            lengthscale_constraint=gpytorch.constraints.Interval(
                1.0 - lengthscale_precision / 2, 1.0 + lengthscale_precision / 2
            ),
        )

    elif config_set_no == -13:
        kernel = gpytorch.kernels.MaternKernel(
            nu=1 / 2,
            lengthscale_constraint=gpytorch.constraints.Interval(
                1.0 - lengthscale_precision / 2, 1.0 + lengthscale_precision / 2
            ),
        )

    elif config_set_no == -4:
        kernel = gpytorch.kernels.RBFKernel()
        # kernel.lengthscale = 1.

    elif config_set_no == -5:
        kernel = gpytorch.kernels.CosineKernel()
        # kernel.period_length=torch.tensor(3)

    else:
        raise ValueError(f"Not supported GP configuration no = {config_set_no}!")

    generator = GPGenerator1D(kernel=kernel)

    return generator, meta, data_min_x, data_max_x


def get_configs(config_set_no=1, net_width=1_000, input_dim=None, sn2=None):
    if config_set_no == 1001:
        generator = AllYouNeed1DRegression()
        meta = _BNNMetadata()  # use the default dummy configuration
        # data_min_x, data_max_x = -15, 15

        # # in the original code for All You Need they rescale the ranges
        # mean, std, eps = 3.89738198, 4.53119575, 1e-10
        # normalize = lambda X: (X - mean) / (std + eps)
        # data_min_x, data_max_x = normalize(data_min_x), normalize(data_max_x)

        data_min_x, data_max_x = -6.0, 6.0

        return generator, meta, data_min_x, data_max_x

    elif config_set_no >= 2000:
        generator = AllYouNeedUCIRegression(input_dim=input_dim, sn2=sn2)
        meta = _BNNMetadata()

        # TODO - check (here might be anything and it doesn't matter I guess)
        data_min_x, data_max_x = -6.0, 6.0

        return generator, meta, data_min_x, data_max_x

    elif config_set_no < 0:
        return get_gp_configs(config_set_no=config_set_no)

    else:
        return get_bnn_configs(config_set_no=config_set_no, net_width=net_width)
