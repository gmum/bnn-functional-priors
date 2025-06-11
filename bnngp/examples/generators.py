import torch
import gpytorch

import sys

sys.path.append("../")

from activations.gp_activations import StepActivation, sigma
from targets import SingleHiddenLayerWideNNWithGaussianPrior
from targets_gp import GPGenerator1D, AllYouNeed1DRegression, AllYouNeedUCIRegression


def get_single_layer_bnn_params(config_set_no):
    """
    Returns parameters for Guassian priors of a single-layer Bayesian neural network.
    """
    if config_set_no == 1:
        # Config set 1: See Priors for Infinite Networks (Neal 94'): Figure 2: left
        activation = torch.nn.Tanh()
        ma = mu = mb = mv = 0.0
        sa = su = 5.0
        sb = wv = 1.0
    elif config_set_no == 2:
        # Config set 2: See Priors for Infinite Networks (Neal 94'): Figure 1
        activation = StepActivation()
        ma = mu = mb = mv = 0.0
        sa = su = sb = wv = 1.0
    elif config_set_no == 3:
        # Config set 3: some random set of parameters, without any reasoning behind
        activation = sigma
        ma = mu = mb = mv = 0
        sa = su = 10.0
        sb = wv = 3.0
    else:
        raise ValueError(f"Not supported config_set_no={config_set_no}!")
    return activation, mv, mb, mu, ma, su, sa, wv, sb


def get_single_layer_bnn_generator(config_set_no=1, net_width=1_000):
    activation, mv, mb, mu, ma, su, sa, wv, sb = get_single_layer_bnn_params(
        config_set_no
    )

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
    return generator


def get_gp_generator(config_set_no=-1, lengthscale_precision=1e-7):

    if (
        config_set_no == -111
    ):  # matching setting for banana classification in Stationary Activations ...
        kernel = gpytorch.kernels.MaternKernel(
            nu=5 / 2,
            lengthscale_constraint=gpytorch.constraints.Interval(
                1.0 - lengthscale_precision / 2, 1.0 + lengthscale_precision / 2
            ),
        )
        kernel.lengthscale = 1.0

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
        kernel.lengthscale = 1.0

    elif config_set_no == -12:
        kernel = gpytorch.kernels.MaternKernel(
            nu=3 / 2,
            lengthscale_constraint=gpytorch.constraints.Interval(
                1.0 - lengthscale_precision / 2, 1.0 + lengthscale_precision / 2
            ),
        )
        kernel.lengthscale = 1.0

    elif config_set_no == -21:
        kernel = gpytorch.kernels.MaternKernel(
            nu=3 / 2,
            lengthscale_constraint=gpytorch.constraints.Interval(
                0.01 - lengthscale_precision / 2, 0.01 + lengthscale_precision / 2
            ),
        )
        kernel.lengthscale = 0.01

    elif config_set_no == -22:
        kernel = gpytorch.kernels.MaternKernel(
            nu=3 / 2,
            lengthscale_constraint=gpytorch.constraints.Interval(
                0.1 - lengthscale_precision / 2, 0.1 + lengthscale_precision / 2
            ),
        )
        kernel.lengthscale = 0.1

    elif config_set_no == -23:
        kernel = gpytorch.kernels.MaternKernel(
            nu=3 / 2,
            lengthscale_constraint=gpytorch.constraints.Interval(
                10.0 - 0.001 / 2, 10.0 + 0.001 / 2
            ),
        )
        kernel.lengthscale = 10.0

    elif config_set_no == -24:
        kernel = gpytorch.kernels.MaternKernel(
            nu=3 / 2,
            lengthscale_constraint=gpytorch.constraints.Interval(
                100.0 - 0.001 / 2, 100.0 + 0.001 / 2
            ),
        )
        kernel.lengthscale = 100.0

    elif config_set_no == -13:
        kernel = gpytorch.kernels.MaternKernel(
            nu=1 / 2,
            lengthscale_constraint=gpytorch.constraints.Interval(
                1.0 - lengthscale_precision / 2, 1.0 + lengthscale_precision / 2
            ),
        )
        kernel.lengthscale = 1.0

    elif config_set_no == -4:
        kernel = gpytorch.kernels.RBFKernel()
        # kernel.lengthscale = 1.

    elif config_set_no == -5:
        kernel = gpytorch.kernels.CosineKernel()
        # kernel.period_length=torch.tensor(3)

    elif config_set_no == -6:
        print("Creating periodic kernel")
        kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())

    elif config_set_no == -61:
        print("Creating periodic kernel")
        kernel = gpytorch.kernels.PeriodicKernel()
        kernel.period_length = 1.0
        kernel.lengthscale = 1.0

    elif config_set_no == -7:
        print("Creating periodic+rbf kernel")
        rbf_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        periodic_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.PeriodicKernel()
        )
        kernel = periodic_kernel + rbf_kernel

    elif config_set_no == -8:
        print(f"Creating periodic kernel")
        kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())

    elif config_set_no == -9:
        print(f"Creating periodic+rbf kernel")
        rbf_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        periodic_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.PeriodicKernel()
        )
        kernel = periodic_kernel + rbf_kernel

    elif config_set_no == -81:
        print("Creating periodic kernel")
        kernel = gpytorch.kernels.PeriodicKernel()
        kernel.period_length = 1.0
        kernel.lengthscale = 1.0

    elif config_set_no == -91:
        print("Creating periodic+rbf kernel")
        rbf_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        periodic_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.PeriodicKernel()
        )
        kernel = periodic_kernel + rbf_kernel

    else:
        raise ValueError(f"Not supported GP configuration no = {config_set_no}!")

    generator = GPGenerator1D(kernel=kernel)

    return generator


def get_generator(config_set_no=1, net_width=1_000, input_dim=None, sn2=None):
    if config_set_no == 1001:
        return AllYouNeed1DRegression()

    elif config_set_no >= 2000:
        return AllYouNeedUCIRegression(input_dim=input_dim, sn2=sn2)

    elif config_set_no < 0:
        return get_gp_generator(config_set_no=config_set_no)

    else:
        return get_single_layer_bnn_generator(
            config_set_no=config_set_no, net_width=net_width
        )


if __name__ == "__main__":

    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    for config_set_no in [-81]: #, -7, -8, -9, -81, -91]:

        generator = get_gp_generator(config_set_no=config_set_no)
        print(generator.kernel)

        x = torch.tensor([np.linspace(-1.5, 1.5, 1000)], dtype=torch.float32).T

        y = generator(x, n_samples=3)
        plt.plot(x, y)#, color="salmon")

        # generator.kernel.base_kernel.period_length = 1.0
        # generator.kernel.base_kernel.lengthscale = 1.0
        # generator.kernel.outputscale = 1.0

        # y = generator(x, n_samples=3)
        # plt.plot(x, y, color="dodgerblue")

        plt.title(f"GP Generator: {config_set_no}")
        plt.show()
