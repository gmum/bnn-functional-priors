import torch
import gpytorch
import logging
import math


class GPGenerator1D:
    def __init__(self, kernel: gpytorch.kernels.Kernel) -> None:
        self.kernel = kernel

    def __call__(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        cov = self.kernel(x)
        dim = cov.shape[-1]
        dist = gpytorch.distributions.MultivariateNormal(
            torch.zeros(dim), covariance_matrix=cov
        )
        return dist.sample(
            sample_shape=torch.Size([n_samples])
        ).T  # output shape = [#grid_nodes, 1]


try:
    from baselines.you_need_a_good_prior.optbnn.gp.models.gpr import GPR
    from baselines.you_need_a_good_prior.optbnn.gp import (
        kernels,
        mean_functions,
        priors,
    )
    from baselines.you_need_a_good_prior.optbnn.utils import util
except Exception as e:
    logging.warning(f"Failed to import baselines.you_need_a_good_prior: {e}")


class AllYouNeed1DRegression:
    def __init__(self) -> None:
        util.set_seed(1)

        # GP hyper-parameters
        self.sn2 = 0.1  # noise variance
        leng = 0.6  # lengthscale
        ampl = 1.0  # amplitude

        # Initialize GP Prior
        self.kernel = kernels.RBF(
            input_dim=1,
            ARD=True,
            lengthscales=torch.tensor([leng], dtype=torch.double),
            variance=torch.tensor([ampl], dtype=torch.double),
        )
        self.gpmodel = None

    def _build(self, x):
        if self.gpmodel is None:
            self.gpmodel = GPR(
                X=x,
                Y=torch.zeros_like(x),
                kern=self.kernel,
                mean_function=mean_functions.Zero(),
            )
            self.gpmodel.likelihood.variance.set(self.sn2)
            self.gpmodel = self.gpmodel.to(x.device)

    def __call__(self, x: torch.Tensor, n_samples: int = 1) -> torch.Any:
        self._build(x)
        return self.gpmodel.sample_functions(x, n_samples).squeeze()


class AllYouNeedUCIRegression:
    def __init__(self, input_dim, sn2=0.1) -> None:
        util.set_seed(1)
        logging.info(
            f"[targets_gp.AllYouNeedUCIRegression] creating with input_dim={input_dim} sn2={sn2}"
        )

        # GP hyper-parameters
        self.sn2 = sn2  # noise variance
        leng = math.sqrt(2.0 * input_dim)  # lengthscale
        ampl = 1.0  # amplitude

        # Initialize the mean and covariance function of the target hierarchical GP prior

        self.kernel = kernels.RBF(
            input_dim=input_dim,
            lengthscales=torch.tensor([leng], dtype=torch.double),
            variance=torch.tensor([ampl], dtype=torch.double),
            ARD=True,
        )

        # Place hyper-priors on lengthscales and variances
        self.kernel.lengthscales.prior = priors.LogNormal(
            torch.ones([input_dim]) * math.log(leng), torch.ones([input_dim]) * 1.0
        )
        self.kernel.variance.prior = priors.LogNormal(
            torch.ones([1]) * 0.1, torch.ones([1]) * 1.0
        )

        self.gpmodel = None

    def _build(self, x):
        if self.gpmodel is None:
            self.gpmodel = GPR(
                X=x,
                Y=torch.zeros_like(x).reshape([-1, 1]),
                kern=self.kernel,
                mean_function=mean_functions.Zero(),
            )
            self.gpmodel.likelihood.variance.set(self.sn2)
            self.gpmodel = self.gpmodel.to(x.device)

    def __call__(self, x: torch.Tensor, n_samples: int = 1) -> torch.Any:
        self._build(x)
        return self.gpmodel.sample_functions(x, n_samples).squeeze()


if __name__ == "__main__":
    lengthscale_precision = 1.0

    kernel = gpytorch.kernels.MaternKernel(
        nu=5 / 2,
        lengthscale_constraint=gpytorch.constraints.Interval(
            1.0 - lengthscale_precision / 2, 1.0 + lengthscale_precision / 2
        ),
    )
    batch_size = 123

    generator = GPGenerator1D(kernel=kernel)
    print("generator = GPGenerator1D")

    # import grids

    # train_grid_x = grids.create_uniform_grid(-10, 10, 100, n_dims=3)
    train_grid_x = torch.arange(-10.0, 10.0, 0.1)[:, None]

    print("train_grid_x.shape=", train_grid_x.shape)
    print("a sample from generator=", generator(train_grid_x).shape)

    batch_target = torch.hstack(
        [generator(train_grid_x) for _ in range(batch_size)]
    ).mT  # shape = (batch_size, train_grid_x.len)
    print(f"{batch_size} samples: {batch_target.shape}.T")

    batch_target = generator(train_grid_x, n_samples=batch_size).T
    print(f"{batch_size} samples at once: {batch_target.shape}.T")

    train_grid_x = torch.tensor([-10.0, -5.0, 0.0, 5.0, 10.0])[:, None]
    print("3 samples on grid with 5 elements: ", generator(train_grid_x, 3))

    #########################################################################

    generator = AllYouNeed1DRegression()
    print(
        "AllYouNeed1DRegression: 3 samples on grid with 5 elements: ",
        generator(train_grid_x, 3),
    )
    print(
        "AllYouNeed1DRegression: 3 samples on grid with 5 elements: ",
        generator(train_grid_x, 3),
    )
