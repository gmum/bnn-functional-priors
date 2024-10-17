from pyro.nn import PyroModule, PyroSample
import logging
import torch
from torch import nn
import pyro
import pyro.distributions as dist


class SingleHiddenLayerWide1DRegressionNNWithGaussianPriors(PyroModule):
    def __init__(
        self, parameters, likelihood_scale, net_width=200, in_dim=1, out_dim=1
    ):
        super().__init__()
        self.in_dim = in_dim
        # assert self.in_dim == 1
        self.likelihood_scale = likelihood_scale

        logging.info(
            f"[SingleHiddenLayerWide1DRegressionNNWithGaussianPriors] Using priors parameters = {parameters}."
        )

        # Activation:
        self.activation = parameters.pop("activation")
        if self.activation is None:
            raise ValueError(
                "[SingleHiddenLayerWide1DRegressionNNWithGaussianPriors] "
                "Activation must be passed as a parameter!"
            )
        # logging.info(f"[SingleHiddenLayerWide1DRegressionNNWithGaussianPriors] Using activation = {self.activation}.")
        if hasattr(
            self.activation, "named_parameters"
        ):  # turn off gradients in the module
            [p.requires_grad_(False) for _, p in self.activation.named_parameters()]

        self.layer1 = PyroModule[nn.Linear](in_dim, net_width)  # Input to hidden layer
        self.layer2 = PyroModule[nn.Linear](
            net_width, out_dim
        )  # Hidden to output layer

        # Set layer parameters as random variables
        self.layer1.weight = PyroSample(
            parameters["layer1.weight"].expand([net_width, in_dim]).to_event(2)
        )
        self.layer1.bias = PyroSample(
            parameters["layer1.bias"].expand([net_width]).to_event(1)
        )
        self.layer2.weight = PyroSample(
            parameters["layer2.weight"].expand([out_dim, net_width]).to_event(2)
        )
        self.layer2.bias = PyroSample(
            parameters["layer2.bias"].expand([out_dim]).to_event(1)
        )

    def forward(self, x, y=None):
        x = x.reshape(-1, self.in_dim)
        if y is not None:
            y = y.flatten()  # assume 1D outputs

        x = self.layer1(x)
        with torch.no_grad():
            x = self.activation(x)
        mu = self.layer2(x)

        mu = mu.flatten()  # assume 1D outputs

        # Likelihood model
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample(
                "obs", dist.Normal(loc=mu, scale=self.likelihood_scale), obs=y
            )

        # Register as a deterministic output
        pyro.deterministic("mu", mu)

        if y is not None:
            log_likelihood = dist.Normal(loc=mu, scale=self.likelihood_scale).log_prob(
                y
            )
            pyro.deterministic("log_likelihood", log_likelihood)

        return mu
