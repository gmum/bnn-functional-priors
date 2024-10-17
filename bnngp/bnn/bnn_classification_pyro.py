from pyro.nn import PyroModule, PyroSample
import logging
import torch
from torch import nn
import pyro
import pyro.distributions as dist


# BNN: Single-hidden-layer NN
class SingleHiddenLayerWide1DClassificationNNWithGaussianPriors(PyroModule):
    def __init__(
        self,
        activation=None,
        priors=None,
        config={},
        net_width=1000,
        in_dim=2,
        out_dim=1,
    ):
        super().__init__()
        self.in_dim = in_dim

        # Activation:
        self.activation = activation or config.get("activation")
        if self.activation is None:
            raise ValueError(
                "[SingleHiddenLayerWide1DClassificationNNWithGaussianPriors] "
                "Activation must be passed as a parameter or loaded from a json file!"
            )
        logging.info(
            f"[SingleHiddenLayerWide1DClassificationNNWithGaussianPriors] Using activation = {self.activation}."
        )

        if hasattr(
            self.activation, "named_parameters"
        ):  # turn off gradients in the module
            [p.requires_grad_(False) for n, p in self.activation.named_parameters()]

        # Priors:
        priors = priors or config
        if priors is None:
            raise ValueError(
                "[SingleHiddenLayerWide1DClassificationNNWithGaussianPriors] "
                "Parameters must be passed or loaded from a json file!"
            )
        logging.info(
            "[SingleHiddenLayerWide1DClassificationNNWithGaussianPriors] "
            f"Loading priors from parameters object = {priors}."
        )

        # Model:
        self.layer1 = PyroModule[nn.Linear](in_dim, net_width)  # Input to hidden layer
        self.layer2 = PyroModule[nn.Linear](
            net_width, out_dim
        )  # Hidden to output layer

        # Set layer parameters as random variables
        self.layer1.weight = PyroSample(
            priors["layer1.weight"].expand([net_width, in_dim]).to_event(2)
        )
        self.layer1.bias = PyroSample(
            priors["layer1.bias"].expand([net_width]).to_event(1)
        )
        self.layer2.weight = PyroSample(
            priors["layer2.weight"].expand([out_dim, net_width]).to_event(2)
        )
        self.layer2.bias = PyroSample(
            priors["layer2.bias"].expand([out_dim]).to_event(1)
        )

    def forward(self, x, y=None):
        x = x.reshape(-1, self.in_dim)
        x = self.layer1(x)
        with torch.no_grad():
            x = self.activation(x)
        mu = self.layer2(x).squeeze()

        # Likelihood model
        p = torch.sigmoid(mu)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Bernoulli(probs=p), obs=y)
            # obs = pyro.sample("obs", dist.RelaxedBernoulliStraightThrough(probs=p, temperature=torch.tensor(0.1)), obs=y)

        # Register p as a deterministic output
        pyro.deterministic("p", p)

        if y is not None:
            log_likelihood = dist.Bernoulli(probs=p).log_prob(y)
            pyro.deterministic("log_likelihood", log_likelihood)

        return p  # TO CHECK ?#?@?@?!
