from . import gp_activations

from .gp_activations import *
from .learnable_conditioned_nn_activations import *
from .learnable_conditioned_nn_activations import (
    NNActivationConditioned,
    PeriodicNNActivationConditioned,
)

try:
    from .learnable_activations import (
        ActivationsCombination,
        NNActivation,
        PeriodicNN,
        NN2Fourier,
    )
except Exception as e:
    print(f"ERROR: Failed to import from learnable_activations: {e}")

try:
    from .pwl_activations import PWLLearnableActivation
except Exception as e:
    print(f"ERROR: Failed to import from pwl_activations: {e}")

try:
    from .pwla_activations import PWLA2dLearnableActivation
except Exception as e:
    print(f"ERROR: Failed to import from pwla_activations: {e}")

try:
    from rational.torch import Rational
except Exception as e:
    print(f"ERROR: Failed to import from rational.torch: {e}")

import logging
from torch import nn


class ActivationAPIUnifierForNonLearnable:
    """Wraps an activation so it follows the API for conditioned activations."""

    def __init__(self, activation_func):
        self.activation_obj = activation_func

    def __call__(self, a, conditional_params=None):
        return self.activation_obj(a)

    def to(self, device):
        try:
            self.activation_obj.to(device)
        except Exception as e:
            logging.warning(
                f"Activation={self.activation_obj} is not a Torch object and cannot be moved to device={device}: {e}"
            )

    def get_conditional_params_shapes(self):
        logging.warning(
            f"Activation={self.activation_obj} does not have any conditioned parameters (target_param_shapes is empty)."
        )
        return []

    def parameters(self):
        try:
            return self.activation_obj.parameters()
        except Exception as e:
            logging.warning(
                f"Activation={self.activation_obj} does not have any trainable parameters: {e}"
            )
            return []

    def named_parameters(self):
        try:
            return self.activation_obj.named_parameters()
        except Exception as e:
            logging.warning(
                f"Activation={self.activation_obj} does not have any trainable parameters: {e}"
            )
            return []

    def eval(self):
        try:
            return self.activation_obj.eval()
        except Exception as e:
            logging.warning(
                f"Activation={self.activation_obj} does not have eval() method: {e}"
            )

    def train(self):
        try:
            return self.activation_obj.train()
        except Exception as e:
            logging.warning(
                f"Activation={self.activation_obj} does not have train() method: {e}"
            )


def get_activation(name, generator=None, meta=None, **kwargs):
    """Returns NN's activation function.

    If activation is a str then returns gp_activations::activation.
    Otherwise, first try to get one from generator.activation.
    If failed returns a dummy one (=meta.activation).
    """
    try:
        return _get_learnable_activation(name=name, **kwargs)
    except ValueError as e:
        logging.info(
            f"Activation with name={name} was not recognized as learnable ({e}). Using fixed activation."
        )

    if name and name.lower() != "none":
        logging.info(f"Selecting activation function by its name = {name}")
        activation = getattr(gp_activations, name)

    else:
        if generator is not None and hasattr(generator, "activation"):
            activation = generator.activation

        elif meta is not None:
            logging.warning(
                "WARNING: Generator has no activation (using a dummy one)! Set --activation!"
            )
            activation = meta.activation

        else:
            raise ValueError(f"I don't know how to set activation={name}!")

    return ActivationAPIUnifierForNonLearnable(activation)


def _parse_activation_name(name):
    """Parses values of additional parameters for an activation object."""
    parts = name.split("_")
    name = parts[0]
    args = parts[1:]

    if len(args) <= 0:
        return name, {}

    kwargs = {}
    if name == "rational":
        return name, {
            "numerator": int(args[0]),
            "denominator": int(args[1]),
            "version": args[2],
        }

    elif name == "pwl":
        kwargs = {
            "breakpoints": int(args[0]),
        }
        try:
            kwargs["min_x"] = float(args[1])
            kwargs["max_x"] = float(args[2])
        except Exception:
            pass
        return (name, kwargs)

    elif name == "pwla":
        kwargs = {
            "N": int(args[0]),
            "momentum": float(args[1]),
        }
        try:
            kwargs["min_x"] = float(args[2])
            kwargs["max_x"] = float(args[3])
        except Exception:
            pass
        return (name, kwargs)

    elif name == "combination":
        kwargs = {"nodes_per_activation": int(args[0])}
        return (name, kwargs)

    elif name == "nn" or name == "nnsilu" or name.startswith("conditionalnn"):
        kwargs = {"hidden_layers": int(args[0]), "width": int(args[1])}
        return (name, kwargs)

    elif "periodic" in name or "fourier" in name:
        kwargs = {"n_nodes": int(args[0])}
        return (name, kwargs)

    else:
        raise ValueError(f"Unknown activation name ({name})!")


def _get_learnable_activation(name, **kwargs):
    """Takes a string e.g. rational_5_4_B and creates a torch module."""
    (name, extracted_kwargs) = _parse_activation_name(name)
    kwargs.update(extracted_kwargs)

    name = name.lower()
    if name == "rational":
        numerator = kwargs.get("numerator", 5)
        denominator = kwargs.get("denominator", 4)
        version = kwargs.get("version", "B")
        logging.info(
            f"[get_learnable_activation] activation=Rational(degrees=({numerator}, {denominator}), version={version})"
        )
        return Rational(degrees=(numerator, denominator), version=version)

    elif name == "pwl":
        breakpoints = kwargs.get("breakpoints", 6)
        min_x = kwargs.get("min_x", None)
        max_x = kwargs.get("max_x", None)
        logging.info(
            f"[get_learnable_activation] activation=PWLLearnableActivation(num_breakpoints={breakpoints},"
            f" min_x={min_x}, max_x={max_x})"
        )
        return PWLLearnableActivation(
            num_breakpoints=breakpoints,
            min_x=min_x,
            max_x=max_x,
        )

    elif name == "pwla":
        N = kwargs.get("N", 16)
        momentum = kwargs.get("momentum", 0.9)
        min_x = kwargs.get("min_x", None)
        max_x = kwargs.get("max_x", None)
        logging.info(
            f"[get_learnable_activation] activation=PWLA2dLearnableActivation(N={N}, momentum={momentum},"
            f" min_x={min_x}, max_x={max_x})"
        )
        return PWLA2dLearnableActivation(
            N=N,
            momentum=momentum,
            min_x=min_x,
            max_x=max_x,
        )

    elif name == "combination":
        nodes_per_activation = kwargs.get("nodes_per_activation", 1)
        logging.info(
            f"[get_learnable_activation] activation=ActivationsCombination(nodes_per_activation={nodes_per_activation})"
        )
        return ActivationsCombination(nodes_per_activation=nodes_per_activation)

    elif name == "periodic":
        n_nodes = kwargs.get("n_nodes", 10)
        logging.info(
            f"[get_learnable_activation] activation=PeriodicNN(n_nodes={n_nodes}, train_freqs=True, train_in=False, train_out=False)"
        )
        return PeriodicNN(
            n_nodes=n_nodes, train_freqs=True, train_in=False, train_out=False
        )

    elif name == "periodic-biased":
        n_nodes = kwargs.get("n_nodes", 10)
        logging.info(
            f"[get_learnable_activation] activation=PeriodicNN(n_nodes={n_nodes}, train_freqs=True, train_in=bias, train_out=False)"
        )
        return PeriodicNN(
            n_nodes=n_nodes, train_freqs=True, train_in="bias", train_out=False
        )

    elif name == "fourier":
        n_nodes = kwargs.get("n_nodes", 10)
        logging.info(
            f"[get_learnable_activation] activation=PeriodicNN(n_nodes={n_nodes}, train_freqs=False)"
        )
        return PeriodicNN(
            n_nodes=n_nodes, train_freqs=False, train_in=True, train_out=True
        )

    elif name == "fourier1":
        n_nodes = kwargs.get("n_nodes", 10)
        logging.info(
            f"[get_learnable_activation] activation=PeriodicNN(n_nodes={n_nodes}, train_freqs=True)"
        )
        return PeriodicNN(
            n_nodes=n_nodes, train_freqs=True, train_in=True, train_out=True
        )

    elif name == "fourier2":  # the best performing periodic variant
        n_nodes = kwargs.get("n_nodes", 10)
        logging.info(
            f"[get_learnable_activation] activation=PeriodicNN(n_nodes={n_nodes}, train_freqs=True)"
        )
        return PeriodicNN(
            n_nodes=n_nodes, train_freqs=True, train_in=False, train_out="weight"
        )

    elif name == "nn2fourier":
        logging.info(f"[get_learnable_activation] activation=NN2Fourier()")
        return NN2Fourier()
    
    elif name == "nn2fourier2":
        logging.info(f"[get_learnable_activation] activation=NN2Fourier()")
        return NN2Fourier(hidden_layers=2, width=10, n_nodes=10, activation=nn.SiLU())
    
    elif name == "nn2fourier3":
        logging.info(f"[get_learnable_activation] activation=NN2Fourier()")
        return NN2Fourier(hidden_layers=3, width=10, n_nodes=20, activation=nn.SiLU())        

    elif name == "nn":
        hidden_layers = kwargs.get("hidden_layers", 1)
        width = kwargs.get("width", 5)
        logging.info(
            f"[get_learnable_activation] activation=NNActivation(hidden_layers={hidden_layers}, width={width})"
        )
        return NNActivation(hidden_layers=hidden_layers, width=width)

    elif name == "nnsilu":
        hidden_layers = kwargs.get("hidden_layers", 1)
        width = kwargs.get("width", 5)
        logging.info(
            f"[get_learnable_activation] activation=NNActivation(hidden_layers={hidden_layers}, width={width}) activation=nn.SiLU()"
        )
        return NNActivation(
            hidden_layers=hidden_layers, width=width, activation=nn.SiLU()
        )

    elif name == "conditionalnn":
        hidden_layers = kwargs.get("hidden_layers", 1)
        width = kwargs.get("width", 5)
        logging.info(
            f"[get_learnable_activation] activation=NNActivationConditioned(hidden_layers={hidden_layers}, width={width})"
        )
        return NNActivationConditioned(hidden_layers=hidden_layers, width=width)

    elif name == "conditionalfourier2" or name == "conditionalperiodic":
        n_nodes = kwargs.get("n_nodes", 10)
        logging.info(
            f"[get_learnable_activation] activation=PeriodicNNActivationConditioned(n_nodes={n_nodes})"
        )
        return PeriodicNNActivationConditioned(n_nodes=n_nodes)

    elif name == "conditionalnnsilu":
        hidden_layers = kwargs.get("hidden_layers", 1)
        width = kwargs.get("width", 5)
        logging.info(
            f"[get_learnable_activation] activation=NNActivationConditioned(hidden_layers={hidden_layers}, width={width} activation=nn.SiLU())"
        )
        return NNActivationConditioned(
            hidden_layers=hidden_layers, width=width, activation=nn.SiLU()
        )

    else:
        raise ValueError(f"Unknown activation name ({name})!")
