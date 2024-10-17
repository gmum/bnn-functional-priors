from weights_sampling import (
    sample_factorized_gaussian_with_hypernet_params,
    sample_functions_from_nn_with_parameters,
)
from typing import Callable, Dict, Any, Optional
import logging
import torch


def sample_from_net(
    net: torch.nn.Module,
    hypernet: Callable[[torch.Tensor], torch.Tensor],
    hypernet_input_builder: Callable[
        [torch.Tensor, Optional[torch.Tensor], Dict[str, Any]], torch.Tensor
    ],
    activation: Callable,
    batch_size: int,
    input_grid_x: torch.Tensor,
    target_hyperparams: Dict[str, Any],
    zero_locations: bool = True,  # for priors: do we want to learn locations or only stds
    batch_target: Optional[torch.Tensor] = None,
    **ignored_kwargs: Any,
):
    """
    Samples from a Bayesian Neural Network (BNN) using a hypernetwork to produce
    its activations and prior parameters.

    Args:
        net (torch.nn.Module): The main neural network (BNN) whose parameters are sampled.
        hypernet (Callable): The hypernetwork that generates the parameters of the priors
            and activation function for the main network.
        hypernet_input_builder (Callable): A function that prepares the input to the
            hypernetwork based on the input grid and target hyperparameters.
        activation (Callable): The activation function used in the neural network.
        batch_size (int): The number of samples to generate from the BNN.
        input_grid_x (torch.Tensor): The input grid used for the network = a batch of input data points.
        target_hyperparams (Dict[str, Any]): Hyperparameters required by the hypernetwork to generate
            the activation and prior parameters.
        zero_locations (bool, optional): If True, only the standard deviations (not the means)
            of the priors are learned. Defaults to True.
        batch_target (Optional[torch.Tensor], optional): Target data points associated with the batch
            input, which might be used by the hypernetwork to construct the priors. Defaults to None.
        **ignored_kwargs: Additional arguments that are not used by the function.

    Returns:
        torch.Tensor: A batch of sampled function values from the BNN corresponding to the
        provided input grid.
    """

    logging.debug("obtain activation & priors' params from a hypernet")
    hypernet_output = hypernet(
        hypernet_input_builder(input_grid_x, batch_target, target_hyperparams)
    )
    activation_params = hypernet_output[:-1]
    priors_params = hypernet_output[-1].flatten()

    logging.debug("sample BNN parameters from the prior")
    parameters_samples, _ = sample_factorized_gaussian_with_hypernet_params(
        net.named_parameters(),
        priors_params,
        n_samples=batch_size,
        zero_locations=zero_locations,
    )

    logging.debug("sample functions from network")
    batch_learning = sample_functions_from_nn_with_parameters(
        net,
        parameters_samples,
        input_grid_x,
        lambda a: activation(a, activation_params),
    )

    return batch_learning
