from .shared_gaussian_priors import (
    create_factorized_shared_gaussian_sampler_zero_loc,
    create_factorized_shared_gaussian_sampler,
    create_factorized_shared_invgamma_gaussian_mixture_sampler,
    create_factorized_shared_tstudent_sampler,
)
from reparameterized import parameters
import torch
import logging


def create_factorized_samplers(
    neural_network,
    create_parameter_sampler=create_factorized_shared_gaussian_sampler_zero_loc,
    separator=".",
    **create_parameter_sampler_kwargs,
):
    """Creates samplers {sampler_name: sampling function}
    and their variational parameters {parameter_name: tensor}.

    It iterates over network parameters and executes create_parameter_sampler for each of them.
    """
    logging.info(
        f"Creating sampler={create_parameter_sampler} for each parameter of {neural_network}"
    )
    samplers = {}
    variational_params = {}
    aux = {}
    for n, p in neural_network.named_parameters():
        sampler, variational_params1, aux1 = create_parameter_sampler(
            p, **create_parameter_sampler_kwargs
        )

        samplers[n] = sampler

        for vp_n, vp_t in variational_params1.items():
            variational_params[n + separator + vp_n] = vp_t

        for a_n, a_o in aux1.items():
            aux[n + separator + a_n] = a_o

    return samplers, variational_params, aux


def create_factorized_sampler_gaussian_zero_loc(
    neural_network, **create_parameter_sampler_kwargs
):
    """Sampling for each of the net parameters ~N(0, sigma)."""
    return create_factorized_samplers(
        neural_network,
        create_parameter_sampler=create_factorized_shared_gaussian_sampler_zero_loc,
        **create_parameter_sampler_kwargs,
    )


def create_factorized_sampler_gaussian(
    neural_network, **create_parameter_sampler_kwargs
):
    """Sampling for each of the net parameters ~N(mu, sigma)."""
    return create_factorized_samplers(
        neural_network,
        create_parameter_sampler=create_factorized_shared_gaussian_sampler,
        **create_parameter_sampler_kwargs,
    )


def create_factorized_sampler_invgamma_gaussian_mixture(
    neural_network, **create_parameter_sampler_kwargs
):
    return create_factorized_samplers(
        neural_network,
        create_parameter_sampler=create_factorized_shared_invgamma_gaussian_mixture_sampler,
        **create_parameter_sampler_kwargs,
    )


def create_factorized_tstudent_sampler(
    neural_network, **create_parameter_sampler_kwargs
):
    return create_factorized_samplers(
        neural_network,
        create_parameter_sampler=create_factorized_shared_tstudent_sampler,
        **create_parameter_sampler_kwargs,
    )


def wrap_nn_with_parameters_resampling(nn, samplers):
    """Wraps nn so parameters are resampled for each new input."""

    def wrapped_nn(*inputs):
        parameters_samples, _ = parameters.sample_parameters(samplers, n_samples=1)
        for parameters_samples1 in parameters.take_parameters_sample(
            parameters_samples
        ):
            parameters.load_state_dict(nn, parameters_samples1)
            return nn(*inputs)

    return wrapped_nn


def sample_functions_from_nn(nn, samplers, *inputs, n_samples=1):
    """Passes input_grid_x through nn with parameters resampled n_samples times."""
    parameters_samples, _ = parameters.sample_parameters(samplers, n_samples=n_samples)
    return sample_functions_from_nn_with_parameters(nn, parameters_samples, *inputs)


def sample_functions_from_nn_with_parameters(nn, parameters_samples, *inputs):
    """Passes input_grid_x through nn for each of parameter samples and returns outputs."""
    sampled_functions = []
    for parameters_samples1 in parameters.take_parameters_sample(parameters_samples):
        parameters.load_state_dict(nn, parameters_samples1)
        y = nn(*inputs)
        sampled_functions.append(y)
    sampled_functions = torch.hstack(sampled_functions).T
    return sampled_functions
