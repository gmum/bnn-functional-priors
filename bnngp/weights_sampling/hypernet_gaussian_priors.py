import torch
from .constraints import make_positive


def sample_factorized_gaussian_with_hypernet_params(
    named_parameters,
    hypernet_output,
    n_samples=1,
    epsilon_scale=1e-8,
    zero_locations=False,
):
    """Returns n_samples sets of parameters for a nn with named_parameters.

    First, Gaussian distribution's hyperparameters are extracted from hypernet_output.
    Then, neural network parameters are sampled from Gaussians with given (hyper-)parameters.

    Args:
        zero_locations (bool): if True hypernet outputs for locations will be ignored (multiplied by zero).
    """
    named_parameters = list(named_parameters)
    assert 2 * len(named_parameters) == len(
        hypernet_output
    ), f"2*{len(named_parameters)} != {len(hypernet_output)}"

    # extract locations and diagonal of the covariance
    locs = hypernet_output[..., : len(named_parameters)] * (
        0.0 if zero_locations else 1.0
    )
    unnormalized_scales = hypernet_output[..., len(named_parameters) :]

    samples = {}
    nlls = {}
    # for each of the network parameters sample from a 1D Gaussian
    for i, (n, parameter) in enumerate(named_parameters):
        loc = locs[i]
        unnormalized_scale = unnormalized_scales[i]

        loc_matched = loc.expand(parameter.shape)
        scale_matched = make_positive(unnormalized_scale).expand(parameter.shape)

        q = torch.distributions.Normal(loc_matched, scale_matched + epsilon_scale)
        sample = q.rsample(torch.Size([n_samples]))

        # calc total NLL for all params (out shape==n_samples)
        data_dims = list(range(1, len(sample.shape)))
        nll = -q.log_prob(sample).sum(dim=data_dims)
        nll = nll.to(sample.device)

        samples[n] = sample
        nlls[n] = nll

    return samples, nlls
