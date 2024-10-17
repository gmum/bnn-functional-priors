import torch
from . import inverse_gamma
from typing import Tuple, Callable, Dict
from .constraints import make_positive


def create_factorized_shared_gaussian_sampler(
    parameter: torch.Tensor,
    device=None,
    epsilon_scale=1e-8,
    loc_initalization=lambda parameter: parameter.flatten()[0].clone().detach(),
    uscale_initialization=lambda parameter: torch.randn_like(parameter.flatten()[0]),
    **ignored_params,
) -> Tuple[Callable, Dict[str, torch.Tensor], Dict[str, object]]:
    """Similar to create_factorized_gaussian with variational params common for all values."""
    loc = loc_initalization(parameter)
    loc = loc.requires_grad_(True).to(device or parameter.device)

    unnormalized_scale = uscale_initialization(parameter)
    unnormalized_scale = unnormalized_scale.requires_grad_(True).to(
        device or parameter.device
    )

    def sample_factorized_gaussian(n_samples=1):
        loc_matched = loc.expand(parameter.shape)
        scale_matched = make_positive(unnormalized_scale).expand(parameter.shape)

        q = torch.distributions.Normal(loc_matched, scale_matched + epsilon_scale)
        sample = q.rsample(torch.Size([n_samples]))

        # calc total NLL for all params (out shape==n_samples)
        data_dims = list(range(1, len(sample.shape)))
        nll = -q.log_prob(sample).sum(dim=data_dims)
        nll = nll.to(sample.device)

        return sample, nll

    variational_params = {"loc": loc, "unnormalized_scale": unnormalized_scale}
    return sample_factorized_gaussian, variational_params, {}


def create_factorized_shared_gaussian_sampler_zero_loc(
    parameter: torch.Tensor,
    device=None,
    epsilon_scale=1e-8,
    uscale_initialization=lambda parameter: torch.randn_like(parameter.flatten()[0]),
    **ignored_params,
) -> Tuple[Callable, Dict[str, torch.Tensor], Dict[str, object]]:
    """Similar to create_factorized_gaussian with variational params common for all values.

    Vairant with fixed locations = 0.
    """
    loc = torch.zeros_like(parameter)
    loc = loc.requires_grad_(False).to(device or parameter.device)

    unnormalized_scale = uscale_initialization(parameter)
    unnormalized_scale = unnormalized_scale.requires_grad_(True).to(
        device or parameter.device
    )

    def sample_factorized_gaussian(n_samples=1):
        loc_matched = loc.expand(parameter.shape)
        scale_matched = make_positive(unnormalized_scale).expand(parameter.shape)

        q = torch.distributions.Normal(loc_matched, scale_matched + epsilon_scale)
        sample = q.rsample(torch.Size([n_samples]))

        # calc total NLL for all params (out shape==n_samples)
        data_dims = list(range(1, len(sample.shape)))
        nll = -q.log_prob(sample).sum(dim=data_dims)
        nll = nll.to(sample.device)

        return sample, nll

    variational_params = {"unnormalized_scale": unnormalized_scale}
    return sample_factorized_gaussian, variational_params, {}


def create_factorized_invgamma_gaussian_mixture_sampler(
    parameter: torch.Tensor,
    device=None,
    epsilon_concentration=1e-8,
    epsilon_rate=1e-8,
    uconcentration_initialization=lambda parameter: torch.randn_like(
        parameter.flatten()[0]
    ),
    urate_initialization=lambda parameter: torch.randn_like(parameter.flatten()[0]),
    **ignored_params,
) -> Tuple[Callable, Dict[str, torch.Tensor], Dict[str, object]]:
    """Creates a sampler X ~ Normal(0, sigma^2), sigma^2 ~ Inv-Gamma(concentration, rate)."""
    loc = torch.zeros_like(parameter)
    loc = loc.requires_grad_(False).to(device or parameter.device)

    unnormalized_concentration = uconcentration_initialization(parameter)
    unnormalized_concentration = unnormalized_concentration.requires_grad_(True).to(
        device or parameter.device
    )

    unnormalized_rate = urate_initialization(parameter)
    unnormalized_rate = unnormalized_rate.requires_grad_(True).to(
        device or parameter.device
    )

    def sampler(n_samples=1):
        rate_matched = make_positive(unnormalized_rate).expand(parameter.shape)
        concentration_matched = make_positive(unnormalized_concentration).expand(
            parameter.shape
        )

        q0 = inverse_gamma.InverseGamma(
            concentration_matched + epsilon_concentration,
            rate_matched + epsilon_rate,
            validate_args=True,
        )
        variances = q0.rsample(torch.Size([n_samples]))
        scales = torch.sqrt(variances)

        q = torch.distributions.Normal(loc, scales)
        sample = q.rsample()

        return sample, None

    variational_params = {
        "unnormalized_concentration": unnormalized_concentration,
        "unnormalized_rate": unnormalized_rate,
    }
    return sampler, variational_params, {}


def sample_invgamma_normal_mixture(n_samples, alpha, beta, loc=0.0):
    # Create an inverse gamma distribution with parameters alpha and beta
    inv_gamma_dist = inverse_gamma.InverseGamma(alpha, beta)

    # Sample variances from the inverse gamma distribution
    sigma_squared_samples = inv_gamma_dist.rsample((n_samples,))

    # Create a normal distribution with mean 0 and sampled variances
    normal_dist = torch.distributions.Normal(loc, torch.sqrt(sigma_squared_samples))
    normal_samples = normal_dist.rsample()

    return normal_samples, None


def create_factorized_shared_invgamma_gaussian_mixture_sampler(
    parameter: torch.Tensor,
    device=None,
    epsilon_concentration=1e-12,
    epsilon_rate=1e-12,
    uconcentration_init=lambda parameter: torch.ones_like(parameter.flatten()[0]),
    urate_init=lambda parameter: torch.ones_like(parameter.flatten()[0]),
    **ignored_params,
):  # -> Tuple[Callable, Dict[str, torch.Tensor], Dict[str, object]]:
    """Creates a sampler X ~ Normal(0, sigma^2), sigma^2 ~ Inv-Gamma(concentration, rate)."""
    loc = torch.zeros_like(parameter)
    loc = loc.requires_grad_(False).to(device or parameter.device)

    uconcentration = uconcentration_init(parameter)
    uconcentration = uconcentration.requires_grad_(True).to(device or parameter.device)

    urate = urate_init(parameter)
    urate = urate.requires_grad_(True).to(device or parameter.device)

    def sampler(n_samples=1):
        rate_matched = make_positive(urate).expand(parameter.shape)
        concentration_matched = make_positive(uconcentration).expand(parameter.shape)

        sample, _ = sample_invgamma_normal_mixture(
            n_samples,
            concentration_matched + epsilon_concentration,
            rate_matched + epsilon_rate,
            loc=loc,
        )

        return sample, None

    variational_params = {
        "unnormalized_concentration": uconcentration,
        "unnormalized_rate": urate,
    }
    return sampler, variational_params, {}


def sample_from_student_t(n_samples, alpha, beta):
    # Student's t-distribution parameters
    df = alpha  # degrees of freedom
    sqrt = torch.sqrt if isinstance(beta, torch.Tensor) else torch.math.sqrt
    scale = sqrt(beta / alpha)  # scale

    # Student's t-distribution
    t_dist = torch.distributions.StudentT(df, scale=scale)
    t_samples = t_dist.rsample((n_samples,))

    return t_samples, -t_dist.log_prob(t_samples)


def create_factorized_shared_tstudent_sampler(
    parameter: torch.Tensor,
    device=None,
    epsilon_concentration=1e-8,
    epsilon_rate=1e-8,
    uconcentration_init=lambda parameter: torch.ones_like(parameter.flatten()[0]),
    urate_init=lambda parameter: torch.ones_like(parameter.flatten()[0]),
    **ignored_params,
):  # -> Tuple[Callable, Dict[str, torch.Tensor], Dict[str, object]]:
    """Creates a sampler X ~ StudentT(df=concentration, sqrt(rate/concentration))."""
    uconcentration = uconcentration_init(parameter)
    uconcentration = uconcentration.requires_grad_(True).to(device or parameter.device)

    urate = urate_init(parameter)
    urate = urate.requires_grad_(True).to(device or parameter.device)

    def sampler(n_samples=1):
        rate_matched = make_positive(urate).expand(parameter.shape)
        concentration_matched = make_positive(uconcentration).expand(parameter.shape)

        sample, nll = sample_from_student_t(
            n_samples,
            concentration_matched + epsilon_concentration,
            rate_matched + epsilon_rate,
        )

        # calc total NLL for all params (out shape==n_samples)
        data_dims = list(range(1, len(sample.shape)))
        nll = nll.sum(dim=data_dims)
        nll = nll.to(sample.device)

        return sample, nll

    variational_params = {
        "unnormalized_concentration": uconcentration,
        "unnormalized_rate": urate,
    }

    return sampler, variational_params, {}
