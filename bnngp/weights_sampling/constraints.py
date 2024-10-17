""" Constraining parameters to use the normalized scales (e.g. scales to be positive). """

import torch


def make_positive(value):
    return torch.nn.functional.softplus(value)


def invert_positive(value):
    """Inverse softplus function.

    Args:
        bias (float or tensor): the value to be softplus-inverted.
    """
    is_tensor = True
    if not isinstance(value, torch.Tensor):
        is_tensor = False
        value = torch.tensor(value)
    out = value.expm1().clamp_min(1e-6).log()
    if not is_tensor and out.numel() == 1:
        return out.item()
    return out


def ensure_normalized(name, value):
    return make_positive(value) if "unnormalized" in name else value


def ensure_all_normalized(variational_params):
    """Conver all unconstrained parameters to positive."""
    return {
        n.replace("unnormalized_", ""): ensure_normalized(n, v)
        for n, v in variational_params.items()
    }
