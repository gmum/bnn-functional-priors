from .shared_gaussian_priors import create_factorized_shared_gaussian_sampler
from .constraints import invert_positive

import torch
import numpy as np
import logging


def create_gaussian_priors(
    net,
    init={
        "mu": 0.0,
        "ma": 0.0,
        "mv": 0.0,
        "mb": 0.0,
        "su": 1.0,
        "sa": 1.0,
        "sb": 1.0,
        "wv": 1.0,
    },
    train_locs: bool = False,
    train_scales: bool = False,
    train_remaining: bool = False,
    create=create_factorized_shared_gaussian_sampler,
    separator: str = ".",
    **samplers_kwargs,
):
    logging.info(f"Creating Gaussian priors with init={init} create={create}")

    n2p = {n: p for n, p in net.named_parameters()}
    if sorted(n2p.keys()) != sorted(
        [
            "fc1.bias",
            "fc1.weight",
            "fc2.bias",
            "fc2.weight",
        ]
    ):
        raise ValueError(
            f"Single-hidden layer net with fc1 & fc2 is expected! Got={net}"
        )

    samplers = {}
    variational_params = {}
    aux = {}

    def _create_for_parameter(parameter_name, loc=0.0, std=1.0):
        p = n2p[parameter_name]
        p_shape = p.flatten()[0].shape
        p_sampler, p_variational_params, p_aux = create(
            p,
            loc_initalization=lambda _: (torch.ones(p_shape) * loc).clone(),
            uscale_initialization=lambda _: (
                torch.ones(p_shape) * invert_positive(std)
            ).clone(),
            **samplers_kwargs,
        )
        # update dictionaries
        samplers[parameter_name] = p_sampler
        for vn, p in p_variational_params.items():
            variational_params[parameter_name + separator + vn] = p
        for n, o in p_aux.items():
            aux[parameter_name + separator + n] = o

    assert n2p["fc1.bias"].shape[-1] == n2p["fc2.weight"].shape[-1]
    net_width = n2p["fc1.bias"].shape[-1]

    _create_for_parameter("fc1.weight", init["mu"], init["su"])
    _create_for_parameter("fc1.bias", init["ma"], init["sa"])
    _create_for_parameter("fc2.weight", init["mv"], init["wv"] / np.sqrt(net_width))
    _create_for_parameter("fc2.bias", init["mb"], init["sb"])

    variational_params = _select_trained_variational_params(
        train_locs, train_scales, train_remaining, variational_params
    )

    return samplers, variational_params, aux


def _select_trained_variational_params(
    train_locs, train_scales, train_remaining, variational_params
):
    locs = {vn: vp for vn, vp in variational_params.items() if vn.endswith("loc")}
    scales = {vn: vp for vn, vp in variational_params.items() if vn.endswith("scale")}
    remaining = {
        vn: vp
        for vn, vp in variational_params.items()
        if not (vn in locs or vn in scales)
    }

    variational_params = {}
    if train_locs:
        variational_params.update(locs)
    else:
        for vp in locs.values():
            vp.requires_grad_(False)

    if train_scales:
        variational_params.update(scales)
    else:
        for vp in scales.values():
            vp.requires_grad_(False)

    if train_remaining:
        variational_params.update(remaining)
    else:
        for vp in remaining.values():
            vp.requires_grad_(False)

    return variational_params
