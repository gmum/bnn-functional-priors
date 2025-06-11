import torch
import gpytorch

import logging

import targets_gp
from conditioning_common import sample_from_net


def create_generator_matern_gp(lengthscale=1.0, outputscale=1.0, nu=5 / 2, prec=1e-6):
    """Creates GPGenerator1D with Matern kernel of given hyperparams.

    Args:
        lengthscale (float, optional): Kernel hyperparam. Defaults to 1.0.
        outputscale (float, optional): Kernel hyperparam. Defaults to 1.0.
        nu (float, optional): Kernel hyperparam. Defaults to 5/2.
        prec (float, optional): Used to set hyperparams' constraints. Defaults to 1e-6.

    Returns:
        GPGenerator1D with Matern kernel.
    """

    halfprec = prec / 2

    kernel = gpytorch.kernels.MaternKernel(
        nu=nu,
        lengthscale_constraint=gpytorch.constraints.Interval(
            lengthscale - halfprec, lengthscale + halfprec
        ),
    )

    kernel = gpytorch.kernels.ScaleKernel(
        kernel,
        outputscale_constraint=gpytorch.constraints.Interval(
            outputscale - halfprec, outputscale + halfprec
        ),
    )

    target = targets_gp.GPGenerator1D(kernel=kernel)
    return target


def yield_conditioning_matern_gp_lengthscale(ls=[0.01, 0.1, 1.0, 10.0]):
    """Generator that continuously yields dictionaries with varying lengthscale values."""
    while True:
        # Iterate over a predefined list of lengthscale values
        for pow in ls:
            # Yield a dictionary with the current lengthscale value
            yield {"lengthscale": pow}


@torch.no_grad()
def evaluation_matern_gp(
    net,
    activation,
    create_target_generator,
    final_evaluation_input_grid,
    final_evaluation_batch_size,
    hypernet,
    hypernet_input_builder,
    loss_func,
    zero_locations=True,
    lengthscales = [0.01, 0.1, 1.0, 10.0],
    **ignored_kwargs,
):
    assert final_evaluation_input_grid is not None, "[evaluation_matern_gp] Please set the final_evaluation_input_grid"
    
    results = []
    for lengthscale in lengthscales:
        target_hyperparams = {"lengthscale": lengthscale}
        logging.info(f"[evaluation_matern_gp] final evaluation for target_hyperparams={target_hyperparams}")

        try:
            generator = create_target_generator(**target_hyperparams)
            gt_target = generator(
                final_evaluation_input_grid, n_samples=final_evaluation_batch_size
            ).T

            final_test = sample_from_net(
                net,
                hypernet,
                hypernet_input_builder,
                activation,
                batch_size=final_evaluation_batch_size,
                input_grid_x=final_evaluation_input_grid,
                target_hyperparams=target_hyperparams,
                zero_locations=zero_locations,
            )

            final_loss = loss_func(gt_target, final_test).item()
            logging.info(f"[evaluation_matern_gp] final evaluation loss for {target_hyperparams}: {final_loss}")
            results.append((target_hyperparams, final_loss))
        except Exception as e:
            logging.warning(f"[evaluation_matern_gp] evaluation failed: {e}")
            
    return results
