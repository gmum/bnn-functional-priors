import tqdm
import logging
import os
import time
import sys
import random

import torch
import pandas as pd
import json

from matplotlib import pyplot as plt

import utils
import args

import grids
import import_export
import losses
import function_stats
import activations

from bnn import SingleHiddenLayerWideNNWithLearnablePriorsAndActivation
from hypernets import create_conditioning_hypernet

from conditioning_common import sample_from_net
import conditioning_on_varying_lengthscale
import conditioning_on_transformed_input


def train(
    create_target_generator,  # for given hyperparams creates a target sampler
    target_hyperparams_generator,  # yields target sampler hyperparams
    net,
    hypernet,
    hypernet_input_builder,
    zero_locations,  # for priors: do we want to learn locations or only stds
    loss_func,
    create_training_grid,
    activation,
    batch_size,
    num_function_samples,
    shuffle_xs_n_times,
    n_iterations,
    lr,
    report_every_n_iterations=100,
    **ignored_kwargs,
):
    hypernet_parameters = list(p for p in hypernet.parameters() if p.requires_grad)
    activation_parameters = list(p for p in activation.parameters() if p.requires_grad)
    optimized_parameters = hypernet_parameters + activation_parameters
    logging.info(
        f"{len(optimized_parameters)} optimized parameters: "
        f" {len(hypernet_parameters)} from hypernet and "
        f"{len(activation_parameters)} from activation"
    )

    history = []
    optimizer = torch.optim.Adam(optimized_parameters, lr=lr)
    for iteration in tqdm.tqdm(range(n_iterations)):
        loss = 0.0
        for _ in range(shuffle_xs_n_times):

            logging.debug("create a target generator and sample from it")
            target_hyperparams = next(target_hyperparams_generator)
            target = create_target_generator(**target_hyperparams)
            train_grid_x = create_training_grid(
                num_function_samples, **target_hyperparams
            )
            batch_target = target(train_grid_x, n_samples=batch_size).T

            batch_learning = sample_from_net(
                net,
                hypernet,
                hypernet_input_builder,
                activation,
                batch_size,
                train_grid_x,
                target_hyperparams,
                zero_locations=zero_locations,
                batch_target=batch_target,
            )

            loss += loss_func(batch_target, batch_learning)
        loss /= shuffle_xs_n_times

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append(
            {"iteration": iteration, "loss": loss.cpu().detach().numpy().item()}
        )
        history[-1].update(target_hyperparams)
        if iteration % report_every_n_iterations == 0:
            reported_stats = report(**locals(), **ignored_kwargs)
            history[-1].update(reported_stats)

    return history


@torch.no_grad()
def report(
    run_name,
    train_grid_x,
    start_time,
    history,
    loss,
    batch_target,
    batch_learning,
    iteration,
    **ignored_kwargs,
):

    loss_wd = losses.wasserstein_distance(batch_target, batch_learning)
    (
        characteristics_target,
        characteristics_learning,
        comparison_str,
    ) = function_stats.compare_characteristics(batch_target, batch_learning)
    logging.info(
        f"[it={iteration}] loss={loss: .3f} wasserstein={loss_wd:.3f}  [target vs learning]: {comparison_str}"
    )

    # Plot target samples
    plotting_grid_x = train_grid_x
    utils.plot_functions_1D_slices(
        plotting_grid_x,
        batch_target,
        title=f"Target functions from generator",
    )
    plt.savefig(f"{run_name}_iter={iteration}_target.png")
    plt.clf()
    # plt.show()

    # Plot network samples
    utils.plot_functions_1D_slices(
        plotting_grid_x,
        batch_learning,
        title=f"Result from BNN after {iteration} iterations",
    )
    plt.savefig(f"{run_name}_iter={iteration}_results.png")
    plt.clf()
    # plt.show()

    summary_stats = {
        "wasserstein_distance": loss_wd.item(),
    }
    for k, (m, s) in characteristics_learning.items():
        summary_stats["learning_" + k + "_mean"] = m.item()
        summary_stats["learning_" + k + "_std"] = s.item()
    for k, (m, s) in characteristics_target.items():
        summary_stats["target_" + k + "_mean"] = m.item()
        summary_stats["target_" + k + "_std"] = s.item()

    for iteration_results in history[-5:]:
        iteration_results = " ".join(
            f"{k}:{v : .4f}" for k, v in iteration_results.items()
        )
        logging.info(f"[{time.time()-start_time:.1f}s] {iteration_results}")

    return summary_stats


def main_parameterized(
    run_name="",
    bnn_width=1_000,
    loss_func="wasserstein_distance",
    batch_size=256,
    num_function_samples=256,
    input_n_dims=1,
    shuffle_xs_n_times=1,
    freeze_training_grid=False,
    input_data_min_x=-3.0,
    input_data_max_x=3.0,
    final_evaluation_num_function_samples=1024,
    final_evaluation_batch_size=1024,
    final_evaluation_on_training_grid=False,
    n_iterations=1000,
    lr=0.01,
    report_every_n_iterations=200,
    activation="conditionalnn",
    hypernet_arch_no=0,
    conditioning="lengthscale",  # lengthscale/transformed_input
    # transformations applied to data (shift/scaling/shift_and_scaling), when conditioning on transformed_input
    conditioning_input_transformation="shift",
    zero_locations=True,  # for priors: do we want to learn locations or only stds
    random_seed=None,
    description={},  # additional info to be stored along with data and results in json files
):
    run_name = run_name or sys.argv[0].replace(".py", "")
    results_path = f"{run_name}_results.json"
    if os.path.isfile(results_path):
        logging.info(f"Computation results ({results_path}) already exist! Exiting.")
        return import_export.load_from_json(results_path)

    start_time = time.time()
    utils.set_random_seed(random_seed)

    loss_func = getattr(losses, loss_func)
    freeze_training_grid = bool(freeze_training_grid)

    logging.info(f"Configuration: {locals()}")
    config = locals().copy()
    utils.save_configuration(f"{run_name}_configuration.json", config)

    net = SingleHiddenLayerWideNNWithLearnablePriorsAndActivation(
        width=bnn_width, indim=input_n_dims
    )
    print(f"net = {net}")

    activation = activations.get_activation(name=activation)
    activation.to(utils.get_device_name())
    print(
        f"activation = {activation} conditional_params_shapes = {activation.get_conditional_params_shapes()}"
    )

    # Conditioning target (generator with varying hyperparameters)
    if conditioning == "lengthscale":
        #######################################################################
        # Vary properties of taget generator's kernel, data structure is fixed

        # Training input grid generator
        def create_training_grid(n_nodes, **ignored_kwargs):
            return grids.create_sampled_uniform_grid(
                input_data_min_x,
                input_data_max_x,
                n_nodes,
                n_dims=input_n_dims,
            )

        if freeze_training_grid:
            create_training_grid = utils.freeze_function(create_training_grid)

        # Hypernet takes 1D lengthscales generaterd by target_hyperparams_generator
        create_target_generator, target_hyperparams_generator, evaluation = (
            conditioning_on_varying_lengthscale.create_generator_matern_gp,
            conditioning_on_varying_lengthscale.yield_conditioning_matern_gp_lengthscale(),
            conditioning_on_varying_lengthscale.evaluation_matern_gp,
        )

        condition_dimensionality = 1
        hypernet_input_builder = (
            lambda train_grid_x, batch_target, generator_hyperparams: torch.tensor(
                generator_hyperparams["lengthscale"]
            )
        )

    elif conditioning == "lengthscale_with_adaptive_input":
        #######################################################################
        # Vary properties of taget generator's kernel, data structure is fixed

        # Training input grid generator implementing requested (by target_hyperparams_generator) transformations
        def create_training_grid(n_nodes, lengthscale=1.0, input_shift=0.0):
            range_scaling = lengthscale

            data_center = (input_data_max_x + input_data_min_x) * 0.5
            range = (input_data_max_x - input_data_min_x) * range_scaling
            data_min_x = data_center + input_shift - 0.5 * range
            data_max_x = data_center + input_shift + 0.5 * range

            return grids.create_sampled_uniform_grid(
                data_min_x,
                data_max_x,
                n_nodes,
                n_dims=input_n_dims,
            )

        if freeze_training_grid:
            create_training_grid = utils.freeze_function(create_training_grid)

        # Hypernet takes 1D lengthscales generaterd by target_hyperparams_generator
        create_target_generator, target_hyperparams_generator, evaluation = (
            conditioning_on_varying_lengthscale.create_generator_matern_gp,
            conditioning_on_varying_lengthscale.yield_conditioning_matern_gp_lengthscale(
                ls=[
                    0.025,
                    10.0,
                    5.0,
                    0.5,
                    0.25,
                    0.75,
                    0.05,
                    2.5,
                    7.5,
                    0.1,
                    0.01,
                    0.075,
                    1.0,
                ]
            ),
            conditioning_on_varying_lengthscale.evaluation_matern_gp,
        )

        condition_dimensionality = 1
        hypernet_input_builder = (
            lambda train_grid_x, batch_target, generator_hyperparams: torch.tensor(
                generator_hyperparams["lengthscale"]
            )
        )

    elif conditioning == "lengthscale_with_adaptive_shifting_input":
        #######################################################################
        # Vary properties of taget generator's kernel, data structure is fixed

        # Training input grid generator implementing requested (by target_hyperparams_generator) transformations
        def create_training_grid(n_nodes, lengthscale=1.0):
            range_scaling = lengthscale

            data_center = (input_data_max_x + input_data_min_x) * 0.5
            range = input_data_max_x - input_data_min_x
            range *= range_scaling
            # randomly generated positions within the range:
            input_shift = random.uniform(input_data_min_x, input_data_max_x)
            data_min_x = data_center + input_shift - 0.5 * range
            data_max_x = data_center + input_shift + 0.5 * range

            return grids.create_sampled_uniform_grid(
                data_min_x,
                data_max_x,
                n_nodes,
                n_dims=input_n_dims,
            )

        if freeze_training_grid:
            create_training_grid = utils.freeze_function(create_training_grid)

        # Hypernet takes 1D lengthscales generaterd by target_hyperparams_generator
        create_target_generator, target_hyperparams_generator, evaluation = (
            conditioning_on_varying_lengthscale.create_generator_matern_gp,
            conditioning_on_varying_lengthscale.yield_conditioning_matern_gp_lengthscale(
                ls=[
                    0.025,
                    10.0,
                    5.0,
                    0.5,
                    0.25,
                    0.75,
                    0.05,
                    2.5,
                    7.5,
                    0.1,
                    0.01,
                    0.075,
                    1.0,
                ]
            ),
            conditioning_on_varying_lengthscale.evaluation_matern_gp,
        )

        condition_dimensionality = 1
        hypernet_input_builder = (
            lambda train_grid_x, batch_target, generator_hyperparams: torch.tensor(
                generator_hyperparams["lengthscale"]
            )
        )

    elif conditioning == "lengthscale_and_shifting_input":
        #######################################################################
        # Vary properties of taget generator's kernel, data structure is fixed

        # Training input grid generator implementing requested (by target_hyperparams_generator) transformations
        def create_training_grid(n_nodes, lengthscale=1.0):
            range_scaling = lengthscale

            data_center = (input_data_max_x + input_data_min_x) * 0.5
            range = input_data_max_x - input_data_min_x
            range *= range_scaling
            # randomly generated positions within the range:
            input_shift = random.uniform(input_data_min_x, input_data_max_x)
            data_min_x = data_center + input_shift - 0.5 * range
            data_max_x = data_center + input_shift + 0.5 * range

            return grids.create_sampled_uniform_grid(
                data_min_x,
                data_max_x,
                n_nodes,
                n_dims=input_n_dims,
            )

        if freeze_training_grid:
            create_training_grid = utils.freeze_function(create_training_grid)

        # Hypernet takes 1D lengthscales generaterd by target_hyperparams_generator
        create_target_generator, target_hyperparams_generator, evaluation = (
            conditioning_on_varying_lengthscale.create_generator_matern_gp,
            conditioning_on_varying_lengthscale.yield_conditioning_matern_gp_lengthscale(
                ls=[
                    0.025,
                    10.0,
                    5.0,
                    0.5,
                    0.25,
                    0.75,
                    0.05,
                    2.5,
                    7.5,
                    0.1,
                    0.01,
                    0.075,
                    1.0,
                ]
            ),
            None,
        )

        condition_dimensionality = create_training_grid(
            num_function_samples, 0.0
        ).numel()  # extract input dim by creating a sample grid
        condition_dimensionality += 1

        hypernet_input_builder = (
            lambda train_grid_x, batch_target, generator_hyperparams: torch.concat(
                [
                    torch.tensor(generator_hyperparams["lengthscale"])[None],
                    train_grid_x.flatten(),
                ]
            )
        )

    elif conditioning == "transformed_input":
        #######################################################################
        # Vary input data charactersitcs (i.e. shift) but generator is not affected

        # Training input grid generator implementing requested (by target_hyperparams_generator) transformations
        def create_training_grid(n_nodes, input_shift=0.0, range_scaling=1.0):
            data_center = (input_data_max_x + input_data_min_x) * 0.5
            range = (input_data_max_x - input_data_min_x) * range_scaling
            data_min_x = data_center + input_shift - 0.5 * range
            data_max_x = data_center + input_shift + 0.5 * range

            return grids.create_sampled_uniform_grid(
                data_min_x,
                data_max_x,
                n_nodes,
                n_dims=input_n_dims,
            )

        assert (
            not freeze_training_grid
        ), "freeze_training_grid must be False if conditioning on inputs"

        # Hypernet takes transformed inputs
        target_hyperparams_generator, create_target_generator, evaluation = (
            conditioning_on_transformed_input.get_target_hyperparams_generator(
                conditioning_input_transformation
            ),
            conditioning_on_transformed_input.get_create_target_generator_func(),
            conditioning_on_transformed_input.get_evaluation_func(),
        )

        # put full batch as conditioning
        condition_dimensionality = create_training_grid(
            num_function_samples, 0.0
        ).numel()  # extract input dim by creating a sample grid
        hypernet_input_builder = (
            lambda train_grid_x, batch_target, generator_hyperparams: train_grid_x.flatten()
        )

    elif conditioning == "input_shift":
        #######################################################################
        # Vary input data charactersitcs (i.e. shift) but generator is not affected

        # Training input grid generator implementing requested (by target_hyperparams_generator) transformations
        def create_training_grid(n_nodes, input_shift=0.0, range_scaling=1.0):
            data_center = (input_data_max_x + input_data_min_x) * 0.5
            range = (input_data_max_x - input_data_min_x) * range_scaling
            data_min_x = data_center + input_shift - 0.5 * range
            data_max_x = data_center + input_shift + 0.5 * range

            return grids.create_sampled_uniform_grid(
                data_min_x,
                data_max_x,
                n_nodes,
                n_dims=input_n_dims,
            )

        assert (
            not freeze_training_grid
        ), "freeze_training_grid must be False if conditioning on inputs"

        # Hypernet takes transformed inputs
        target_hyperparams_generator, create_target_generator, evaluation = (
            conditioning_on_transformed_input.get_target_hyperparams_generator("shift"),
            conditioning_on_transformed_input.get_create_target_generator_func(),
            conditioning_on_transformed_input.get_evaluation_func(),
        )

        # put full batch as conditioning
        sample_hyperparams = next(target_hyperparams_generator)
        condition_dimensionality = 2
        # extract input dim by creating a sample grid
        range = input_data_max_x - input_data_min_x
        hypernet_input_builder = (
            lambda train_grid_x, batch_target, generator_hyperparams: torch.tensor(
                [
                    generator_hyperparams["input_shift"] - 0.5 * range,
                    generator_hyperparams["input_shift"] + 0.5 * range,
                ]  # data range
            )
        )

    else:
        #######################################################################
        raise ValueError(f"Unsupported conditioning={conditioning}!")

    logging.info(
        "Conditioning settings: "
        f"create_target_generator={create_target_generator} "
        f"target_hyperparams_generator={target_hyperparams_generator} "
        f"evaluation={evaluation}"
    )

    # Hypernet output = activation parameters + priors' parameters
    net_n_parameters = len(list(net.named_parameters()))
    target_param_shapes = activation.get_conditional_params_shapes() + [
        (net_n_parameters * 2,)
    ]
    hypernet = create_conditioning_hypernet(
        indim=condition_dimensionality,
        target_param_shapes=target_param_shapes,
        arch_no=hypernet_arch_no,
    )
    print(f"hypernet = {hypernet}")

    # Fixed input grid for the final evaluation
    final_evaluation_input_grid = grids.create_uniform_grid(
        input_data_min_x,
        input_data_max_x,
        final_evaluation_num_function_samples,
        n_dims=input_n_dims,
    )
    if final_evaluation_on_training_grid:
        if final_evaluation_num_function_samples != num_function_samples:
            raise ValueError(
                "Requested grid evaluation=training, but number of nodes is different!"
            )
        final_evaluation_input_grid = create_training_grid(
            final_evaluation_num_function_samples
        )

    if evaluation:
        print("Performing evaluation before training")
        evaluation(**locals())

    history = train(**locals())

    history = pd.DataFrame(history)
    print(f"training history:\n{history}")
    history.to_csv(run_name + "_history.csv", index=False)

    # Save evaluation and learned parameters to a file
    results = {
        "prior_type": "hypernet",
        "priors_parameters": hypernet,
        "net_width": bnn_width,
        "activation": activation,
        ##########################################
        "hypernet_input_builder": hypernet_input_builder,
        "condition_dimensionality": condition_dimensionality,
        "target_hyperparams_generator": target_hyperparams_generator,
        "create_target_generator": create_target_generator,
        "evaluation_func": evaluation,
        ##########################################
        "evaluation": evaluation(**locals()) if evaluation else None,
        "net": net,
        "history": json.loads((history.to_json(orient="records"))),
        "config": config,
        "script": sys.argv[0],
        "time": time.time() - start_time,
    }
    import_export.save_to_json(results_path, results)
    return results


def main():
    # set up logging
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
        force=True,
    )

    print(
        "Training Bayesian Neural Network (BNN) to match GP prior. "
        "Priors and activations are amortized by hypernetworks. "
        " Target=GPs with Matern kernel with varying hyperparameters (e.g. lengthscales). "
        " This script utilizes a hypernetwork conditioned on the hyperparameters "
        " to dynamically set both the priors and activation functions. "
    )

    parsed_args = args.parse_args()
    utils.setup_torch(
        parsed_args.pop("device", None),
        parsed_args.pop("dtype", None),
        parsed_args.pop("n_threads", None),
    )
    return main_parameterized(**parsed_args)


if __name__ == "__main__":
    main()
