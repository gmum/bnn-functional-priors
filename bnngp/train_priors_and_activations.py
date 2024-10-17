import torch
import pandas as pd
import tqdm
from matplotlib import pyplot as plt
import os
import time
import json
import sys
import gc

import activations

import losses
import function_stats
from examples import configs as example_configs
from bnn import SingleHiddenLayerWideNNWithLearnablePriorsAndActivation
import weights_sampling
from weights_sampling import (
    wrap_nn_with_parameters_resampling,
    sample_functions_from_nn,
)
import weights_sampling.constraints as constraints
import conditioning_on_transformed_input
import import_export
import grids
import utils
import args

import logging


@torch.no_grad()
def evaluation(
    loss_func,
    final_evaluation_input_grid,
    final_evaluation_batch_size,
    generator,
    learnable_activation,
    net,
    samplers,
    **ignored_kwargs,
):
    logging.info("final evaluation")

    # gt_target = torch.hstack(
    #     [
    #         generator(final_evaluation_input_grid)
    #         for _ in range(final_evaluation_batch_size)
    #     ]
    # ).T
    gt_target = generator(
        final_evaluation_input_grid, n_samples=final_evaluation_batch_size
    ).T

    final_test = sample_functions_from_nn(
        net,
        samplers,
        final_evaluation_input_grid,
        learnable_activation,
        n_samples=final_evaluation_batch_size,
    )

    final_loss = loss_func(gt_target, final_test).item()
    logging.info(f"final evaluation loss:\n{final_loss}")
    return final_loss


def train(
    generator,
    net,
    samplers,
    variational_params,
    learnable_activation,
    loss_func,
    create_training_grid,
    batch_size,
    num_function_samples,
    shuffle_xs_n_times,
    n_iterations,
    lr,
    report_every_n_iterations,
    **ignored_kwargs,
):
    # if ignored_kwargs:
    #     logging.info(f"[train] WARNING: IGNORED KWARGS: {ignored_kwargs}")

    history = []
    # TODO - maybe change to AdamW?
    priors_variational_params = list(
        p for p in variational_params.values() if p.requires_grad
    )
    activation_variational_params = list(
        p for p in learnable_activation.parameters() if p.requires_grad
    )

    logging.info(
        "Optimizing "
        f"{len(priors_variational_params)} priors_variational_params "
        f"(= {sum(v.numel() for v in priors_variational_params)} elements) and "
        f"{len(activation_variational_params)} activation_variational_params "
        f"(= {sum(v.numel() for v in activation_variational_params)} elements)"
    )
    optimizer = torch.optim.Adam(
        priors_variational_params + activation_variational_params,
        lr=lr,
    )
    start = time.time()
    for iteration in tqdm.tqdm(range(n_iterations)):
        loss = 0.0
        for _ in range(shuffle_xs_n_times):
            ###########################################################################################################

            train_grid_x = create_training_grid(num_function_samples)
            batch_target = generator(train_grid_x, n_samples=batch_size).T

            # prepare a sample from the trained network
            batch_learning = sample_functions_from_nn(
                net, samplers, train_grid_x, learnable_activation, n_samples=batch_size
            )

            loss += loss_func(batch_target, batch_learning)

            ############################################################################################################
        loss /= shuffle_xs_n_times

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append(
            {
                "iteration": iteration,
                "loss": float(loss.cpu()),
                "eta": time.time() - start,
            }
        )

        if iteration % report_every_n_iterations == 0:
            reported_stats = report(**locals(), **ignored_kwargs)
            history[-1].update(reported_stats)

        gc.collect()

    return history


@torch.no_grad()
def report(
    run_name,
    generator,
    net,
    samplers,
    variational_params,
    activation,
    learnable_activation,
    batch_target,
    batch_learning,
    loss,
    plotting_grid_x,
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

    summary_stats = {
        "wasserstein_distance": loss_wd.item(),
    }
    for k, (m, s) in characteristics_learning.items():
        summary_stats["learning_" + k + "_mean"] = m.item()
        summary_stats["learning_" + k + "_std"] = s.item()
    for k, (m, s) in characteristics_target.items():
        summary_stats["target_" + k + "_mean"] = m.item()
        summary_stats["target_" + k + "_std"] = s.item()

    vp_positive = constraints.ensure_all_normalized(variational_params)
    vp_positive = {n: f"{v:.2f}" for n, v in vp_positive.items()}
    logging.info(f"priors variational_params={vp_positive}")

    # Plot target samples
    utils.plot_network_functions_1D(
        plotting_grid_x,
        generator,
        title=f"Target functions from {generator}",
    )
    plt.savefig(f"{run_name}_iter={iteration}_target.png")
    plt.clf()
    # plt.show()

    # Plot network samples
    utils.plot_network_functions_1D_learnable_activation(
        plotting_grid_x,
        learnable_activation,
        wrap_nn_with_parameters_resampling(net, samplers),
        title=f"Result (from {net}) after {iteration} iterations",
    )
    plt.savefig(f"{run_name}_iter={iteration}_results.png")
    plt.clf()
    # plt.show()

    utils.plot_compare_activations(activation, learnable_activation)
    plt.savefig(f"{run_name}_iter={iteration}_activations.png")
    plt.clf()
    # plt.show()

    return summary_stats


def prepare_input_grid_sampled_hypercube_create_func(
    input_n_dims,
    input_data_min_x,
    input_data_max_x,
    uniform_training_grid=False,
):
    """Samples inputs from hypercube.

    Returns:
        Function which takes the desired number of nodes and produces a new X.
    """

    create_grid_func = (
        grids.create_uniform_grid
        if uniform_training_grid
        else grids.create_sampled_uniform_grid
    )
    logging.info(f"Training grid will be created using {create_grid_func}")

    return lambda n_nodes: create_grid_func(
        data_min_x=input_data_min_x,
        data_max_x=input_data_max_x,
        n_nodes=n_nodes,
        n_dims=input_n_dims,
    )


def prepare_input_grid_transformed_hypercube_create_func(
    input_n_dims,
    input_data_min_x,
    input_data_max_x,
    uniform_training_grid=False,
    transformations=conditioning_on_transformed_input.yield_input_random_shifts(),
):
    """Samples inputs from hypercube, which is translated (shifted) and scaled for each iteration.

    Returns:
        Function which takes the desired number of nodes and produces a new X.
    """

    def create_grid_func(n_nodes, input_shift=0.0, range_scaling=1.0):
        logging.debug(
            f"Generating data with input_shift={input_shift} range_scaling={range_scaling}"
        )
        data_center = (input_data_max_x + input_data_min_x) * 0.5
        range = (input_data_max_x - input_data_min_x) * range_scaling
        data_min_x = data_center + input_shift - 0.5 * range
        data_max_x = data_center + input_shift + 0.5 * range

        return grids.create_sampled_uniform_grid(
            data_min_x=data_min_x,
            data_max_x=data_max_x,
            n_nodes=n_nodes,
            n_dims=input_n_dims,
        )

    logging.info(f"Training grid will be created using {create_grid_func}")

    return lambda n_nodes: create_grid_func(n_nodes, **next(transformations))


def prepare_uci_input_create_func(config_set_no, dataset, test_set_size):
    """Samples UCI data."""
    assert config_set_no >= 2000, "For input_type=uci config_set_no>=2000 is allowed!"
    split_no = int(config_set_no - 2000)
    logging.info(f"Loading UCI input: dataset={dataset}, split_no={split_no}")
    uci_generators = utils.UCIGenerators(dataset)

    create = lambda n_nodes: grids.create_sampled_uci_grid(
        n_nodes,
        split_no=split_no,
        uci_generators=uci_generators,
    )
    return create, create(test_set_size)


def main_parameterized(
    run_name="",
    generator_width=1_000,
    bnn_width=1_000,
    loss_func="wasserstein_distance",
    create_parameter_sampler="create_factorized_sampler_gaussian_zero_loc",
    activation="rational",
    n_iterations=1_000,
    lr=0.01,
    report_every_n_iterations=200,
    config_set_no=-11,
    priors_config_set_no=None,  # what configuration should be used to initialize the network, if not set: use the same as for generator
    input_type="sampled_grid",  # what training data should be used
    input_n_dims=1,
    batch_size=512,
    num_function_samples=512,
    shuffle_xs_n_times=1,
    uniform_training_grid=False,  # if False the input grid is sampled on a hypercube
    freeze_training_grid=False,
    ##############################
    # Training on UCI data:
    uci_training_grid=False,  # if True, the training grid would be generated following "All you need" paper procedure
    dataset="boston",  # which UCI dataset
    sn2=0.1,  # UCI regression likelihood scale parameter
    ##############################
    final_evaluation_on_training_grid=False,
    final_evaluation_batch_size=1024,
    final_evaluation_num_function_samples=512,
    random_seed=None,
    test_grid_n_nodes=None,  # plotting grid for 1D case
    description={},  # additional info to be stored along with data and results in json files
    force_recomputing=False,
):
    run_name = run_name or sys.argv[0].replace(".py", "")
    results_path = f"{run_name}_results.json"
    if os.path.isfile(results_path) and not force_recomputing:
        logging.info(f"Computation results ({results_path}) already exist! Exiting.")
        return import_export.load_from_json(results_path)

    start_time = time.time()
    utils.set_random_seed(random_seed)

    ###############################################################################################
    loss_func = getattr(losses, loss_func)
    sampler_renaming = {
        "create_factorized_shared_gaussian_sampler_zero_loc": "create_factorized_sampler_gaussian_zero_loc",
        "create_factorized_shared_gaussian_sampler": "create_factorized_sampler_gaussian",
        "create_factorized_shared_invgamma_gaussian_mixture_sampler": "create_factorized_sampler_invgamma_gaussian_mixture",
        "create_factorized_shared_tstudent_sampler": "create_factorized_tstudent_sampler",
    }  # backward compatibility
    create_parameter_sampler = sampler_renaming.get(
        create_parameter_sampler, create_parameter_sampler
    )
    create_parameter_sampler = getattr(weights_sampling, create_parameter_sampler)
    freeze_training_grid = bool(freeze_training_grid)

    logging.info(f"Configuration: {locals()}")
    config = locals().copy()
    utils.save_configuration(f"{run_name}_configuration.json", config)

    learnable_activation = activations.get_activation(name=activation)
    learnable_activation.to(utils.get_device_name())

    ###############################################################################################
    (generator, meta, data_min_x, data_max_x,) = example_configs.get_configs(
        config_set_no=config_set_no,
        net_width=generator_width,
        input_dim=input_n_dims,
        sn2=sn2,
    )
    activation = meta.activation

    if priors_config_set_no:  # override generator configuration
        logging.info(
            f"Overriding initial configuration of priors: {config_set_no} -> {priors_config_set_no}"
        )
        meta = example_configs.get_configs(
            config_set_no=priors_config_set_no,
            net_width=bnn_width,
            input_dim=input_n_dims,
            sn2=sn2,
        )[1]

    ###############################################################################################
    create_plotting_grid_x = (
        grids.create_uniform_grid
        if uniform_training_grid
        else grids.create_sampled_uniform_grid
    )
    test_grid_n_nodes = test_grid_n_nodes or num_function_samples
    plotting_grid_x = create_plotting_grid_x(
        data_min_x,
        data_max_x,
        test_grid_n_nodes,
        n_dims=input_n_dims,
    )  # used for plotting results
    logging.info(
        f"Plotting grid shape = {plotting_grid_x.shape} (test_grid_n_nodes={test_grid_n_nodes})"
    )

    ###############################################################################################
    # Training input generator preparation
    if uci_training_grid:
        logging.warning("Because uci_training_grid=True, I force input_type=uci!")
        input_type = "uci"

    if input_type == "sampled_grid":
        create_training_grid = prepare_input_grid_sampled_hypercube_create_func(
            input_n_dims,
            data_min_x,
            data_max_x,
            uniform_training_grid,
        )
        final_evaluation_input_grid = grids.create_sampled_uniform_grid(
            data_min_x,
            data_max_x,
            final_evaluation_num_function_samples,
            n_dims=input_n_dims,
        )

    elif input_type == "transformed_input":
        assert (
            uniform_training_grid == False
        ), "input=transformed_input does not support uniform_training_grid!"
        create_training_grid = prepare_input_grid_transformed_hypercube_create_func(
            input_n_dims,
            data_min_x,
            data_min_x,
        )
        final_evaluation_input_grid = grids.create_uniform_grid(
            data_min_x,
            data_max_x,
            final_evaluation_num_function_samples,
            n_dims=input_n_dims,
        )

    elif input_type == "uci":
        (
            create_training_grid,
            final_evaluation_input_grid,
        ) = prepare_uci_input_create_func(
            config_set_no, dataset, final_evaluation_num_function_samples
        )

    else:
        raise ValueError(f"Unknown input type = {input_type}!")

    if freeze_training_grid:
        logging.info("Training grid will be frozen")
        create_training_grid = utils.freeze_function(create_training_grid)
    else:
        logging.info("Training grid will be recreated in every epoch")

    ###############################################################################################
    # Fixed input grid for the final evaluation
    if final_evaluation_on_training_grid:
        if final_evaluation_num_function_samples != num_function_samples:
            raise ValueError(
                "Requested grid evaluation=training, but number of nodes is different!"
            )
        logging.info(
            f"Creating the final evaluation grid using training grid func = {create_training_grid}"
        )
        final_evaluation_input_grid = create_training_grid(
            final_evaluation_num_function_samples
        )

    ###############################################################################################
    inferred_input_n_dims = create_training_grid(1).shape[-1]
    assert (
        input_n_dims == inferred_input_n_dims
    ), f"input_n_dims={input_n_dims} != inferred_input_n_dims={inferred_input_n_dims}"

    net = SingleHiddenLayerWideNNWithLearnablePriorsAndActivation(
        width=bnn_width, indim=input_n_dims
    )
    samplers, variational_params, aux_objs = create_parameter_sampler(
        net, init=meta._asdict()
    )

    ###############################################################################################
    # Train variational parameters
    history = train(**locals())

    history = pd.DataFrame(history)
    logging.info(f"training history:\n{history}")
    history.to_csv(f"{run_name}_history.csv", index=False)

    ###############################################################################################
    # Save evaluation and learned parameters to a file
    net.learnable_activation = learnable_activation
    results = {
        "prior_type": "gaussian",
        "priors_parameters": constraints.ensure_all_normalized(variational_params),
        "net_width": bnn_width,
        "activation": learnable_activation,
        "evaluation": evaluation(**locals()),
        "generator": generator,
        "net": net,
        "samplers": samplers,
        "history": json.loads((history.to_json(orient="records"))),
        "config": config,
        "script": sys.argv[0],
        "time": time.time() - start_time,
    }
    results = import_export.save_to_json(results_path, results)

    results["posterior_predictive"] = wrap_nn_with_parameters_resampling(net, samplers)
    return results


def main():
    # set up logging
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
        force=True,
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
