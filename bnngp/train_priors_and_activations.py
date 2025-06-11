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

from metrics import compute_distribution_distances


@torch.no_grad()
def evaluation(
    run_name,
    loss_func,
    final_evaluation_input_grid,
    final_evaluation_batch_size,
    generator,
    learnable_activation,
    net,
    samplers,
    header="final evaluation",
    **ignored_kwargs,
):
    logging.info(f"[evaluation] {header}")

    batch_target = generator(
        final_evaluation_input_grid, n_samples=final_evaluation_batch_size
    ).T

    batch_learned = sample_functions_from_nn(
        net,
        samplers,
        final_evaluation_input_grid,
        learnable_activation,
        n_samples=final_evaluation_batch_size,
    )
    metrics = {"step": header}

    distributional_metrics = compute_distribution_distances(batch_learned, batch_target)
    summary = " ".join(f"{k}={v:.3f}" for k, v in distributional_metrics.items())
    logging.info(
        f"[evaluation] Calculating distributional metrics for priors = {summary}"
    )
    metrics.update({f"eval_{k}": v for k, v in distributional_metrics.items()})

    summary, distributional_stats = function_stats.compute_distributional_stats(
        batch_learned, batch_target
    )
    logging.info(
        f"[evaluation] Calculating distributional stats for priors = {summary}"
    )
    metrics.update({f"eval_{k}": v for k, v in distributional_stats.items()})

    final_loss = loss_func(batch_target, batch_learned).item()
    logging.info(f"[evaluation] Evaluation loss: {final_loss}")
    metrics["final_loss"] = final_loss

    metrics["eval_final_evaluation_batch_size"] = final_evaluation_batch_size
    metrics["eval_samples_generator_target_shape"] = list(batch_target.shape)
    metrics["eval_samples_learned_test_shape"] = list(batch_learned.shape)

    suffix = header.lower().replace(" ", "_").replace(":", "")
    path = run_name + f"_metrics_{suffix}.json"
    logging.info(f"[evaluation] Writing metrics to {path}.")
    f = open(path, "w")
    json.dump(metrics, f)

    return metrics


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
    initial_performance=None,
    run_name=None,
    **ignored_kwargs,
):
    # if ignored_kwargs:
    #     logging.info(f"[train] WARNING: IGNORED KWARGS: {ignored_kwargs}")

    history = (
        [{"iteration": 0, "eta": 0, **initial_performance, "loss": float("nan")}]
        if initial_performance
        else []
    )

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

            evaluation_stats = evaluation(
                header=f"Evaluation at {iteration}", **locals(), **ignored_kwargs
            )
            history[-1].update(evaluation_stats)
                    
            history_df = pd.DataFrame(history)
            history_df.to_csv(f"{run_name}_history.csv", index=False)
            
        gc.collect()

    return history


@torch.no_grad()
def report(
    run_name,
    generator,
    net,
    samplers,
    variational_params,
    learnable_activation,
    batch_target,
    batch_learning,
    loss,
    plotting_grid_x,
    iteration,
    **ignored_kwargs,
):
    loss_wd = losses.wasserstein_distance(batch_target, batch_learning)
    comparison_str, summary_stats = function_stats.compute_distributional_stats(
        batch_learning, batch_target
    )

    logging.info(
        f"[it={iteration}] loss={loss: .3f} wasserstein={loss_wd:.3f}  [target vs learning]: {comparison_str}"
    )

    vp_positive = constraints.ensure_all_normalized(variational_params)
    vp_positive = {n: f"{v:.2f}" for n, v in vp_positive.items()}
    logging.info(f"priors variational_params={vp_positive}")

    if plotting_grid_x is not None and plotting_grid_x.shape[-1] == 1:
        # Plot target samples
        utils.plot_network_functions_1D(
            plotting_grid_x,
            generator,
            n_plots=4,
            color="dodgerblue",
            # title=f"Target functions from {generator}",
        )
        plt.savefig(
            f"{run_name}_iter={iteration}_target.pdf",
            bbox_inches="tight",
            transparent=True,
            pad_inches=0,
        )
        plt.clf()
        # plt.show()

        # Plot network samples
        utils.plot_network_functions_1D_learnable_activation(
            plotting_grid_x,
            learnable_activation,
            wrap_nn_with_parameters_resampling(net, samplers),
            n_plots=4,
            color="orange",
            # title=f"Result (from {net}) after {iteration} iterations",
        )
        plt.savefig(
            f"{run_name}_iter={iteration}_results.pdf",
            bbox_inches="tight",
            transparent=True,
            pad_inches=0,
        )
        plt.clf()
        # plt.show()
        
        # Plot target samples
        utils.plot_network_functions_1D(
            plotting_grid_x,
            generator,
            n_plots=4,
            # title=f"Target functions from {generator}",
        )
        plt.savefig(
            f"{run_name}_iter={iteration}_target2.pdf",
            bbox_inches="tight",
            transparent=True,
            pad_inches=0,
        )
        plt.clf()
        # plt.show()

        # Plot network samples
        utils.plot_network_functions_1D_learnable_activation(
            plotting_grid_x,
            learnable_activation,
            wrap_nn_with_parameters_resampling(net, samplers),
            n_plots=4,
            # title=f"Result (from {net}) after {iteration} iterations",
        )
        plt.savefig(
            f"{run_name}_iter={iteration}_results2.pdf",
            bbox_inches="tight",
            transparent=True,
            pad_inches=0,
        )
        plt.clf()
        # plt.show()
        

    if hasattr(generator, "activation"):
        utils.plot_compare_activations(generator.activation, learnable_activation)
        plt.savefig(f"{run_name}_iter={iteration}_activations.png")
        plt.clf()
        # plt.show()

    for _ in range(10):
        gc.collect()
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


def _create_plotting_grid_x(
    input_data_range_config_set_no,
    test_grid_n_nodes=1024,
    uniform_grid=True,
):
    """Creates a plotting grid for the x-axis based on the specified data range configuration.
    Args:
        input_data_range_config_set_no (int): Identifier for the input data range configuration set.
        test_grid_n_nodes (int, optional): Number of nodes in the test grid. Defaults to 1024.
        uniform_grid (bool, optional): Whether to create a uniform grid.
    """
    plotting_grid_x = None
    if input_data_range_config_set_no is None:
        return plotting_grid_x

    try:
        create_plotting_grid_x = (
            grids.create_uniform_grid
            if uniform_grid
            else grids.create_sampled_uniform_grid
        )
        data_min_x, data_max_x = example_configs.get_data_ranges(
            input_data_range_config_set_no
        )
        plotting_grid_x = create_plotting_grid_x(
            data_min_x,
            data_max_x,
            test_grid_n_nodes,
            n_dims=1,
        )  # used for plotting results
        logging.info(
            f"Plotting grid shape = {plotting_grid_x.shape} (test_grid_n_nodes={test_grid_n_nodes})"
        )
    except Exception as e:
        logging.warning(f"Failed to create plotting grid: {e}")
    return plotting_grid_x


def run_main(
    run_name="",
    generator_width=1_000,
    bnn_width=1_000,
    loss_func="wasserstein_distance",
    create_parameter_sampler="create_factorized_sampler_gaussian_zero_loc",
    activation="rational",
    n_iterations=1_000,
    lr=0.01,
    report_every_n_iterations=200,
    generator_config_set_no=-11,  # generator configuration
    priors_config_set_no=None,  # what configuration should be used to initialize the network, if not set: use generator
    input_data_range_config_set_no=None,  # range of the training input data
    batch_size=512,
    num_function_samples=512,
    shuffle_xs_n_times=1,
    freeze_training_grid=False,
    ##############################
    # Data:
    create_training_grid=None,  # function to create training grid
    sn2=0.1,  # UCI regression likelihood scale parameter
    test_grid_n_nodes=None,  # plotting grid for 1D case
    final_evaluation_input_grid=None,  # grid for final evaluation
    final_evaluation_batch_size=1024,
    ##############################
    random_seed=None,
):
    start_time = time.time()
    utils.set_random_seed(random_seed)

    ###############################################################################################

    # assert priors_config_set_no is not None
    # assert input_data_range_config_set_no is not None
    assert create_training_grid is not None

    if final_evaluation_input_grid is None:
        logging.warning(
            "final_evaluation_input_grid is not set, "
            f"creating one with create_training_grid with using #data-pts={num_function_samples}"
        )
        final_evaluation_input_grid = create_training_grid(num_function_samples)

    ###############################################################################################
    # Retrieve objects from strings

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

    # Save configuration
    logging.info(f"Configuration: {locals()}")
    config = locals().copy()
    utils.save_configuration(f"{run_name}_configuration.json", config)

    learnable_activation = activations.get_activation(name=activation)
    learnable_activation.to(utils.get_device_name())

    input_n_dims = create_training_grid(1).shape[-1]

    ###############################################################################################
    # Prepare a method for generating target functions which we want to learn

    generator = example_configs.get_generator(
        config_set_no=generator_config_set_no,
        input_dim=input_n_dims,
        net_width=generator_width,  # used when generator is a neural network
        sn2=sn2,  # used by AllYouNeedUCIRegression
    )

    ###############################################################################################
    # Grid for plotting

    plotting_grid_x = _create_plotting_grid_x(
        input_data_range_config_set_no,
        (test_grid_n_nodes or num_function_samples),
        # uniform_training_grid,
    )

    ###############################################################################################
    # Prepare single-hidden-layer network and samplers for its parameters

    net = SingleHiddenLayerWideNNWithLearnablePriorsAndActivation(
        width=bnn_width, indim=input_n_dims
    )
    # fixes priors on weights if not trained; used by create_gaussian_priors
    priors_params_fixed = example_configs.get_parameters_for_priors_on_weights(
        priors_config_set_no=priors_config_set_no
    )
    samplers, variational_params, aux_objs = create_parameter_sampler(
        net, init=priors_params_fixed._asdict()
    )

    ###############################################################################################
    # Train variational parameters
    initial_performance = evaluation(**locals(), header="Evaluation before training:")
    history = train(**locals())

    history = pd.DataFrame(history)
    logging.info(f"training history:\n{history}")
    history.to_csv(f"{run_name}_history.csv", index=False)

    ###############################################################################################
    # Save evaluation and learned parameters
    net.learnable_activation = learnable_activation
    results = {
        "prior_type": "gaussian",
        "priors_parameters": constraints.ensure_all_normalized(
            {**aux_objs, **variational_params}
        ),  # by default we load the values from variational_params, but if weight priors are fixed we use aux_objs
        "net_width": bnn_width,
        "activation": learnable_activation,
        "evaluation": evaluation(**locals()),
        "initial_performance": initial_performance,
        "generator": generator,
        "net": net,
        "samplers": samplers,
        "history": json.loads((history.to_json(orient="records"))),
        "config": config,
        "script": sys.argv[0],
        "time": time.time() - start_time,
    }
    results["posterior_predictive"] = wrap_nn_with_parameters_resampling(net, samplers)
    return results


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
    config_set_no=-11,  # generator configuration
    priors_config_set_no=None,  # what configuration should be used to initialize the network, if not set: use generator
    input_type="sampled_grid",  # what training data should be used
    input_data_range_config_set_no=None,  # range of the training input data
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

    ###############################################################################################
    # Update config numbers if not set

    if input_data_range_config_set_no is None:
        logging.warning(
            f"input_data_range_config_set_no is not set, using config_set_no={config_set_no}"
        )
        input_data_range_config_set_no = config_set_no

    if priors_config_set_no is None:
        logging.warning(
            f"priors_config_set_no is not set, using config_set_no={config_set_no}"
        )
        priors_config_set_no = config_set_no

    ###############################################################################################
    # Preparation of X-s generator
    if uci_training_grid:
        logging.warning("Because uci_training_grid=True, I force input_type=uci!")
        input_type = "uci"

    if input_type == "sampled_grid":
        data_min_x, data_max_x = example_configs.get_data_ranges(
            input_data_range_config_set_no
        )
        create_training_grid = prepare_input_grid_sampled_hypercube_create_func(
            input_n_dims,
            data_min_x,
            data_max_x,
            uniform_training_grid,
        )
        final_evaluation_input_grid = grids.create_uniform_grid(
            data_min_x,
            data_max_x,
            final_evaluation_num_function_samples,
            n_dims=input_n_dims,
        )

    elif input_type == "transformed_input":
        data_min_x, data_max_x = example_configs.get_data_ranges(
            input_data_range_config_set_no
        )
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
        # extract data subset based on config_set_no
        assert (
            config_set_no >= 2000
        ), "For input_type=uci config_set_no>=2000 is allowed!"
        split_no = int(config_set_no - 2000)

        logging.info(f"Loading UCI input: dataset={dataset}, split_no={split_no}")
        uci_generators = utils.UCIGenerators(dataset)

        create_training_grid = lambda n_nodes: grids.create_sampled_uci_grid(
            n_nodes,
            split_no=split_no,
            uci_generators=uci_generators,
        )
        final_evaluation_input_grid = create_training_grid(
            final_evaluation_num_function_samples
        )

    else:
        raise ValueError(f"Unknown input type = {input_type}!")

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
    # Check if the input grid has the correct number of dimensions
    inferred_input_n_dims = create_training_grid(1).shape[-1]
    assert (
        input_n_dims == inferred_input_n_dims
    ), f"input_n_dims={input_n_dims} != inferred_input_n_dims={inferred_input_n_dims}"

    ###############################################################################################

    freeze_training_grid = bool(freeze_training_grid)
    if freeze_training_grid:
        logging.info("Training grid will be frozen")
        create_training_grid = utils.freeze_function(create_training_grid)
    else:
        logging.info("Training grid will be recreated in every epoch")

    ###############################################################################################

    results = run_main(
        run_name=run_name,
        generator_width=generator_width,
        bnn_width=bnn_width,
        loss_func=loss_func,
        create_parameter_sampler=create_parameter_sampler,
        activation=activation,
        n_iterations=n_iterations,
        lr=lr,
        report_every_n_iterations=report_every_n_iterations,
        generator_config_set_no=config_set_no,
        priors_config_set_no=priors_config_set_no,
        input_data_range_config_set_no=input_data_range_config_set_no,
        batch_size=batch_size,
        num_function_samples=num_function_samples,
        shuffle_xs_n_times=shuffle_xs_n_times,
        create_training_grid=create_training_grid,
        sn2=sn2,
        final_evaluation_input_grid=final_evaluation_input_grid,
        final_evaluation_batch_size=final_evaluation_batch_size,
        test_grid_n_nodes=test_grid_n_nodes,
        random_seed=random_seed,
    )

    results = import_export.save_to_json(results_path, results)
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
