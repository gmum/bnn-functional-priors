#!/usr/bin/env python
# coding: utf-8


import logging
import sys

import torch
from torch import nn

import numpy as np
import math
import matplotlib.pylab as plt


import datetime
import os
import gc


import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from pyro.infer import Predictive


import args
import targets_gp
import import_export
import utils

from train_priors_and_activations import main_parameterized as train_priors
from metrics import compute_distribution_distances

from bnn import SingleHiddenLayerWide1DRegressionNNWithGaussianPriors


def get_data():
    util_set_seed(1)

    N = 64
    M = 100
    a, b = -10, 10

    # Generate data
    X = np.random.rand(N, 1) * (b - a) + a
    make_random_gap(X, gap_ratio=0.4)
    y = gp_sample(X, ampl=1.6, leng=1.8)
    Xtest = np.linspace(a - 5, b + 5, M).reshape(-1, 1)

    return X, y, Xtest


def make_random_gap(X, gap_ratio=0.2):
    a, b = X.min(), X.max()
    gap_a = a + np.random.rand() * (b - a) * (1 - gap_ratio)
    gap_b = gap_a + (b - a) * gap_ratio
    idx = np.logical_and(gap_a < X, X < gap_b)
    if gap_a - a > b - gap_b:
        X[idx] = a + np.random.rand(idx.sum()) * (gap_a - a)
    else:
        X[idx] = gap_b + np.random.rand(idx.sum()) * (b - gap_b)


def gp_sample(X, ampl=1, leng=1, sn2=0.1):
    n, x = X.shape[0], X / leng
    sum_xx = np.sum(x * x, 1).reshape(-1, 1).repeat(n, 1)
    D = sum_xx + sum_xx.transpose() - 2 * np.matmul(x, x.transpose())
    C = ampl**2 * np.exp(-0.5 * D) + np.eye(n) * sn2
    return np.random.multivariate_normal(np.zeros(n), C).reshape(-1, 1)


def plot_samples(
    X, samples, var=None, n_keep=12, color="xkcd:bluish", smooth_q=False, ax=None
):
    if ax is None:
        ax = plt.gca()
    if samples.ndim > 2:
        samples = samples.squeeze()
    n_keep = int(samples.shape[1] / 10) if n_keep is None else n_keep
    keep_idx = np.random.permutation(samples.shape[1])[:n_keep]
    mu = samples.mean(1)
    if var is None:
        q = 97.72  ## corresponds to 2 stdevs in Gaussian
        # q = 99.99  ## corresponds to 3 std
        Q = np.percentile(samples, [100 - q, q], axis=1)
        # ub, lb = Q[1,:], Q[0,:]
        ub, lb = mu + 2 * samples.std(1), mu - 2 * samples.std(1)
    else:
        ub = mu + 3 * np.sqrt(var)
        lb = mu - 3 * np.sqrt(var)
    ####
    ax.fill_between(X.flatten(), ub, lb, color=color, alpha=0.25, lw=0)
    ax.plot(X, samples[:, keep_idx], color=color, alpha=0.8)
    ax.plot(X, mu, color="xkcd:red")


def zscore_normalization(X, mean=None, std=None, eps=1e-10):
    """Apply z-score normalization on a given data.

    Args:
        X: numpy array, shape [batchsize, num_dims], the input dataset.
        mean: numpy array, shape [num_dims], the given mean of the dataset.
        var: numpy array, shape [num_dims], the given variance of the dataset.

    Returns:
        tuple: the normalized dataset and the resulting mean and variance.
    """
    if X is None:
        return None, None, None

    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)

    X_normalized = (X - mean) / (std + eps)

    return X_normalized, mean, std


def zscore_unnormalization(X_normalized, mean, std):
    """Unnormalize a given dataset.

    Args:
        X_normalized: numpy array, shape [batchsize, num_dims], the
            dataset needs to be unnormalized.
        mean: numpy array, shape [num_dims], the given mean of the dataset.
        var: numpy array, shape [num_dims], the given variance of the dataset.

    Returns:
        numpy array, shape [batch_size, num_dims] the unnormalized dataset.
    """
    return X_normalized * std + mean


def unnormalize_predictions(pred_mean, pred_var, y_mean, y_std):
    """Unnormalize the regression predictions.

    Args:
        pred_mean: np.array, [n_data, 1], the predictive mean.
        pred_var: np.array, [n_data, 1], the predictive variance.
        y_mean: np.array, [n_data, 1], the mean estimated from training data.
        y_std: np.array, [n_data, 1], the std estimated from training data.
    """
    pred_mean = zscore_unnormalization(pred_mean, y_mean, y_std)
    pred_var = pred_var * (y_std**2)

    return pred_mean, pred_var


def util_set_seed(seed=99):
    """Set seed for reproducibility purpose."""
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def rmse(y_pred, y):
    """Calculates the root mean squared error.

    Args:
        y_pred: numpy array, shape [batch_size], the predictions.
        y: numpy array, shape [batch_size], the corresponding labels.

    Returns:
        rmse: float, the resulting root mean squared error.
    """
    y_pred, y = y_pred.squeeze(), y.squeeze()
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))

    return float(rmse)




def prior_predictive_sampling(model, input_tensor, num_samples=119):
    predictive = pyro.infer.Predictive(model, num_samples=num_samples)
    samples = predictive(input_tensor)
    return samples


def main_parameterized(
    run_name="",
    random_seed=123,
    # prior learning:
    train_prior=True,
    prior_n_iterations=2001,
    prior_num_function_samples=512,  # 512  # = n_data=200 in the AllYouNeed notebook
    prior_batch_size=512,
    prior_activation="nn_1_5",
    # posterior learning:
    posterior_net_width=1_000,
    posterior_mcmc_n_samples=1000,
    posterior_mcmc_num_chains=4,  # you cannot use more than 1 in notebooks
    # regression model params
    sn2=0.1,  # noise variance
    **fit_prior_kwargs,
):

    # ## Init
    prior_str = "trained_priors" if train_prior else "default_priors"
    target_dir = (
        f"pretrained/e2e_allyouneed_1D_regression_{prior_str}_{run_name}_"
        + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    )
    results_prefix = target_dir + "/"

    logging.info(f"Saving results to target_dir = {target_dir}")

    if target_dir is not None and target_dir != "":
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

    dtype = torch.get_default_dtype()

    likelihood_scale = math.sqrt(sn2)

    ############################################################################
    # Data:

    X, y, Xtest = get_data()

    logging.info("Normalize the dataset (normalized values have _ suffix)")
    X_, X_mean, X_std = zscore_normalization(X)
    y_, y_mean, y_std = zscore_normalization(y)

    Xtest_, _, _ = zscore_normalization(Xtest, X_mean, X_std)
    Xtest_tensor_ = torch.from_numpy(Xtest_)

    logging.info(
        f"Transformations in X: {X_mean} +/- {X_std} Transformations in Y: {y_mean} +/- {y_std}"
    )

    plt.figure()
    plt.plot(X, y, "ko", ms=5)
    plt.title("Dataset")
    plt.savefig(results_prefix + "dataset.png")
    plt.clf()

    plt.figure()
    plt.plot(X_, y_, "ko", ms=5)
    plt.title("Dataset (transformed by Z-score)")
    plt.savefig(results_prefix + "dataset_normalized.png")
    plt.clf()

    ############################################################################
    # ## Targets (prior and posterior samples from a GP)
    gpmodel = targets_gp.AllYouNeed1DRegression()

    # Draw functions from the priors
    n_plot = 4000
    util_set_seed(8)

    gp_samples = gpmodel(Xtest_tensor_, n_plot).detach().cpu().numpy().squeeze()
    gp_samples = zscore_unnormalization(gp_samples, y_mean, y_std)
    gp_samples_tensor = torch.from_numpy(gp_samples).T

    # Posterior inference (for evaluation)
    gp_preds = gpmodel.gpmodel.predict_f_samples(Xtest_tensor_, n_plot)
    gp_preds = gp_preds.detach().cpu().numpy().squeeze()
    gp_preds = zscore_unnormalization(gp_preds, y_mean, y_std)

    gp_posterior_samples_tensor = torch.from_numpy(gp_preds).T

    # Evaluation: prior and posterior samples from GP

    metrics_batch_size = 1024  # == number of functions
    evaluation_batch_priors = gp_samples_tensor[:metrics_batch_size, ...]
    evaluation_batch_posteriors = gp_posterior_samples_tensor[:metrics_batch_size, ...]

    assert evaluation_batch_priors.shape == torch.Size([metrics_batch_size, Xtest_tensor_.shape[0]]), f"Expected dims = (number of functions, grid elements) got {evaluation_batch_priors.shape}"
    assert evaluation_batch_posteriors.shape == torch.Size([metrics_batch_size, Xtest_tensor_.shape[0]]), f"Expected dims = (number of functions, grid elements) got {evaluation_batch_posteriors.shape}"

    plot_samples(Xtest, gp_samples, ax=plt.gca(), n_keep=5)
    plt.savefig(results_prefix + "target_gp_samples.png")
    plt.clf()

    ############################################################################
    # ## Training priors

    if train_prior:
        fit = train_priors(
            random_seed=random_seed,
            bnn_width=posterior_net_width,
            config_set_no=1001,  # AllYouNeed setting from 1D_regression_gaussian_priors.ipynb
            uniform_training_grid=False,  # AllYouNeed they have True
            n_iterations=prior_n_iterations,
            num_function_samples=prior_num_function_samples,
            batch_size=prior_batch_size,
            activation=prior_activation,
            run_name=results_prefix + "prior_fit",
            force_recomputing=True,
            final_evaluation_batch_size=256,
            final_evaluation_num_function_samples=256,
            **fit_prior_kwargs,
        )
        gc.collect()

        # Note use of Xtest_ vs Xtest
        logging.info(
            f"Xtest_ vs Xtest = {Xtest.min()}-{Xtest.max()} vs. {Xtest_.min()}-{Xtest_.max()}"
        )

        net = fit["posterior_predictive"]
        net_samples = torch.stack(
            [net(torch.tensor(Xtest_, dtype=dtype)) for _ in range(metrics_batch_size)]
        )
        net_samples = net_samples.squeeze(-1).T
        net_samples = net_samples.cpu().detach().numpy()
        net_samples = zscore_unnormalization(net_samples, y_mean, y_std)

        # Evaluation of prior samples:
        prediction_batch_priors = torch.from_numpy(net_samples).T

        # comparison
        metrics_priors = compute_distribution_distances(prediction_batch_priors, evaluation_batch_priors)
        logging.info(f"Calculating distributional metrics for priors after training = {metrics_priors}")
        utils.save_configuration(results_prefix + "metrics_priors_after_training.json", metrics_priors)

        plot_samples(
            Xtest, net_samples, ax=plt.gca(), color="xkcd:yellowish orange", n_keep=5
        )
        plt.title("Samples from the trained prior")
        plt.savefig(results_prefix + "prior_samples.png")
        plt.clf()

    ############################################################################
    # ## Finding posterior

    # Set Pyro random seed
    pyro.set_rng_seed(random_seed)

    if train_prior:
        parameters = import_export.import_parameters(
            fit["parameters"], target_net_width=posterior_net_width
        )

    else:
        parameters = {
            "layer1.weight": dist.Normal(0.0, 1.0),
            "layer1.bias": dist.Normal(0.0, 1.0),
            "layer2.weight": dist.Normal(0.0, 1.0 / np.sqrt(posterior_net_width)),
            # "layer2.weight": dist.Normal(0., 1.),  # NOTE: uncomment to enable overconfidence for ReLU !!!
            "layer2.bias": dist.Normal(0.0, 1.0),
            "activation": nn.Tanh(),
        }

    logging.info(f"Creating a BNN from parameters = {parameters}")
    bnn_model = SingleHiddenLayerWide1DRegressionNNWithGaussianPriors(
        net_width=posterior_net_width,
        parameters=parameters,
        likelihood_scale=likelihood_scale,
    )

    ############################################################################

    plt.plot(
        torch.arange(-10, 10, 0.05), bnn_model.activation(torch.arange(-10, 10, 0.05))
    )
    plt.grid(True)
    plt.title("used model activation")
    plt.savefig(results_prefix + "model_activation.png")
    plt.clf()

    ############################################################################
    logging.info("Sampling from prior predictive (PyroModule)")

    samples = prior_predictive_sampling(
        bnn_model, Xtest_tensor_.type(dtype), num_samples=metrics_batch_size
    )
    net_samples = samples["mu"].squeeze(1).T
    net_samples = zscore_unnormalization(net_samples, y_mean, y_std)

    prediction_batch_prior_predictive = net_samples.T

    # comparison
    metrics_priors = compute_distribution_distances(prediction_batch_prior_predictive, evaluation_batch_priors)
    logging.info(f"Calculating distributional metrics for priors after training (PyroModule) = {metrics_priors}")
    utils.save_configuration(results_prefix + "metrics_priors_after_training_pyro.json", metrics_priors)

    plot_samples(
        Xtest, net_samples.detach().cpu().numpy(), ax=plt.gca(), color="xkcd:yellowish orange", n_keep=5
    )
    plt.title("Samples from the trained prior loaded to PyroModule")
    plt.savefig(results_prefix + "prior_samples_pyro.png")
    plt.clf()

    ############################################################################

    logging.info("Posterior training is done on normalized data")
    train_x, train_y = torch.tensor(X_, dtype=dtype), torch.tensor(y_, dtype=dtype)

    logging.info(
        f"train_x ({train_x.shape}) = [{train_x.min()}, {train_x.max()}], "
        f"train_y ({train_y.shape}) = [{train_y.min()}, {train_y.max()}]"
    )

    predictive = Predictive(model=bnn_model, num_samples=111)
    preds = predictive(train_x, train_y.type(dtype))

    logging.info(f"Predictive log likelihood shape: {preds['log_likelihood'].shape}")
    logging.info(
        f"Predictive log likelihood of the training data: [TODO fix aggregation] {preds['log_likelihood'].mean()}"
    )

    rmse = torch.math.sqrt(
        ((preds["mu"].squeeze(1).mean(0).unsqueeze(-1) - train_y) ** 2).sum()
    )
    logging.info(f"RMSE {rmse}")

    ############################################################################

    # Define Hamiltonian Monte Carlo (HMC) kernel

    # NUTS = "No-U-Turn Sampler" (https://arxiv.org/abs/1111.4246), gives HMC an adaptive step size
    nuts_kernel = NUTS(
        bnn_model, jit_compile=True
    )  # jit_compile=True is faster but requires PyTorch 1.6+

    # Define MCMC sampler, get MC posterior samples
    mcmc = MCMC(
        nuts_kernel,
        num_samples=posterior_mcmc_n_samples,
        num_chains=posterior_mcmc_num_chains,
    )  # , disable_validation=True)

    # Run MCMC
    mcmc.run(train_x, train_y)

    # Print summary statistics
    mcmc.summary(prob=0.8)  # This gives you the 80% credible interval by default

    try:
        logging.info(f"MCMC diagnostics: {mcmc.diagnostics()}")
    except Exception as e:
        logging.info(f"MCMC Failed diagnostics: {e}")

    try:

        # Check the convergence using R-hat (potential scale reduction factor)
        # R-hat values close to 1.0 suggest convergence
        rhats = pyro.infer.mcmc.util.summary(mcmc.get_samples(), prob=0.5)["r_hat"]
        logging.info(f"MCMC R-hat diagnostics:\n{rhats}")
    except Exception as e:
        logging.info(f"MCMC Failed diagnostics: {e}")

    predictive = Predictive(model=bnn_model, posterior_samples=mcmc.get_samples())
    preds = predictive(train_x, train_y.type(dtype))

    logging.info(f"Predictive log likelihood shape: {preds['log_likelihood'].shape}")
    logging.info(
        f"Predictive log likelihood of the training data: [TODO fix aggregation] {preds['log_likelihood'].mean()}"
    )

    rmse = torch.math.sqrt(
        ((preds["mu"].squeeze(1).mean(0).unsqueeze(-1) - train_y) ** 2).sum()
    )
    logging.info(f"RMSE {rmse}")

    ############################################################################

    test_grid = Xtest_tensor_
    logging.info(f"Posterior predictive on normalized test_grid = {test_grid.shape}")
    predictive = Predictive(model=bnn_model, posterior_samples=mcmc.get_samples())
    samples = predictive(test_grid.type(dtype))

    net_samples = samples["mu"].squeeze(1).T

    logging.info(
        f"Unnormalize X and y (net_samples={net_samples.shape}) for plotting posterior predictve"
    )
    net_samples = zscore_unnormalization(net_samples, y_mean, y_std)
    test_grid_unnormalized = zscore_unnormalization(test_grid, X_mean, X_std)

    # Evaluate posterior
    prediction_batch_posterior_predictive = net_samples.T

    # comparison
    common_batch_size = min(prediction_batch_posterior_predictive.shape[0], evaluation_batch_posteriors.shape[0])
    if prediction_batch_posterior_predictive.shape[0] != evaluation_batch_posteriors.shape[0]:
        logging.warning(f"GP posterior batch size={evaluation_batch_posteriors.shape[0]} vs "
                        f"BNN posterior batch size={prediction_batch_posterior_predictive.shape[0]} "
                        "The lower will be used.")
    if common_batch_size < 1024:
        logging.warning(f"Posterior evaluation is performed on {common_batch_size} samples, which is too little!")
    metrics_priors = compute_distribution_distances(prediction_batch_posterior_predictive[:common_batch_size, ...],
                                                    evaluation_batch_posteriors[:common_batch_size, ...])
    logging.info(f"Calculating distributional metrics for posterior predictive = {metrics_priors}")
    utils.save_configuration(results_prefix + "metrics_posterior_predictive.json", metrics_priors)

    plot_samples(
        test_grid_unnormalized,
        net_samples.detach().cpu().numpy(),
        ax=plt.gca(),
        color="xkcd:yellowish orange",
        n_keep=5,
    )
    plt.scatter(X, y)  # plot unnormalized data
    plt.ylim(-5, 5)
    plt.title("Posterior predictive")
    plt.savefig(results_prefix + "posterior_samples_pyro.png")
    plt.clf()

    ############################################################################

    logging.info(f"Saving results to target_dir = {target_dir}")


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
