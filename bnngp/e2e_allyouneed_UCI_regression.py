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
from pyro.nn import PyroModule, PyroSample
from pyro.infer import MCMC, NUTS
from pyro.infer import Predictive


import args
import targets_gp
import import_export
import utils

from train_priors_and_activations import main_parameterized as train_priors
from metrics import compute_distribution_distances

from baselines.you_need_a_good_prior.optbnn.utils.util import load_uci_data
from baselines.you_need_a_good_prior.optbnn.utils.normalization import normalize_data
from baselines.you_need_a_good_prior.optbnn.utils.exp_utils import get_input_range
from baselines.you_need_a_good_prior.optbnn.metrics import uncertainty as uncertainty_metrics


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


class SingleHiddenLayerWideUCIRegressionNNWithGaussianPriors(PyroModule):
    def __init__(
        self, parameters, likelihood_scale, net_width=200, in_dim=1, out_dim=1
    ):
        super().__init__()
        self.in_dim = in_dim
        self.likelihood_scale = likelihood_scale

        logging.info(
            f"[SingleHiddenLayerWideUCIRegressionNNWithGaussianPriors] Using priors parameters = {parameters}."
        )

        # Activation:
        self.activation = parameters.pop("activation")
        if self.activation is None:
            raise ValueError(
                "[SingleHiddenLayerWideUCIRegressionNNWithGaussianPriors] "
                "Activation must be passed as a parameter!"
            )
        # logging.info(f"[SingleHiddenLayerWide1DRegressionNNWithGaussianPriors] Using activation = {self.activation}.")
        if hasattr(
            self.activation, "named_parameters"
        ):  # turn off gradients in the module
            [p.requires_grad_(False) for _, p in self.activation.named_parameters()]

        self.layer1 = PyroModule[nn.Linear](in_dim, net_width)  # Input to hidden layer
        self.layer2 = PyroModule[nn.Linear](
            net_width, out_dim
        )  # Hidden to output layer

        # Set layer parameters as random variables
        self.layer1.weight = PyroSample(
            parameters["layer1.weight"].expand([net_width, in_dim]).to_event(2)
        )
        self.layer1.bias = PyroSample(
            parameters["layer1.bias"].expand([net_width]).to_event(1)
        )
        self.layer2.weight = PyroSample(
            parameters["layer2.weight"].expand([out_dim, net_width]).to_event(2)
        )
        self.layer2.bias = PyroSample(
            parameters["layer2.bias"].expand([out_dim]).to_event(1)
        )

    def forward(self, x, y=None):
        # logging.info("x.shape=", x.shape)
        x = x.reshape(-1, self.in_dim)
        # logging.info("x.shape=", x.shape)
        if y is not None:
            y = y.flatten()  # assume 1D outputs

        x = self.layer1(x)
        # logging.info("x.shape before activation=", x.shape)
        with torch.no_grad():
            x = self.activation(x)
        # logging.info("x.shape after activation=", x.shape)
        mu = self.layer2(x)

        # logging.info("mu.shape=", mu.shape)
        mu = mu.flatten()  # assume 1D outputs

        # Likelihood model
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample(
                "obs", dist.Normal(loc=mu, scale=self.likelihood_scale), obs=y
            )

        # Register as a deterministic output
        pyro.deterministic("mu", mu)

        if y is not None:
            log_likelihood = dist.Normal(loc=mu, scale=self.likelihood_scale).log_prob(
                y
            )
            pyro.deterministic("log_likelihood", log_likelihood)

        return mu


def prior_predictive_sampling(model, input_tensor, num_samples=119):
    predictive = pyro.infer.Predictive(model, num_samples=num_samples)
    samples = predictive(input_tensor)
    return samples


def main_parameterized(
    run_name="",
    random_seed=123,
    split_id=0,
    input_dim=13,
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
    dataset="boston",
    **fit_prior_kwargs,
):

    # ## Init
    prior_str = "trained_priors" if train_prior else "default_priors"
    target_dir = (
        f"pretrained/e2e_allyouneed_UCI_regression_{dataset}_split_{split_id}_{prior_str}_{run_name}_"
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
    data_dir = 'data/uci'

    X_train, y_train, X_test, y_test = load_uci_data(
        data_dir, split_id, dataset)
    X_train_, y_train_, X_test_, y_test_, y_mean, y_std = normalize_data(
        X_train, y_train, X_test, y_test)
    x_min, x_max = get_input_range(X_train_, X_test_)
    input_dim, output_dim = int(X_train.shape[-1]), 1
    print(input_dim)

    Xtest_tensor_ = torch.from_numpy(X_test_)
    logging.info("Normalize the dataset (normalized values have _ suffix)")

    ############################################################################
    # ## Targets (prior and posterior samples from a GP)
    gpmodel = targets_gp.AllYouNeedUCIRegression(input_dim=input_dim, sn2=sn2)

    # Draw functions from the priors
    util_set_seed(8)

    # gp_samples = gpmodel(Xtest_tensor_, n_plot).detach().cpu().numpy().squeeze()
    # gp_samples = zscore_unnormalization(gp_samples, y_mean, y_std)
    # gp_samples_tensor = torch.from_numpy(gp_samples).T
    #
    # # Posterior inference (for evaluation)
    # print(f"Xtest_tensor_={Xtest_tensor_.shape}")
    # gp_preds = gpmodel.gpmodel.predict_f_samples(Xtest_tensor_[0], n_plot)
    # gp_preds = gp_preds.detach().cpu().numpy().squeeze()
    # gp_preds = zscore_unnormalization(gp_preds, y_mean, y_std)
    #
    # gp_posterior_samples_tensor = torch.from_numpy(gp_preds).T

    # Evaluation: prior and posterior samples from GP

    # metrics_batch_size = 1024  # == number of functions
    # evaluation_batch_priors = gp_samples_tensor[:metrics_batch_size, ...]
    # evaluation_batch_posteriors = gp_posterior_samples_tensor[:metrics_batch_size, ...]
    #
    # assert evaluation_batch_priors.shape == torch.Size([metrics_batch_size, Xtest_tensor_.shape[0]]), f"Expected dims = (number of functions, grid elements) got {evaluation_batch_priors.shape}"
    # assert evaluation_batch_posteriors.shape == torch.Size([metrics_batch_size, Xtest_tensor_.shape[0]]), f"Expected dims = (number of functions, grid elements) got {evaluation_batch_posteriors.shape}"

    ############################################################################
    # ## Training priors

    if train_prior:
        fit = train_priors(
            random_seed=random_seed,
            bnn_width=posterior_net_width,
            config_set_no=int(2000 + split_id),  # AllYouNeed setting from 1D_regression_gaussian_priors.ipynb #TODO - to be changed!
            uniform_training_grid=False,  # AllYouNeed they have True
            uci_training_grid=True,
            dataset=dataset,
            sn2=sn2,
            n_iterations=prior_n_iterations,
            num_function_samples=prior_num_function_samples,
            batch_size=prior_batch_size,
            activation=prior_activation,
            run_name=results_prefix + "prior_fit",
            force_recomputing=True,
            final_evaluation_batch_size=256,
            final_evaluation_num_function_samples=256,
            input_n_dims=input_dim,
            **fit_prior_kwargs,
        )
        gc.collect()

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
    bnn_model = SingleHiddenLayerWideUCIRegressionNNWithGaussianPriors(
        in_dim=input_dim,
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


    ############################################################################

    logging.info("Posterior training is done on normalized data")
    train_x, train_y = torch.tensor(X_train_, dtype=dtype), torch.tensor(y_train_, dtype=dtype)

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


    rmse_our = torch.math.sqrt(
        ((zscore_unnormalization(preds["mu"].squeeze(1).mean(0).unsqueeze(-1), y_mean, y_std) - y_train) ** 2).sum()
    )
    logging.info(f"RMSE {rmse_our}")

    results_prior = dict()
    results_prior['train_rmse_our'] = rmse_our
    train_x = torch.tensor(X_train_, dtype=dtype)
    train_y = torch.tensor(y_train_, dtype=dtype)
    preds = predictive(train_x, train_y.type(dtype))
    rmse = uncertainty_metrics.rmse(zscore_unnormalization(preds["mu"].squeeze(1).mean(0).unsqueeze(-1).detach().cpu().numpy(), y_mean, y_std), y_train)
    print(f"RMSE_OUR={rmse_our}")
    print(f"RMSE={rmse}")
    nll = uncertainty_metrics.gaussian_nll(y_train, zscore_unnormalization(preds["mu"].squeeze(1).mean(0).unsqueeze(-1), y_mean, y_std),
                                           (preds["mu"].squeeze(1).var(0).unsqueeze(-1) + sn2)*(y_std**2))

    # print(preds['log_likelihood'].squeeze(1).mean(0).shape)
    nll_our = -preds['log_likelihood'].squeeze(1).mean(0).mean(0).detach().cpu().item()
    print(f"NLL={nll}")
    print(f"NLL_OUR={nll_our}")
    results_prior['train_rmse'] = rmse
    results_prior['train_nll'] = nll
    results_prior['train_nll_our'] = nll_our
    utils.save_configuration(results_prefix + "preds_prior.json", preds)
    utils.save_configuration(results_prefix + "results_prior.json", results_prior)

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
        # Get extra diagnostic fields such as divergences, step size, etc.
        extra_fields = mcmc.get_extra_fields()

        # Access specific diagnostics
        divergences = extra_fields["diverging"].numpy()
        step_size = extra_fields["adapt_state"]["step_size"].numpy()

        logging.info(f"MCMC Divergences:\n{divergences}")
        logging.info(f"MCMC Step Size:\n{step_size}")
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
    #TODO - check the following

    # net_samples = zscore_unnormalization(net_samples, y_mean, y_std)
    # test_grid_unnormalized = zscore_unnormalization(test_grid, X_mean, X_std)

    # Evaluate posterior
    # prediction_batch_posterior_predictive = net_samples.T
    #
    # # comparison
    # common_batch_size = min(prediction_batch_posterior_predictive.shape[0], evaluation_batch_posteriors.shape[0])
    # if prediction_batch_posterior_predictive.shape[0] != evaluation_batch_posteriors.shape[0]:
    #     logging.warning(f"GP posterior batch size={evaluation_batch_posteriors.shape[0]} vs "
    #                     f"BNN posterior batch size={prediction_batch_posterior_predictive.shape[0]} "
    #                     "The lower will be used.")
    # if common_batch_size < 1024:
    #     logging.warning(f"Posterior evaluation is performed on {common_batch_size} samples, which is too little!")
    # metrics_priors = compute_distribution_distances(prediction_batch_posterior_predictive[:common_batch_size, ...],
    #                                                 evaluation_batch_posteriors[:common_batch_size, ...])
    # logging.info(f"Calculating distributional metrics for posterior predictive = {metrics_priors}")
    # utils.save_configuration(results_prefix + "metrics_posterior_predictive.json", metrics_priors)

    #TODO

    #
    # logging.info(f"RMSE {rmse}")
    #
    # results_prior = dict()
    # results_prior['train_rmse_our'] = rmse
    # train_x = torch.tensor(X_train_, dtype=dtype)
    # train_y = torch.tensor(y_train_, dtype=dtype)
    # preds = predictive(train_x, train_y.type(dtype))
    # rmse = uncertainty_metrics.rmse(zscore_unnormalization(preds["mu"].squeeze(1).mean(0).unsqueeze(-1).detach().cpu().numpy(), y_mean, y_std), y_train)
    # print(f"RMSE={rmse}")
    # nll = uncertainty_metrics.gaussian_nll(y_train, zscore_unnormalization(preds["mu"].squeeze(1).mean(0).unsqueeze(-1), y_mean, y_std),
    #                                        (preds["mu"].squeeze(1).var(0).unsqueeze(-1) + torch.Tensor(sn2))*(y_std**2) )


    results_posterior = dict()

    test_x = torch.tensor(X_test_, dtype=dtype)
    test_y = torch.tensor(y_test_, dtype=dtype)
    preds = predictive(test_x, test_y.type(dtype))
    rmse_our = torch.math.sqrt(
        ((zscore_unnormalization(preds["mu"].squeeze(1).mean(0).unsqueeze(-1), y_mean, y_std) - y_test) ** 2).sum()
    )
    rmse = uncertainty_metrics.rmse(zscore_unnormalization(preds["mu"].squeeze(1).mean(0).unsqueeze(-1).detach().cpu().numpy(), y_mean, y_std), y_test)
    print(f"RMSE={rmse}")
    print(f"RMSE_OUR={rmse_our}")
    nll = uncertainty_metrics.gaussian_nll(y_test, zscore_unnormalization(preds["mu"].squeeze(1).mean(0).unsqueeze(-1), y_mean, y_std), (preds["mu"].squeeze(1).var(0).unsqueeze(-1) + sn2)*(y_std**2) )

    nll_our = -preds['log_likelihood'].squeeze(1).mean(0).mean(0).detach().cpu().item()
    print(f"NLL={nll}")
    print(f"NLL_OUR={nll_our}")

    results_posterior['test_rmse'] = rmse
    results_posterior['test_rmse_our'] = rmse_our
    results_posterior['test_nll'] = nll
    results_posterior['test_nll_our'] = nll_our

    preds_mu = preds['mu']
    preds_ll = preds['log_likelihood']
    preds_obs = preds['obs']
    # print(f"preds_mu={preds_mu}")
    # print(f"preds_ll={preds_ll}")
    # print(f"preds_obs={preds_obs}")
    ############################################################################
    utils.save_configuration(results_prefix + "preds_posterior_predictive.json", preds)
    logging.info(f"Saving results to target_dir = {target_dir}")
    utils.save_configuration(results_prefix + "results_posterior_predictive.json", results_posterior)


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

    # parsed_args['split_id'] = split_id
    return main_parameterized(**parsed_args)


if __name__ == "__main__":
    # for split_id in range(10):
    # main(split_id)
    main()
