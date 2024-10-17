#!/usr/bin/env python
# coding: utf-8

import sys
import json
import datetime
import os

from tqdm import tqdm

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import gpytorch

from gpytorch.models import AbstractVariationalGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.mlls.variational_elbo import VariationalELBO


import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from pyro.infer import Predictive

import args
import utils

import import_export
from examples import configs
import activations

from train_priors_and_activations import (
    main_parameterized as train_priors_and_activations,
)

import metrics

import logging

from bnn import SingleHiddenLayerWide1DClassificationNNWithGaussianPriors


class GPClassificationModel(AbstractVariationalGP):
    def __init__(
        self,
        train_x,
        mean_module=gpytorch.means.ConstantMean(),
        kernel=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
    ):

        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        # variational_distribution = gpytorch.variational.NaturalVariationalDistribution(train_x.size(0))

        variational_strategy = VariationalStrategy(
            self, train_x, variational_distribution
        )
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred


def sample_predictive_bernoulli_variational_gp(gp_model, input_x, n_samples=100):

    # Get posterior distribution of the latent function
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        latent_pred = gp_model(input_x)

        mean = latent_pred.mean
        variance = latent_pred.variance
        samples = torch.distributions.Normal(mean, variance.sqrt()).sample(
            torch.Size([n_samples])
        )

        # Apply sigmoid to transform to [0,1] interval representing probability p
        p_samples = torch.sigmoid(samples)
    return p_samples


def sigma(x, nu_ind=2, ell=0.5):
    """Implements the Matern activation function denoted as sigma(x) in Equation 9.
    sigma(x) corresponds to a Matern kernel, with specified smoothness
    parameter nu and length-scale parameter ell.

    Args:
    x: Input to the activation function
    device: A torch.device object
    nu_ind: Index for choosing Matern smoothness (look at nu_list below)
    ell: Matern length-scale, only 0.5 and 1 available with precalculated scaling coefficients
    """
    nu_list = [
        1 / 2,
        3 / 2,
        5 / 2,
        7 / 2,
        9 / 2,
    ]  # list of available smoothness parameters
    nu = torch.tensor(nu_list[nu_ind])  # smoothness parameter
    lamb = torch.sqrt(2 * nu) / ell  # lambda parameter
    v = nu + 1 / 2
    # Precalculated scaling coefficients for two different lengthscales (q divided by Gammafunction at nu + 0.5)
    ell05A = [
        4.0,
        19.595917942265423,
        65.31972647421809,
        176.69358285524189,
        413.0710073859664,
    ]
    ell1A = [
        2.0,
        4.898979485566356,
        8.16496580927726,
        11.043348928452618,
        12.90846898081145,
    ]
    if ell == 0.5:
        A = ell05A[nu_ind]
    if ell == 1:
        A = ell1A[nu_ind]
    y = A * torch.sign(x) * torch.abs(x) ** (v - 1) * torch.exp(-lamb * torch.abs(x))
    y[x < 0] = 0  # Values at x<0 must all be 0
    return y


# Specify neural network architecture: fully connected network with one hidden layer of 50 nodes.
class littleMLP(nn.Module):
    def __init__(self, num_classes=2, dropout=0.2, net_width=50):
        super(littleMLP, self).__init__()

        # FC layers
        self.fc1 = nn.Linear(2, net_width)
        self.fc2 = nn.Linear(net_width, num_classes)

        # Dropout layer
        self.drop_layer = nn.Dropout(p=dropout)

    def forward(self, x):
        x = sigma(self.fc1(x))
        x = self.drop_layer(x)
        x = self.fc2(x)
        return x


def sample_predictive_bernoulli_mlp(mlp, input_x, n_samples):
    # Function to allow dropout in model evaluation
    def apply_dropout(m):
        if type(m) == nn.Dropout:
            m.train()

    mlp.apply(apply_dropout)  # allow dropout in evaluation mode

    # MC dropout samples
    MCoutputs = np.zeros((input_x.shape[0], 2, n_samples))
    for i in tqdm(range(n_samples)):
        grid_outputs = F.softmax(mlp(input_x), dim=-1)
        grid_outputs = torch.squeeze(grid_outputs).detach().numpy()
        MCoutputs[:, :, i] = grid_outputs
    p_samples = MCoutputs[:, 1, :].T

    assert p_samples.shape == torch.Size([n_samples, input_x.shape[0]])
    return p_samples


def main_parameterized(
    # (Variational) GP:
    target_gp_config_set_no=-111,
    gp_optimize_hyperparameters=False,  # if false only variational params are optimized
    # test grid:
    gridwidth=100,  # 300 #number of test samples in each dimension  #300?
    gridlength=3.75,  # how far to extend the test samples around 0 in both dimensions
    # HMC posterior
    MCMC_n_samples=1000,
    MCMC_n_chains=4,
    MCMC_warmup=500,
    bnn_net_width=1000,
    # Plotting
    plot_predictive_n_samples=1000,
    # Priors
    priors="train",
    # priors = "default_with_relu"
    # priors = "normal_with_relu"
    # prior learning
    train_input_n_dims=2,
    train_n_iterations=3001,
    train_num_function_samples=512,
    train_batch_size=512,
    train_activation="nnsilu_1_5",
    **fit_prior_kwargs,
):
    assert (
        MCMC_n_samples % MCMC_n_chains == 0
    ), "MCMC_n_samples must be divisible by MCMC_n_chains"

    # ## Init
    target_dir = (
        f"pretrained/e2e_twomoons_2d_classification_{priors}_{train_input_n_dims}d_"
        + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    )
    results_prefix = target_dir + "/"
    logging.info(f"Saving results to target_dir = {target_dir}")

    if target_dir is not None and target_dir != "":
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

    # Set GP kernel:
    generator, meta, data_min_x, data_max_x = configs.get_configs(
        config_set_no=target_gp_config_set_no
    )
    kernel = generator.kernel

    # # Load and preprocess data.

    # !rm *.csv
    # !mkdir data/
    # !wget https://raw.githubusercontent.com/AaltoML/stationary-activations/main/data/datapoints.csv
    # !mv datapoints.csv data/
    # !wget https://raw.githubusercontent.com/AaltoML/stationary-activations/main/data/classes.csv
    # !mv classes.csv data/

    # Download data from .csv files
    data = pd.read_csv("data/datapoints.csv", header=None)  # 2D points
    data.columns = [f"col{c}" for c in data.columns]

    classes = pd.read_csv("data/classes.csv", header=None)  # class labels
    classes.columns = [f"col{c}" for c in classes.columns]

    N_samples = len(classes)  # Number of training samples

    # Data to numpy arrays
    data_array = np.zeros((N_samples, 2))
    data_array[:, 0] = np.asarray(data.col0[:])
    data_array[:, 1] = np.asarray(data.col1[:])
    class_array = np.asarray(classes.col0)

    # Data to torch tensors for training
    training_data = torch.from_numpy(data_array).float()
    training_targets = torch.from_numpy(class_array).long()

    x_vals = np.linspace(-gridlength, gridlength, gridwidth)
    y_vals = np.linspace(-gridlength, gridlength, gridwidth)
    grid_samples = np.zeros((gridwidth * gridwidth, 2))
    for i in range(gridwidth):
        for j in range(gridwidth):
            grid_samples[i * gridwidth + j, 0] = x_vals[i]
            grid_samples[i * gridwidth + j, 1] = y_vals[j]

    grid_set = torch.from_numpy(grid_samples).float()  # Grid samples to torch tensor

    xv, yv = np.meshgrid(x_vals, y_vals)  # Sample grid as a meshgrid

    train_x = training_data
    train_y = training_targets

    # # Aux

    def set_style(ax1):
        ax1.grid(False)
        ax1.set_yticklabels([])
        ax1.set_xticklabels([])
        ax1.set_xticks([], [])
        ax1.set_yticks([], [])
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")

    def plot_grid(ax, vals, vmax=1.0, contour=False, cmap="bwr_r"):
        im = ax.imshow(
            vals,
            cmap=cmap,  # cmap='gray_r',
            extent=[-gridlength, gridlength, -gridlength, gridlength],
            origin="lower",
            alpha=0.5,
            vmin=0,
            vmax=vmax,
        )
        ax.scatter(
            data_array[class_array == 0, 0],
            data_array[class_array == 0, 1],
            c="red",
            s=10,
            alpha=0.5,
        )
        ax.scatter(
            data_array[class_array == 1, 0],
            data_array[class_array == 1, 1],
            c="dodgerblue",
            s=10,
            alpha=1.0,
        )
        if contour:
            ax.contour(xv, yv, vals, [0.5], colors="k")
        return im

    def plot_predictive_bernoulli(
        p_values,
        means_title="Predictive Expectations E[p]",
        stds_title="Predictive Std[p]",
        means_label="E[p]",
        stds_label="Std[p]",
        title=None,
    ):
        exp_p = p_values.mean(
            axis=0
        )  # =E[y] https://en.wikipedia.org/wiki/Law_of_total_expectation
        std_p = p_values.std(axis=0)

        std_exp_y_cond_p = p_values.std(axis=0)
        var_exp_y_cond_p = std_exp_y_cond_p**2  # Var[E[y|p]]

        exp_var_y_cond_p = (p_values * (1.0 - p_values)).mean(axis=0)  # E[Var[y|p]]
        exp_std_y_cond_p = np.sqrt(exp_var_y_cond_p + 1e-12)

        var_y = (
            exp_var_y_cond_p + var_exp_y_cond_p
        )  # https://en.wikipedia.org/wiki/Law_of_total_variance
        std_y = np.sqrt(var_y + 1e-12)

        exp_p = exp_p.reshape(gridwidth, gridwidth).T
        std_p = std_p.reshape(gridwidth, gridwidth).T
        std_exp_y_cond_p = std_exp_y_cond_p.reshape(gridwidth, gridwidth).T
        exp_std_y_cond_p = exp_std_y_cond_p.reshape(gridwidth, gridwidth).T
        std_y = std_y.reshape(gridwidth, gridwidth).T

        fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(20, 8))
        if title:
            plt.suptitle(title)

        file_prefix = results_prefix + (title or "fig")
        file_prefix = file_prefix.lower().replace(" ", "_").replace(":", "-")

        ax = axes[0, 0]
        im = plot_grid(ax, exp_p, vmax=1.0, contour=True, cmap="bwr_r")
        fig.colorbar(im, label="E[y] (=E[p])", ax=ax)
        set_style(ax)
        ax.set_title("Predictive E[y] (=E[p])")

        ax = axes[0, 1]
        im = plot_grid(
            ax,
            np.sqrt(exp_p * (1 - exp_p) + 1e-12),
            vmax=0.5,
            contour=False,
            cmap="hot",
        )
        fig.colorbar(im, label="Std(E[y])", ax=ax)
        set_style(ax)
        ax.set_title("Predictive Std(E[y])")

        ax = axes[0, 2]
        im = plot_grid(ax, std_p, vmax=0.5, contour=False, cmap="hot")
        fig.colorbar(im, ax=ax, label="Std[p]")
        set_style(ax)
        ax.set_title("Predictive Std[p]")

        ax = axes[1, 2]
        im = plot_grid(ax, std_exp_y_cond_p, vmax=0.5, contour=False, cmap="hot")
        fig.colorbar(im, ax=ax, label="Std[E[y|p]]")
        set_style(ax)
        ax.set_title("Predictive Std[E[y|p]]")

        ax = axes[1, 1]
        im = plot_grid(ax, exp_std_y_cond_p, vmax=0.5, contour=False, cmap="hot")
        fig.colorbar(im, ax=ax, label="E[Std[y|p]]")
        set_style(ax)
        ax.set_title("Predictive E[Std[y|p]]")

        ax = axes[1, 0]
        im = plot_grid(ax, std_y, vmax=0.5, contour=False, cmap="hot")
        fig.colorbar(im, ax=ax, label="Std[y]")
        set_style(ax)
        ax.set_title("Std[y] (=sqrt( E[Var[y|p]] + Var[E[y|p]]) )")

        fig.savefig(file_prefix + ".pdf")

        fig.tight_layout()

    def report(line, verbose=True):
        if verbose:
            logging.info(line)

        line = line.replace("\n", "\\ ")
        with open(results_prefix + "e2e_results.txt", "a") as results:
            results.write(line + "\n")

    def report_metric(name, value, **kwargs):
        report(str(name) + " = " + str(value), **kwargs)

    logging.info("Reporting configurations...")

    variables = locals().copy()
    for name, value in variables.items():
        if hasattr(value, "shape"):
            value = value.shape
        value = str(value)
        if name.startswith("_") or value.startswith("<"):
            continue
        if len(str(value)) > 200:
            value = value[:200]
        report_metric(name, value, verbose=False)

    ##########################################################################
    # # Variational GP learning: Ground-truth solution

    # Initialize model and likelihood
    gp_model = GPClassificationModel(train_x, kernel=kernel)
    likelihood = gpytorch.likelihoods.BernoulliLikelihood()

    report(f"GP kernel = {kernel}")
    report(
        f"GP parameters = {list((n, p.shape) for n, p in gp_model.named_parameters())}"
    )

    # Go into eval mode
    gp_model.eval()
    likelihood.eval()

    n_discrepancies = sum(
        likelihood(gp_model(grid_set)).mean * (1 - likelihood(gp_model(grid_set)).mean)
        != likelihood(gp_model(grid_set)).variance
    )
    logging.info(
        f"NOTE: Predictive variance in likelihoods is just = p*(1-p). Number of discrepancies: {n_discrepancies}"
    )

    MC = 100
    with torch.no_grad():
        # Get classification predictions
        observed_pred = torch.stack(
            [likelihood(gp_model(grid_set)).mean for _ in range(MC)]
        )

    plot_predictive_bernoulli(observed_pred.numpy(), title="GP Model initialized")
    # plt.show()

    gp_model.eval()
    likelihood.eval()

    NLL = (
        -likelihood(gp_model(train_x))
        .log_prob(torch.tensor(train_y, dtype=train_x.dtype))
        .sum()
    )
    report(f"GP prior total NLL = {NLL}")

    # Optimize kernel hyperparams via marginal likelihood

    if gp_optimize_hyperparameters:
        trained_params = {
            n: p for n, p in gp_model.named_parameters()
        }  # optimize all parameters
    else:
        trained_params = {
            n: p for n, p in gp_model.named_parameters() if "variational" in n
        }  # optimize variational parameters only

    trained_params_str = "\n - ".join(trained_params.keys())
    logging.info(f"Optimizing GP hyperparameters:\n - {trained_params_str}")

    # Find optimal model hyperparameters
    gp_model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(trained_params.values(), lr=0.01)

    # "Loss" for GPs - the marginal log likelihood
    # num_data refers to the amount of training data
    mll = VariationalELBO(likelihood, gp_model, train_y.numel())

    training_iter = 200
    for i in range(training_iter):
        # Zero backpropped gradients from previous iteration
        optimizer.zero_grad()
        # Get predictive output
        output = gp_model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()

        train_bce = (
            likelihood(gp_model(train_x))
            .log_prob(torch.tensor(train_y, dtype=torch.float))
            .exp()
            .mean()
        )
        logging.info(
            "Iter %d/%d - Loss: %.3f - data mean prob: :%.3f"
            % (i + 1, training_iter, loss.item(), train_bce.item())
        )
        optimizer.step()

    # Go into eval mode
    gp_model.eval()
    likelihood.eval()

    n_discrepancies = sum(
        likelihood(gp_model(grid_set)).mean * (1 - likelihood(gp_model(grid_set)).mean)
        != likelihood(gp_model(grid_set)).variance
    )
    logging.info(
        f"NOTE: Predictive variance in likelihoods is just = p*(1-p). Number of discrepancies: {n_discrepancies}"
    )

    MC = 10
    with torch.no_grad():
        # Get classification predictions
        observed_pred = torch.stack(
            [likelihood(gp_model(grid_set)).mean for _ in range(MC)]
        )

    plot_predictive_bernoulli(
        observed_pred.numpy(),
        title="GP Model fitted [NOTE: THERE IS NO REAL MC SAMPLING HERE!]",
    )
    # plt.show()

    NLL = (
        -likelihood(gp_model(train_x))
        .log_prob(torch.tensor(train_y, dtype=train_x.dtype))
        .sum()
    )
    report(f"GP posterior total NLL (for mean p) = {NLL}")

    gp_p_samples_for_eval = p_samples = sample_predictive_bernoulli_variational_gp(
        gp_model, grid_set, plot_predictive_n_samples
    )
    plot_predictive_bernoulli(
        p_samples, title=f"GP model fitted ({plot_predictive_n_samples} MC samples)"
    )

    y = train_y
    p = sample_predictive_bernoulli_variational_gp(
        gp_model, train_x, plot_predictive_n_samples
    )
    assert p.shape == torch.Size([plot_predictive_n_samples, y.shape[0]])
    NLL = -(np.log(p + 1e-20) * (y) + np.log(1 - p + 1e-20) * (1 - y)).mean(0).sum(-1)
    report(f"GP posterior total NLL ({plot_predictive_n_samples} MC samples) = {NLL}")

    ##########################################################################
    # # (MC) Dropout with Sigma Activation

    # Specify parameters for model specification and training.

    LR = 0.02  # learning rate
    n_epochs = 2000
    batch_size = 400  # full batch

    MC = plot_predictive_n_samples  # number of MC dropout samples
    dropout = 0.2

    net_width = 50  # width from the original code = 50

    # Specify matern activation function with precalculated scaling coefficients. By default the Matern-5/2 with a length-scale 0.5 is chosen to replicate what was done in the paper. This activation function is Equation 9 in the paper.

    # Initialize the network.
    mlp = littleMLP(dropout=dropout, net_width=net_width)

    # Initialize training setup.
    criterion = nn.CrossEntropyLoss()  # Loss function to be used
    optimizer = optim.Adam(mlp.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[n_epochs / 8, n_epochs / 4, n_epochs / 2], gamma=0.1
    )

    accuracies = np.zeros(
        n_epochs
    )  # Initialize array for saving mid training classification accuracies

    # Training loop.
    for epoch in tqdm(range(n_epochs)):  # loop over the dataset multiple times
        # random index for data shuffling for epochs
        index = np.arange(0, N_samples)
        random.shuffle(index)

        # shuffle data for each epoch
        curr_input_set = training_data[index, :]
        curr_target_set = training_targets[index]
        curr_ind = 0
        while (
            curr_ind < N_samples
        ):  # Loop through all batches (with the default batch size just one iteration)
            new_ind = min(N_samples, curr_ind + batch_size)
            outputs = mlp(
                curr_input_set[curr_ind:new_ind, :]
            )  # Calculate network outputs
            outputs = torch.squeeze(outputs)
            loss = criterion(
                outputs, curr_target_set[curr_ind:new_ind]
            )  # Calculate loss
            curr_ind = new_ind
            optimizer.zero_grad()  # zero the parameter gradients
            loss.backward()  # backpropagate gradients
            optimizer.step()  # update network weights
        scheduler.step()  # Update learning rate

        mlp.eval()
        outputs = mlp(training_data)  # Network outputs for accuracy tracking
        outputs = torch.squeeze(outputs).detach().numpy()
        preds = np.argmax(outputs, axis=1)  # Predicted classes
        accuracies[epoch] = (
            np.sum(class_array.astype(int) == preds) / N_samples
        )  # Fraction of correct predictions
        mlp.train()

    # Plot classification accuracy on training samples for each epoch.
    plt.figure()
    plt.plot(accuracies)
    plt.title("Classification accuracy on training samples")
    plt.xlabel("epoch number")
    plt.ylabel("accuracy")
    plt.savefig(results_prefix + "mlp_accuracy.png")

    # Generate a grid of test samples.

    # Model testing on a grid of test samples to get the predicted classes.
    mlp.eval()

    # Predict grid samples to classes without MC dropout
    grid_outputs = F.softmax(mlp(grid_set), dim=-1)
    grid_outputs = torch.squeeze(grid_outputs).detach().numpy()
    grid_preds = np.argmax(grid_outputs, axis=1)  # get predicted class

    p_samples = grid_preds[
        None, ...
    ]  # add "sampling" dimension despite there's no sampling
    plot_predictive_bernoulli(p_samples, title="Dropout frozen")

    # #### Model testing on a grid of test samples using MC dropout to get uncertainty estimates.
    mlp_p_samples_for_eval = p_samples = sample_predictive_bernoulli_mlp(
        mlp, grid_set, plot_predictive_n_samples
    )
    logging.info(f"p_samples.shape = {p_samples.shape}")

    plot_predictive_bernoulli(
        p_samples,
        title=f"Dropout MC: Mean and Std from {plot_predictive_n_samples} MC samples",
    )
    # plt.show()

    # ### Dropout NLL evaluation on training set
    y = train_y.cpu().detach().numpy()
    p = sample_predictive_bernoulli_mlp(mlp, train_x, plot_predictive_n_samples)
    assert p.shape == torch.Size([plot_predictive_n_samples, y.shape[0]])
    NLL = -(np.log(p + 1e-20) * (y) + np.log(1 - p + 1e-20) * (1 - y)).mean(0).sum()
    logging.info(
        f"Dropout (MC) total NLL (for sampled Bernoulli probabilites (p)) = {NLL}"
    )

    mlp.eval()

    # Predict grid samples to classes without MC dropout
    grid_outputs = F.softmax(mlp(train_x), dim=-1)
    grid_outputs = torch.squeeze(grid_outputs).detach().numpy()

    p = grid_outputs[:, 1]
    assert p.shape == y.shape
    NLL = -(np.log(p) * (y) + np.log(1 - p) * (1 - y)).sum()

    report(f"Dropout (frozen) total NLL (for frozen Dropout) = {NLL}")

    ##########################################################################
    # # BNN

    # ### BNN parameters
    # Set Pyro random seed
    pyro.set_rng_seed(41)

    # Pretrained priors:
    config_json = "pretrained/train_priors_and_activations_Ann_1_5_Pgaussian_0s_S1_results.json"

    # Building model
    logging.info(f"Parsing priors={priors}")

    if priors == "train":
        fit = train_priors_and_activations(
            config_set_no=target_gp_config_set_no,
            bnn_width=bnn_net_width,
            input_n_dims=train_input_n_dims,
            n_iterations=train_n_iterations,
            num_function_samples=train_num_function_samples,
            batch_size=train_batch_size,
            activation=train_activation,
            run_name=results_prefix + "prior_fit",
            freeze_training_grid=False,
            final_evaluation_batch_size=512,
            final_evaluation_num_function_samples=512,
            uniform_training_grid=False,
            force_recomputing=True,
            **fit_prior_kwargs,
        )

        bnn_model = SingleHiddenLayerWide1DClassificationNNWithGaussianPriors(
            net_width=bnn_net_width,
            priors=import_export.decode_gaussian_priors(
                fit["parameters"]["prior"], target_net_width=bnn_net_width
            ),
            activation=import_export.decode_activation(fit["parameters"]["activation"]),
        )

    elif priors == "pretrained":
        config = import_export.load_parameters_from_json(
            config_json, net_width=bnn_net_width
        )
        bnn_model = SingleHiddenLayerWide1DClassificationNNWithGaussianPriors(
            net_width=bnn_net_width,
            config=config,
        )

    elif priors.startswith("pretrained_with_"):
        activation_name = priors.split("_")[-1]
        activation = activations.get_activation(activation_name)
        logging.info(f"Retrieved activation = {activation}")

        config = import_export.load_parameters_from_json(
            config_json, net_width=bnn_net_width
        )
        bnn_model = SingleHiddenLayerWide1DClassificationNNWithGaussianPriors(
            net_width=bnn_net_width,
            config=config,
            activation=activation,
        )

    elif priors.startswith("normal_with_"):
        prior_parameters = {  # defaults which may be overwritten below
            "layer1.weight": dist.Normal(0.0, 1.0),
            "layer1.bias": dist.Normal(0.0, 1.0),
            "layer2.weight": dist.Normal(0.0, 1.0 / np.sqrt(bnn_net_width)),
            "layer2.bias": dist.Normal(0.0, 1.0),
        }

        activation_name = priors.split("_")[-1]
        activation = activations.get_activation(activation_name)
        logging.info(f"Retrieved activation = {activation}")

        bnn_model = SingleHiddenLayerWide1DClassificationNNWithGaussianPriors(
            net_width=bnn_net_width,
            priors=prior_parameters,
            activation=activation,
        )

    elif priors.startswith("default_with_"):
        prior_parameters = {  # defaults which may be overwritten below
            "layer1.weight": dist.Normal(0.0, 1.0),
            "layer1.bias": dist.Normal(0.0, 1.0),
            "layer2.weight": dist.Normal(
                0.0, 1.0
            ),  # NOTE: enable overconfidence for ReLU !!!
            "layer2.bias": dist.Normal(0.0, 1.0),
        }

        activation_name = priors.split("_")[-1]
        activation = activations.get_activation(activation_name)
        logging.info(f"Retrieved activation = {activation}")

        bnn_model = SingleHiddenLayerWide1DClassificationNNWithGaussianPriors(
            net_width=bnn_net_width,
            priors=prior_parameters,
            activation=activation,
        )

    else:
        raise ValueError(f"Failed to interpret priors = {priors}!")

    logging.info(f"BNN: model = {bnn_model}")
    plt.plot(
        torch.arange(-10, 10, 0.05), bnn_model.activation(torch.arange(-10, 10, 0.05))
    )
    plt.grid(True)
    plt.title("BNN activation")

    # #### Sample from prior
    p_samples = torch.stack(
        [bnn_model(grid_set) for _ in range(plot_predictive_n_samples)]
    )

    plot_predictive_bernoulli(
        p_samples, title=f"BNN: {plot_predictive_n_samples} MC prior samples"
    )
    # plt.show()

    # ### Posterior training and evaluation

    # Define Hamiltonian Monte Carlo (HMC) kernel

    # NUTS = "No-U-Turn Sampler" (https://arxiv.org/abs/1111.4246), gives HMC an adaptive step size
    nuts_kernel = NUTS(
        bnn_model, jit_compile=True
    )  # jit_compile=True is faster but requires PyTorch 1.6+

    # Define MCMC sampler, get MC posterior samples
    mcmc = MCMC(
        nuts_kernel,
        num_samples=MCMC_n_samples // MCMC_n_chains,
        num_chains=MCMC_n_chains,
        warmup_steps=max(MCMC_warmup, MCMC_n_samples // MCMC_n_chains),
    )

    # Run MCMC
    mcmc.run(train_x, torch.tensor(train_y, dtype=torch.float))

    # Print summary statistics
    mcmc.summary(prob=0.8)  # This gives you the 80% credible interval by default

    predictive = Predictive(model=bnn_model, posterior_samples=mcmc.get_samples())
    preds = predictive(train_x, train_y.type(torch.float32))

    report_metric(
        "BNN posterior predictive log likelihood of the training data (mean)",
        preds["log_likelihood"].mean(),
    )
    report_metric(
        "BNN posterior predictive log likelihood of the training data (total)",
        preds["log_likelihood"].mean(0).sum(),
    )

    # p_values = torch.stack([bnn_model(grid_set) for _ in range(MC)])
    preds = predictive(grid_set)
    logging.info(
        f"BNN preds: grid_set.shape={grid_set.shape} p.shape={preds['p'].shape} obs={preds['obs'].shape}"
    )
    bnn_p_samples_for_eval = p_samples = preds["p"][:, 0, :]  # Get Bernoulli p values

    MC = preds["p"].shape[0]
    plot_predictive_bernoulli(p_samples, title=f"BNN: {MC} MCMC posterior samples")
    # plt.show()

    preds = predictive(train_x)
    y = train_y.cpu().detach().numpy()
    p = preds["p"][:, 0, :]  # Get Bernoulli p values
    assert p.shape == torch.Size([MCMC_n_samples, y.shape[0]])

    NLL = -(np.log(p + 1e-20) * (y) + np.log(1 - p + 1e-20) * (1 - y)).mean(0).sum()
    logging.info(f"BNN posterior total NLL = {NLL}")

    ##########################################################################
    # ## Evaluation

    mlp_p_samples_for_eval = torch.tensor(
        mlp_p_samples_for_eval, dtype=gp_p_samples_for_eval.dtype
    )

    min_n_func_samples = min(
        bnn_p_samples_for_eval.shape[0],
        gp_p_samples_for_eval.shape[0],
        mlp_p_samples_for_eval.shape[0],
    )
    report(
        f"Final evaluation was performed on batch_size={min_n_func_samples} functions"
    )

    report_metric("mlp_p_samples_for_eval", mlp_p_samples_for_eval.shape)
    report_metric("bnn_p_samples_for_eval", bnn_p_samples_for_eval.shape)
    report_metric("gp_p_samples_for_eval", gp_p_samples_for_eval.shape)

    mlp_p_samples_for_eval = mlp_p_samples_for_eval[:min_n_func_samples, ...]
    bnn_p_samples_for_eval = bnn_p_samples_for_eval[:min_n_func_samples, ...]
    gp_p_samples_for_eval = gp_p_samples_for_eval[:min_n_func_samples, ...]

    bnn_metrics = metrics.compute_distribution_distances(
        bnn_p_samples_for_eval, gp_p_samples_for_eval
    )
    mlp_metrics = metrics.compute_distribution_distances(
        mlp_p_samples_for_eval, gp_p_samples_for_eval
    )
    cross_metrics1 = metrics.compute_distribution_distances(
        mlp_p_samples_for_eval, bnn_p_samples_for_eval
    )
    cross_metrics2 = metrics.compute_distribution_distances(
        bnn_p_samples_for_eval, mlp_p_samples_for_eval
    )

    for metric_name in bnn_metrics.keys():
        logging.info("=============")
        report_metric(f"BNN<-GP {metric_name}", bnn_metrics[metric_name])
        report_metric(f"MLP<-GP {metric_name}", mlp_metrics[metric_name])
        report_metric(
            f"BNN<-MLP (BNN as ground truth) {metric_name}", cross_metrics1[metric_name]
        )
        report_metric(
            f"BNN->MLP (MLP as ground truth) {metric_name}", cross_metrics2[metric_name]
        )

    with open(results_prefix + "bnn_metrics.json", "w") as fp:
        json.dump(bnn_metrics, fp)
    with open(results_prefix + "mlp_metrics.json", "w") as fp:
        json.dump(mlp_metrics, fp)
    with open(results_prefix + "cross_metrics1.json", "w") as fp:
        json.dump(cross_metrics1, fp)
    with open(results_prefix + "cross_metrics2.json", "w") as fp:
        json.dump(cross_metrics2, fp)

    logging.info(f"Saving results to target_dir = {target_dir}")


def main():
    # set up logging
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
        force=True,
    )
    # logging.getLogger().addHandler(logging.StreamHandler())

    parsed_args = args.parse_args()
    utils.setup_torch(
        parsed_args.pop("device", None),
        parsed_args.pop("dtype", None),
        parsed_args.pop("n_threads", None),
    )
    return main_parameterized(**parsed_args)


if __name__ == "__main__":
    main()
