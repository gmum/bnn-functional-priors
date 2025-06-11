import numpy as np
import torch
import matplotlib.pyplot as plt

from activations.gp_activations import sigma
import logging


def plot_activation(activation=sigma, x_min=-3.0, x_max=3.0):
    x = torch.tensor(np.arange(x_min, x_max, 0.1))
    plt.title("Activation function")
    plt.plot(x.cpu().detach(), activation(x).cpu().detach())
    return


def _create_fig(x_dim):
    # Determine the number of rows and columns for the subplots
    n = len(x_dim)
    n_cols = min(n, 4)
    n_rows = int(np.ceil(n / n_cols))

    # Prepare canvas
    if n_cols == 1:
        fig = plt.figure(figsize=(4.0, 1.5))
        axes = [plt.gca()]
    else:
        # Create a figure and a set of subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_rows * 4.0, n_cols * 1.5))
        # Flatten the axes array for easy iteration, handle case where n < n_cols * n_rows
        axes = axes.flatten()
    return fig, axes


def plot_functions_1D_slices(
    input_grid_x,
    batch_functions,
    n_plots=3,  # how many sampled functions
    title=None,
    x_dim=None,  # select which dimension should be plotted, if None plots all of them
    slices_width=0.5,  # in all dimensions apart from the currently plotted we need to slice data
):
    if x_dim is None:
        x_dim = list(range(input_grid_x.shape[-1]))
    elif isinstance(x_dim, int):
        x_dim = [x_dim]

    fig, axes = _create_fig(x_dim)
    x0 = input_grid_x.detach().cpu().numpy()
    for d in x_dim:

        # select points in the slice only
        mask = _extract_data_slice_mask(x0, d, slices_width)

        plt.sca(axes[d])
        for func_no in range(n_plots):
            # sample a network
            y = batch_functions[func_no]
            y = y.flatten().cpu().detach().numpy()
            y = y[mask]
            x = x0[mask, d]
            x, y = zip(*sorted(zip(x, y)))
            x, y = np.array(x), np.array(y)
            plt.plot(x, y)

        plt.xlabel(f"x dim={d}" if input_grid_x.shape[-1] > 1 else "x")
        plt.ylabel(f"f(x)")
    fig.suptitle(title or f"Distribution of functions f(x)")


def plot_network_functions_1D_slices(
    input_grid_x,
    *additional_inputs,
    network,
    n_plots=3,  # how many sampled functions
    title=None,
    x_dim=None,  # select which dimension should be plotted, if None plots all of them
    slices_width=0.5,  # in all dimensions apart from the currently plotted we need to slice data
    **kwargs,
):
    """
    Plot the output of a neural network over specific dimensions of the input.

    This function visualizes the output of the provided network over one or multiple dimensions
    of the input data. It allows slicing through multidimensional inputs to display
    function values for better interpretation of the model's behavior.
    """
    if x_dim is None:
        x_dim = list(range(input_grid_x.shape[-1]))
    elif isinstance(x_dim, int):
        x_dim = [x_dim]

    fig, axes = _create_fig(x_dim)
    x0 = input_grid_x.detach().cpu().numpy()
    for d in x_dim:

        # select points in the slice only
        mask = _extract_data_slice_mask(x0, d, slices_width)

        plt.sca(axes[d])
        for _ in range(n_plots):
            # sample a network
            y = network(input_grid_x, *additional_inputs)
            y = y.flatten().cpu().detach().numpy()
            y = y[mask]
            x = x0[mask, d]
            x, y = zip(*sorted(zip(x, y)))
            x, y = np.array(x), np.array(y)
            plt.plot(x, y, **kwargs)

        # plt.xlabel(f"x dim={d}" if input_grid_x.shape[-1] > 1 else "x")
        # plt.ylabel(f"f(x)")
    if title:
        fig.suptitle(title)


def _extract_data_slice_mask(x, d, slices_width):
    """
    Extracts a mask for selecting points within a specified width in all dimensions except the target dimension.

    This function generates a mask for filtering data points such that the selected points are
    within a defined distance from the slice center in all dimensions except for the one currently being plotted.

    Parameters:
    -----------
    x : np.ndarray
        Input data points as a numpy array.
    d : int
        Dimension that should be plotted (excluded from the slicing constraints).
    slices_width : float
        Initial width of the slice.

    Returns:
    --------
    mask : np.ndarray
        A boolean mask indicating the data points that fall within the defined slice.

    """
    n = x.shape[-1]
    slice_width = slices_width
    for _ in range(10):

        # iterate over all dimensions and select points within small distance from the slice center
        mask = np.ones(x.shape[0], dtype=bool)
        for d1 in range(n):
            if d == d1:
                continue

            mean = x[..., d1].mean().item()
            mask &= x[..., d1] > mean - 0.5 * slice_width
            mask &= x[..., d1] < mean + 0.5 * slice_width

        if np.sum(mask) > 10:
            break
        slice_width *= 2.0

    if slice_width > slices_width:
        logging.warning(
            f"[utils_plotting] Slicing window width for dim={d} increased from {slices_width} to {slice_width} to capture {np.sum(mask)} pts."
        )

    return mask


def plot_network_functions_1D(input_grid_x, network, **kwargs):
    return plot_network_functions_1D_slices(input_grid_x, network=network, **kwargs)


def plot_network_functions_1D_learnable_activation(
    input_grid_x, learnable_activation, network, **kwargs
):
    return plot_network_functions_1D_slices(
        input_grid_x, learnable_activation, network=network, **kwargs
    )


def plot_generator(generator, data_min_x, data_max_x, n_plots=10):
    test_grid_x = torch.tensor(
        np.linspace(start=data_min_x, stop=data_max_x, num=1000).reshape(-1, 1),
        dtype=torch.float32,
    )
    plot_network_functions_1D(
        test_grid_x,
        generator,
        n_plots=n_plots,
        title="Target distribution of functions",
    )


def plot_network(net, data_min_x, data_max_x, n_plots=10):
    test_grid_x = torch.tensor(
        np.linspace(start=data_min_x, stop=data_max_x, num=1000).reshape(-1, 1),
        dtype=torch.float32,
    )
    plot_network_functions_1D(
        test_grid_x,
        net,
        n_plots=n_plots,
        title="Learned distribution of functions f(x)",
    )


def eval_pwl_learnable_activation(learnable_activation):
    learnable_activation.eval()
    x = torch.tensor(np.arange(-5, 5, 0.1), dtype=torch.float32).unsqueeze(1)
    plt.plot(x.cpu().detach(), learnable_activation(x).cpu().detach())
    plt.title("Learnable activation function")
    learnable_activation.train()
    return


def plot_compare_pwl_activations(activation, learnable_activation):
    learnable_activation.eval()
    x = torch.tensor(np.arange(-5, 5, 0.1), dtype=torch.float32)
    plt.plot(
        x.cpu().detach(),
        learnable_activation(x.unsqueeze(1)).cpu().detach(),
        label="learnable activation",
    )
    plt.plot(x.cpu().detach(), activation(x).cpu().detach(), label="GT activation")
    plt.legend()
    plt.title("Activation functions - comparison")
    learnable_activation.train()
    return


def eval_learnable_activation(learnable_activation):
    learnable_activation.eval()
    x = torch.tensor(np.arange(-5, 5, 0.1), dtype=torch.float32).unsqueeze(1)
    plt.plot(x.cpu().detach(), learnable_activation(x).cpu().detach())
    plt.title("Learnable activation function")
    learnable_activation.train()
    return


def plot_compare_activations(activation, learnable_activation):
    learnable_activation.eval()
    x = torch.tensor(np.arange(-5, 5, 0.1), dtype=torch.float32)
    plt.plot(
        x.cpu().detach(),
        learnable_activation(x).cpu().detach(),
        label="learnable activation",
    )
    plt.plot(x.cpu().detach(), activation(x).cpu().detach(), label="GT activation")
    plt.legend()
    plt.title("Activation functions - comparison")
    learnable_activation.train()
    return
