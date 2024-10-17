""" Generating input measure sets. """

import torch
import numpy as np
import math
import logging

from baselines.you_need_a_good_prior.optbnn.utils.util import load_uci_data
from baselines.you_need_a_good_prior.optbnn.utils.normalization import normalize_data
from baselines.you_need_a_good_prior.optbnn.utils.exp_utils import get_input_range


def create_uniform_1D_grid(
    data_min_x, data_max_x, n_nodes, dtype=torch.get_default_dtype()
):
    return torch.tensor(
        np.linspace(start=data_min_x, stop=data_max_x, num=n_nodes).reshape(-1, 1),
        dtype=dtype,
    )


def create_sampled_uniform_1D_grid(
    data_min_x,
    data_max_x,
    n_nodes,
    dtype=torch.get_default_dtype(),
):
    train_grid_x = torch.tensor(
        np.array(
            sorted(
                np.random.uniform(
                    low=data_min_x,
                    high=data_max_x,
                    size=n_nodes,
                )
            )
        ),
        dtype=dtype,
    ).reshape(-1, 1)

    return train_grid_x


def _get_hypercube_dims(data_min_x, data_max_x, n_dims):
    # Check if data_min_x and data_max_x are scalars or lists/arrays
    if isinstance(data_min_x, (int, float)) and isinstance(data_max_x, (int, float)):
        if n_dims is None:
            raise ValueError(
                "n_dims must be specified if data_min_x and data_max_x are scalars."
            )
        low = [data_min_x] * n_dims
        high = [data_max_x] * n_dims
    elif isinstance(data_min_x, (list, np.ndarray)) and isinstance(
        data_max_x, (list, np.ndarray)
    ):
        if len(data_min_x) != len(data_max_x):
            raise ValueError("data_min_x and data_max_x must have the same length.")
        if n_dims is not None and n_dims != len(data_min_x):
            raise ValueError(
                "n_dims is specified and does not match the length of data_min_x and data_max_x."
            )
        low = data_min_x
        high = data_max_x
        n_dims = len(data_min_x)
    else:
        raise TypeError(
            "data_min_x and data_max_x must be both scalars or both lists/arrays."
        )
    return low, high, n_dims


def create_uniform_grid(
    data_min_x, data_max_x, n_nodes, n_dims=None, dtype=torch.get_default_dtype()
):
    if n_dims == 1:
        return create_uniform_1D_grid(data_min_x, data_max_x, n_nodes, dtype)

    low, high, n_dims = _get_hypercube_dims(data_min_x, data_max_x, n_dims)
    # Generate uniform points within the specified bounds
    if n_dims == 1:
        points = np.linspace(low, high, n_nodes).reshape(-1, 1)

    else:

        # Match (approximately) the number of desired nodes
        n_nodes_per_dim = int(np.round(math.pow(n_nodes, 1.0 / n_dims)))
        if n_nodes_per_dim <= 1:
            logging.warning(
                f"ERROR: Grid(n_nodes={n_nodes}, n_dims={n_dims}) collapsed to a single point! "
                f"Increase n_nodes (currently ={n_nodes})!"
            )

        # Generate grid points for each dimension
        axes = [
            np.linspace(low[dim], high[dim], n_nodes_per_dim) for dim in range(n_dims)
        ]
        grid = np.meshgrid(*axes, indexing="ij")
        # Flatten and transpose the grid to list points as rows
        points = np.stack([axis.ravel() for axis in grid], axis=-1)
        logging.info(
            f"[create_uniform_grid] created grid with {len(points)} vs. requested={n_nodes}"
        )

    # Convert to tensor with the specified dtype
    uniform_grid = torch.tensor(points, dtype=dtype)

    return uniform_grid


def create_sampled_uniform_grid(
    data_min_x,
    data_max_x,
    n_nodes,
    n_dims=None,
    dtype=torch.get_default_dtype(),
):
    if n_dims == 1:
        return create_sampled_uniform_1D_grid(data_min_x, data_max_x, n_nodes, dtype)

    low, high, n_dims = _get_hypercube_dims(data_min_x, data_max_x, n_dims)

    # Generate random points within the specified bounds
    sampled_points = np.random.uniform(low=low, high=high, size=(n_nodes, n_dims))
    if n_dims == 1:  # there should be no ordering if D>1
        sampled_points.sort(axis=0)  # Sort along each dimension

    # Convert to tensor with the specified dtype
    train_grid = torch.tensor(sampled_points, dtype=dtype)

    return train_grid




class MeasureSetGenerator(object):  # to compare against AllYouNeedIsGoodFunctionalPriors
    def __init__(self, X, x_min, x_max, real_ratio=0.8, fix_data=False):
        if not isinstance(X, torch.Tensor):
            self.X = torch.from_numpy(X).float()
        else:
            self.X = X.float()
        self.x_min = x_min
        self.x_max = x_max
        self.real_ratio = real_ratio

        # Initialize generator to create random pointss
        self.rand_generator = torch.distributions.uniform.Uniform(
            torch.from_numpy(self.x_min).float(), torch.from_numpy(self.x_max).float()
        )

        self.use_cache = fix_data
        self.X_cached = None

    def get(self, n_data):
        if self.use_cache and self.X_cached is not None:
            return self.X_cached

        n_real = int(n_data * self.real_ratio)
        # assert n_real < self.X.shape[0]
        n_real = min(n_real, int(self.X.shape[0]))
        n_rand = n_data - n_real

        # Choose randomly training inputs
        indices = torch.randperm(self.X.shape[0])[:n_real]

        # Generate random points
        X_real = self.X[indices, ...]
        X_rand = self.rand_generator.rsample([n_rand])

        # Concatenate both sets
        X = torch.cat((X_real, X_rand), axis=0)
        indices = torch.randperm(X.shape[0])
        X = X[indices, ...]

        if self.use_cache:
            self.X_cached = X
        return X


class UCIGenerators(object):  # to compare against AllYouNeedIsGoodFunctionalPriors
    def __init__(self, dataset):
        data_dir = "data/uci"

        self.split_generators = []

        for split_no in range(10):
            X_train, y_train, X_test, y_test = load_uci_data(
                data_dir, split_no, dataset
            )
            X_train_, y_train_, X_test_, y_test_, y_mean, y_std = normalize_data(
                X_train, y_train, X_test, y_test
            )
            x_min, x_max = get_input_range(X_train_, X_test_)
            rand_generator = MeasureSetGenerator(X_train_, x_min, x_max, 0.7)
            self.split_generators.append(rand_generator)

    def get_split_generator(self, split_no):
        return self.split_generators[split_no]

    def get(self, n_nodes, split_no):
        return self.split_generators[split_no].get(n_nodes)


def create_sampled_uci_grid(
    n_nodes,
    split_no=0,
    uci_generators=None,
    dtype=torch.get_default_dtype(),
):
    return torch.tensor(uci_generators.get(n_nodes, split_no), dtype=dtype)
