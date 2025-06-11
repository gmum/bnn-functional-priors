from torch.utils.data import TensorDataset, DataLoader
import hashlib

import numpy as np
import torch

# TODO - to be set accordingly to the papers
minibatch_training_size = 1024
batch_size = 1024
train_prop = 6 / 9
tt_seed = 0
preprocess_method = "scaling"
datafile = "individual+household+electric+power+consumption.npy"


def standard_scale(x_train_r, y_train_r, x_test_r, y_test_r):
    """
    Scales the training and test data using the mean and standard deviation of the training data.
    The function returns the standardized training and test data along with the mean and standard deviation of the training target values.
    This ensures that both training and test data are on the same scale, which is important for many machine learning algorithms.
    """
    x_train_r_mean = x_train_r.mean(axis=0)
    x_train_r_std = x_train_r.std(axis=0)
    y_train_r_mean = y_train_r.mean()
    y_train_r_std = y_train_r.std()

    x_train = (x_train_r - x_train_r_mean) / x_train_r_std
    y_train = (y_train_r - y_train_r_mean) / y_train_r_std

    x_test = (x_test_r - x_train_r_mean) / x_train_r_std
    y_test = (y_test_r - y_train_r_mean) / y_train_r_std

    return x_train, y_train, x_test, y_test, y_train_r_mean, y_train_r_std


def whiten(x_train_r, y_train_r, x_test_r, y_test_r, dim):
    """
    This function whitens the input data by removing the mean and scaling the data
    to have unit variance. It first centers the training data by subtracting the mean,
    then computes the covariance matrix of the centered data. The Cholesky decomposition
    of the covariance matrix is used to transform the data to have unit variance.
    The same transformation is applied to the test data using the mean and Cholesky
    decomposition from the training data. The target values are standardized by
    subtracting the mean and dividing by the standard deviation of the training targets.

    Args:
        x_train_r (torch.Tensor): Raw training input data.
        y_train_r (torch.Tensor): Raw training target data.
        x_test_r (torch.Tensor): Raw test input data.
        y_test_r (torch.Tensor): Raw test target data.
        dim (int): Dimensionality of the input data.

    Returns:
        tuple: Transformed training input data, transformed training target data,
               transformed test input data, transformed test target data,
               mean of the training target data, standard deviation of the training target data.
    """
    x_train_r_mean = x_train_r.mean(axis=0)
    x_train_centered = x_train_r - x_train_r_mean
    x_train_covar = (x_train_centered.T @ x_train_centered) / x_train_centered.shape[0]
    chol_tri = torch.linalg.cholesky(
        x_train_covar + torch.eye(x_train_covar.shape[0]) * 1e-5
    )

    x_train = torch.linalg.solve_triangular(
        chol_tri, x_train_centered.T, upper=False
    ).T / np.sqrt(dim)
    x_test_centered = x_test_r - x_train_r_mean
    x_test = torch.linalg.solve_triangular(
        chol_tri, x_test_centered.T, upper=False
    ).T / np.sqrt(dim)

    y_train_r_mean = y_train_r.mean()
    y_train_r_std = y_train_r.std()
    y_train = (y_train_r - y_train_r_mean) / y_train_r_std
    y_test = (y_test_r - y_train_r_mean) / y_train_r_std

    return x_train, y_train, x_test, y_test, y_train_r_mean, y_train_r_std


def load_raw_data(datafile):
    data_r = torch.tensor(np.load(datafile), dtype=torch.float32, requires_grad=False)
    x_all_r = data_r[:, :-1]
    y_all_r = data_r[:, -1]
    n_all, dim = x_all_r.shape
    return x_all_r, y_all_r, n_all, dim


def split_train_test(x_all_r, y_all_r, n_all, train_prop=train_prop):
    inds = np.arange(n_all)
    np.random.seed(tt_seed)
    np.random.shuffle(inds)
    hsh = hashlib.md5(np.ascontiguousarray(inds)).hexdigest()
    n_train = int(train_prop * n_all)
    n_test = n_all - n_train
    train_inds = inds[:n_train]
    test_inds = inds[n_train:]
    x_train_r = x_all_r[train_inds]
    y_train_r = y_all_r[train_inds]
    x_test_r = x_all_r[test_inds]
    y_test_r = y_all_r[test_inds]
    return x_train_r, y_train_r, x_test_r, y_test_r, n_train, n_test, hsh


def run_preprocessing(preprocess_method, x_train_r, y_train_r, x_test_r, y_test_r, dim):
    if preprocess_method == "scaling":
        (
            x_train,
            y_train,
            x_test,
            y_test,
            y_train_r_mean,
            y_train_r_std,
        ) = standard_scale(x_train_r, y_train_r, x_test_r, y_test_r)

    elif preprocess_method == "whitening":
        x_train, y_train, x_test, y_test, y_train_r_mean, y_train_r_std = whiten(
            x_train_r, y_train_r, x_test_r, y_test_r, dim
        )

    else:
        assert preprocess_method == "none"
        x_train, y_train, x_test, y_test, y_train_r_mean, y_train_r_std = (
            x_train_r,
            y_train_r,
            x_test_r,
            y_test_r,
            y_train_r.mean(),
            y_train_r.std(),
        )

    return x_train, y_train, x_test, y_test, y_train_r_mean, y_train_r_std


def prepare_data(datafile=datafile, preprocess_method=preprocess_method, train_prop=train_prop):
    x_all_r, y_all_r, n_all, dim = load_raw_data(datafile=datafile)

    x_train_r, y_train_r, x_test_r, y_test_r, n_train, n_test, hsh = split_train_test(
        x_all_r, y_all_r, n_all, train_prop=train_prop,
    )

    x_train, y_train, x_test, y_test, y_train_r_mean, y_train_r_std = run_preprocessing(
        preprocess_method, x_train_r, y_train_r, x_test_r, y_test_r, dim
    )
    # if torch.cuda.is_available():
    #     x_train, y_train, x_test, y_test = (
    #         x_train.cuda(),
    #         y_train.cuda(),
    #         x_test.cuda(),
    #         y_test.cuda(),
    #     )
    return x_train, y_train, x_test, y_test, y_train_r_mean, y_train_r_std, hsh


if __name__ == "__main__":
    (
        x_train,
        y_train,
        x_test,
        y_test,
        y_train_r_mean,
        y_train_r_std,
        tt_split_hsh,
    ) = prepare_data()
    dim, n_all, n_train, n_test = (
        x_train.shape[1],
        len(y_train) + len(y_test),
        len(y_train),
        len(y_test),
    )
    print(
        "Data: dim={}, n_all={}, n_train={}, n_test={}".format(
            dim, n_all, n_train, n_test
        )
    )

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=minibatch_training_size, shuffle=True
    )

    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
