""" Extract characteristics from functions. """

import torch


def compute_variance(y_batch):
    """
    Compute the variance for each function in the batch.

    Args:
        y_batch (torch.Tensor): Tensor of shape (batch_size, sequence_length).

    Returns:
        torch.Tensor: Variance of each function, shape (batch_size,).
    """
    variance = torch.var(y_batch, dim=1, unbiased=False)
    return variance


def compute_mean(y_batch):
    """
    Compute the mean for each function in the batch.

    Args:
        y_batch (torch.Tensor): Tensor of shape (batch_size, sequence_length).

    Returns:
        torch.Tensor: Mean of each function, shape (batch_size,).
    """
    mean = torch.mean(y_batch, dim=1)
    return mean


def compute_standard_deviation(y_batch):
    """
    Compute the standard deviation for each function in the batch.

    Args:
        y_batch (torch.Tensor): Tensor of shape (batch_size, sequence_length).

    Returns:
        torch.Tensor: Standard deviation of each function, shape (batch_size,).
    """
    std = torch.std(y_batch, dim=1, unbiased=False)
    return std


def compute_skewness(y_batch):
    """
    Compute the skewness for each function in the batch.

    Args:
        y_batch (torch.Tensor): Tensor of shape (batch_size, sequence_length).

    Returns:
        torch.Tensor: Skewness of each function, shape (batch_size,).
    """
    mean = compute_mean(y_batch)
    std = compute_standard_deviation(y_batch)
    skewness = torch.mean(((y_batch - mean[:, None]) / std[:, None]) ** 3, dim=1)
    return skewness


def compute_kurtosis(y_batch):
    """
    Compute the kurtosis for each function in the batch.

    Args:
        y_batch (torch.Tensor): Tensor of shape (batch_size, sequence_length).

    Returns:
        torch.Tensor: Kurtosis of each function, shape (batch_size,).
    """
    mean = compute_mean(y_batch)
    std = compute_standard_deviation(y_batch)
    kurtosis = (
        torch.mean(((y_batch - mean[:, None]) / std[:, None]) ** 4, dim=1) - 3
    )  # Subtract 3 for excess kurtosis
    return kurtosis


def extract_characteristics(y_batch):
    """
    Extract multiple characteristics from a batch of functions.

    Args:
        y_batch (torch.Tensor): Tensor of shape (batch_size, sequence_length).

    Returns:
        dict: A dictionary containing tensors of characteristics.
    """
    characteristics = {}
    characteristics["mean"] = compute_mean(y_batch)
    # characteristics["var"] = compute_variance(y_batch)
    characteristics["std"] = compute_standard_deviation(y_batch)
    characteristics["skewness"] = compute_skewness(y_batch)
    characteristics["kurtosis"] = compute_kurtosis(y_batch)
    return characteristics


def aggregate_characteristics(characteristics):
    """
    Aggregate characteristics over the batch.

    Args:
        characteristics (dict): Dictionary of tensors containing characteristics.

    Returns:
        dict: Dictionary of aggregated characteristics.
    """
    aggregated = {}
    for key, value in characteristics.items():
        aggregated[key] = (torch.mean(value), torch.std(value, unbiased=False))
    return aggregated


def compare_characteristics(y_batch1, y_batch2):
    c1 = aggregate_characteristics(extract_characteristics(y_batch1))
    c2 = aggregate_characteristics(extract_characteristics(y_batch2))
    summaries = []
    for k, (m1, s1) in c1.items():
        m2, s2 = c2[k]
        summaries.append(f"{k} = {m1:.2f}+/-{s1:.2f} vs {m2:.2f}+/-{s2:.2f}")
    return c1, c2, ",  ".join(summaries)


def compute_distributional_stats(batch_learned, batch_target):

    (
        characteristics_target,
        characteristics_learning,
        comparison_str,
    ) = compare_characteristics(batch_target, batch_learned)

    stats = {}
    for k, (m, s) in characteristics_learning.items():
        stats["learning_" + k + "_mean"] = m.item()
        stats["learning_" + k + "_std"] = s.item()
    for k, (m, s) in characteristics_target.items():
        stats["target_" + k + "_mean"] = m.item()
        stats["target_" + k + "_std"] = s.item()
    return comparison_str, stats
