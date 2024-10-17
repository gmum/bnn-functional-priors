import torch
import numpy as np
import json
import random
import logging


from activations.gp_activations import sigma
from utils_plotting import *

from grids import MeasureSetGenerator, UCIGenerators  # backward compatibility imports


try:
    from rational.utils.find_init_weights import find_weights

    def find_rational_activation_weights(activation=sigma):
        weights = find_weights(activation)
        return weights

except Exception as e:
    logging.warning(f"WARNING: rational-activations are not installed ({e})!")


def setup_torch(device=None, dtype=None, num_threads=None):
    device = (device or "cuda:0",)
    dtype = dtype or "FloatTensor"
    setup_torch.device_name = device

    if ("gpu" in device or "cuda" in device) and torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda." + dtype)
        env = torch.cuda
        logging.info("[setup_torch] Using GPU")
        setup_torch.device_name = device

    else:
        torch.set_default_tensor_type("torch." + dtype)
        env = torch
        logging.info("[setup_torch] Using CPU")
        setup_torch.device_name = "cpu"

    if num_threads:
        torch.set_num_threads(num_threads)
    logging.info(f"[setup_torch] #threads={torch.get_num_threads()}")
    return env, torch.get_default_dtype()


setup_torch.device_name = None


def get_device_name():
    return setup_torch.device_name


def save_configuration(filename, cfg_dict):
    return json.dump(cfg_dict, open(filename, "w"), default=str)


def load_configuration(filename):
    return json.load(open(filename, "r"))


def freeze_function(func):
    """Runs function only once and then always returns a saved result."""

    def caching_func(*args, **kwargs):
        if caching_func.result is None:
            caching_func.result = func(*args, **kwargs)
        return caching_func.result

    caching_func.result = None
    return caching_func


def set_random_seed(seed):
    if seed is None:
        return

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
