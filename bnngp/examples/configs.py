from typing import NamedTuple

from .generators import get_generator, get_single_layer_bnn_params
from .data_ranges import get_data_ranges
import logging


class _BNNMetadata(NamedTuple):
    ma: float = 0.0
    mu: float = 0.0
    mb: float = 0.0
    mv: float = 0.0
    sa: float = 1.0
    su: float = 1.0
    sb: float = 1.0
    wv: float = 0.001
    activation: callable = lambda x: x * 0.0  # dummy activation


def get_parameters_for_priors_on_weights(priors_config_set_no=1):
    """
    Retrieves parameters for priors on weights for the no.

    Args:
        config_set_no (int, optional): Configuration set number. Defaults to 1.

    Returns:
        _BNNMetadata: Metadata containing parameters for priors on weights.
    """
    try:
        activation, mv, mb, mu, ma, su, sa, wv, sb = get_single_layer_bnn_params(
            priors_config_set_no
        )
        meta = _BNNMetadata(
            ma=ma,
            mu=mu,
            mb=mb,
            mv=mv,
            sa=sa,
            su=su,
            sb=sb,
            wv=wv,
            activation=activation,
        )
        logging.warning(
            f"[get_parameters_for_priors_on_weights] Using the following parameters for priors on weights: {meta}!"
        )
    except ValueError:
        meta = _BNNMetadata()
        logging.warning(
            f"[get_parameters_for_priors_on_weights] priors_config_set_no={priors_config_set_no}: "
            f"Using dummy set of meta parameters: {meta}!"
        )
    return meta


def get_configs(config_set_no=1, net_width=1_000, input_dim=None, sn2=None):
    generator = get_generator(config_set_no, net_width, input_dim, sn2)
    priors_params = get_parameters_for_priors_on_weights(config_set_no)
    data_min_x, data_max_x = get_data_ranges(config_set_no)
    return generator, priors_params, data_min_x, data_max_x
