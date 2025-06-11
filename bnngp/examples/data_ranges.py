import logging


DATA_RANGES_1D = {
    # BNN Configurations
    1: (-1.0, 1.0),
    2: (-1.0, 1.0),
    3: (-10.0, 10.0),
    # GP Configurations
    -111: (
        -3.75,
        3.75,
    ),  # matching setting for banana classification in Stationary Activations ...
    -1: (-3.0, 3.0),
    -2: (-3.0, 3.0),
    -3: (-3.0, 3.0),
    -11: (-3.0, 3.0),
    -12: (-3.0, 3.0),
    -13: (-3.0, 3.0),
    -4: (-3.0, 3.0),
    -5: (-3.0, 3.0),
    # periodic kernel:
    -6: (-0.75, 0.75),
    -7: (-0.75, 0.75),
    -8: (-3.0, 3.0),
    -9: (-3.0, 3.0),
    -81: (-1.5, 1.5),
    -91: (-1.5, 1.5),
    # All You Need Configurations
    1001: (-6.0, 6.0),  # AllYouNeed1DRegression
    # # in the original code for All You Need they rescale the ranges
    # mean, std, eps = 3.89738198, 4.53119575, 1e-10
    # normalize = lambda X: (X - mean) / (std + eps)
    # data_min_x, data_max_x = normalize(data_min_x), normalize(data_max_x)
    # 2000: (-6.0, 6.0),  # UCI (unused)
}


def get_data_ranges(config_set_no):
    if config_set_no not in DATA_RANGES_1D:
        raise ValueError(
            f"Unknown config_set_no={config_set_no}! "
            f"Supported values: {list(DATA_RANGES_1D.keys())}"
        )
    result = DATA_RANGES_1D[config_set_no]
    logging.info(f"[get_data_ranges] Selected input range for config_set_no={config_set_no}: {result}")
    return result
