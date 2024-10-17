import random
from examples import configs

# generation of varying settings
def yield_input_random_shifts():
    while True:
        yield {"input_shift": random.uniform(-20, 20)}


def yield_input_random_shifts_with_warmup(warmup_n_zeros=500):
    for _ in range(warmup_n_zeros):
        yield {"input_shift": 0.0}
    while True:
        yield {"input_shift": random.uniform(-20, 20)}


def yield_input_random_shifts_with_longer_warmup(warmup_n_zeros=1000):
    for _ in range(warmup_n_zeros):
        yield {"input_shift": 0.0}
    while True:
        yield {"input_shift": random.uniform(-20, 20)}


def yield_input_random_scaling():
    while True:
        yield {"range_scaling": random.uniform(0.1, 3)}


def yield_input_random_shifts_and_scaling():
    while True:
        yield {
            "input_shift": random.uniform(-20, 20),
            "range_scaling": random.uniform(0.1, 3),
        }


input_transformations = {
    "shift_and_scaling": yield_input_random_shifts_and_scaling,
    "shift": yield_input_random_shifts,
    "shift_warmup": yield_input_random_shifts_with_warmup,
    "shift_warmup2": yield_input_random_shifts_with_longer_warmup,
    "scaling": yield_input_random_scaling,
}


def get_target_hyperparams_generator(conditioning_input_transformation):
    return input_transformations[conditioning_input_transformation]()


def get_create_target_generator_func():
    # generator does not to be changed
    fixed_target_generator = configs.get_configs(config_set_no=-11)[0]

    def create_target_generator(**kwargs):
        return fixed_target_generator

    return create_target_generator


def get_evaluation_func():
    # not implemented yet
    return None
