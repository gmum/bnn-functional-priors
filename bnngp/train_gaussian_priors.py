import sys
import utils
import args
import logging
from train_priors_and_activations import main_parameterized as train


def main_parameterized(**kwargs):
    activation = kwargs.pop("activation", "sigma")
    create_parameter_sampler = kwargs.pop(
        "create_parameter_sampler", "create_factorized_sampler_gaussian_zero_loc"
    )
    return train(
        create_parameter_sampler=create_parameter_sampler,
        activation=activation,
        **kwargs
    )


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

    return main_parameterized(**parsed_args)


if __name__ == "__main__":
    main()
