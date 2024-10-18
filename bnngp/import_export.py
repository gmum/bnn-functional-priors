""" Saving and loading learned parameters."""

import torch
import json
import io
import base64
import pyro.distributions as dist
import numpy as np
from collections.abc import Iterable

import logging


import pickle
import activations


def pickle_to_str(obj):
    return base64.b64encode(pickle.dumps(obj)).decode("utf-8")


def pickle_object(obj):
    return {
        "name": str(obj),
        "type": "pickle",
        "pickle": pickle_to_str(obj),
    }


def unpickle_from_str(obj):
    return pickle.loads(base64.b64decode(obj))


def unpickle_object(dct):
    assert dct["type"] == "pickle"
    return unpickle_from_str(dct["pickle"])


def pack_resursively(obj):

    if isinstance(obj, dict):
        return {k1: pack_resursively(v1) for k1, v1 in obj.items()}

    elif isinstance(obj, list) or isinstance(obj, tuple):
        return list(pack_resursively(v1) for v1 in obj)

    elif isinstance(obj, float) or isinstance(obj, int) or isinstance(obj, str):
        return obj

    else:
        try:
            return json.dumps(obj)

        except Exception as e:
            logging.debug(
                f"[import_export] Failed to dump {obj} to JSON ({e}). Pickling instead."
            )

            try:
                return pickle_object(obj)
            except Exception as e:
                logging.warning(
                    f"[import_export] Failed to pickle {obj} ({e}). The object will not be recoverable."
                )
                return obj  # will be saved thanks to default=str below


def unpack_recursively(dct):
    dct_unpacked = {}
    for k, v in dct.items():
        if isinstance(v, dict):
            if v.get("type") == "pickle":
                dct_unpacked[k] = unpickle_object(v)
            else:
                dct_unpacked[k] = unpack_recursively(v)

        else:
            dct_unpacked[k] = v

    return dct_unpacked


def encode_gaussian_priors(parameters):
    def unify_name(name):
        replacements = {"fc1.": "layer1.", "fc2.": "layer2."}
        for s, r in replacements.items():
            name = name.replace(s, r)
        return name

    # unify names and convert values to float
    parameters = {unify_name(n): v.item() for n, v in parameters.items()}

    # if no mean is provided, assume it is 0
    parameters["layer1.weight.loc"] = parameters.get("layer1.weight.loc", 0.0)
    parameters["layer2.weight.loc"] = parameters.get("layer2.weight.loc", 0.0)
    parameters["layer1.bias.loc"] = parameters.get("layer1.bias.loc", 0.0)
    parameters["layer2.bias.loc"] = parameters.get("layer2.bias.loc", 0.0)
    return parameters


def decode_gaussian_priors2pyro(prior, target_net_width):
    """Converts Gaussian prior description into Pyro distributions for single-hidden layer NN with.

    Args:
        prior (dict): Gaussian prior description
        net_width (int): network target (desired) width

    Returns:
        dict: dictionary {layer[1/2].[weight/bias]: Normal(.)}
    """
    src_net_width = prior["net_width"]
    vals = prior["values"]

    if src_net_width != target_net_width:
        logging.warning(
            f"[import_export][decode_gaussian_priors] The priors trained for net_width={src_net_width},"
            f"but loaded for net_width={target_net_width}"
        )

    layer2_weight_scale = (
        vals["layer2.weight.scale"] * np.sqrt(src_net_width) / np.sqrt(target_net_width)
    )

    result = {
        "layer1.weight": dist.Normal(
            vals["layer1.weight.loc"], vals["layer1.weight.scale"]
        ),
        "layer1.bias": dist.Normal(vals["layer1.bias.loc"], vals["layer1.bias.scale"]),
        "layer2.weight": dist.Normal(vals["layer2.weight.loc"], layer2_weight_scale),
        "layer2.bias": dist.Normal(vals["layer2.bias.loc"], vals["layer2.bias.scale"]),
    }

    return result


def decode_gaussian_priors(*args, **kwargs):
    return decode_gaussian_priors2pyro(*args, **kwargs)


def encode_activation(activation):
    if isinstance(activation, str):
        return {"activation": {"name": activation, "type": "gp_activations"}}

    return {"activation": pickle_object(activation)}


def decode_activation(activation):
    if activation["type"] == "gp_activations":
        return activations.get_activation(activation)

    elif activation["type"] == "pickle":
        activation_obj = unpickle_from_str(activation["pickle"])

        # Bugfix for NNActivation: pyro fails when object attributes are lists
        if hasattr(activation_obj, "fc") and isinstance(activation_obj.fc, list):
            activation_obj.fc = torch.nn.Sequential(*activation_obj.fc)
            logging.warning(
                "[import_export][decode_activation] Converting list field <fc> to Sequential in activation object."
            )
        return activation_obj

    raise ValueError(f"Unknown activation type {activation['type']}!")


def export_parameters(priors_parameters, net_width, prior_type, activation=None):
    """Converts parameters dictionary to a standardized form."""
    if prior_type == "gaussian":
        values = encode_gaussian_priors(priors_parameters)

    elif prior_type == "hypernet":
        # values = _encode_torch_module(prior)
        values = pickle_to_str(priors_parameters)

    else:
        raise ValueError(f"Unknown prior type {prior_type}!")

    encoded_parameters = {
        "prior": {
            "type": prior_type,
            "net_width": net_width,
            "values": values,
        }
    }

    if activation is not None:
        encoded_parameters.update(encode_activation(activation))

    return encoded_parameters


def import_parameters(encoded_parameters, target_net_width=None, framework="pyro"):
    """Converts parameters dictionary to pyro distributions.

    Args:
        net_width (int): network target (desired) width
    """
    if "prior" not in encoded_parameters:
        logging.warning(
            "[import_parameters] Failed to parse encoded_parameters. Perhaps already parsed?"
        )
        return encoded_parameters

    encoded_prior = encoded_parameters["prior"]

    target_net_width = target_net_width or encoded_prior["net_width"]
    prior_type = encoded_prior["type"]

    if prior_type == "gaussian":
        if framework.lower() == "pyro":
            prior_parameters = decode_gaussian_priors(encoded_prior, target_net_width)
        else:
            raise ValueError(
                f"[import_export] I don't know how to decode priors for framework={framework}!"
            )
        if "activation" in encoded_parameters:
            prior_parameters["activation"] = decode_activation(
                encoded_parameters["activation"]
            )
        return prior_parameters

    elif prior_type == "hypernet":
        assert encoded_prior["net_width"] == target_net_width, (
            f"For priors given by hypernet net_width must match the original one = {encoded_prior['net_width']}. "
            "I don't know how to scale variance!"
        )
        prior_parameters = {"hypernet": unpickle_from_str(encoded_prior["values"])}
        if "activation" in encoded_parameters:
            prior_parameters["activation"] = decode_activation(
                encoded_parameters["activation"]
            )
        return prior_parameters

    raise ValueError(f"Unknown prior type {prior_type}")


def _extract_value(results, name, default=None):
    value = default or results.get(name, None)
    if value is None:
        raise ValueError(
            f"{name} must be passed either as a parameter or via results dict!"
        )
    return value


def save_to_json(
    json_filename,
    results,
    net_width=None,
    activation=None,
    prior_type=None,
    pickle_objects=True,
):
    """Save prior parameters to json file."""
    # results_str = str(results).replace("\n", "  ")
    logging.info(f"[import_export] Writing to {json_filename}")

    net_width = _extract_value(results, "net_width", net_width)
    prior_type = _extract_value(results, "prior_type", prior_type)
    activation = _extract_value(results, "activation", activation)
    results.pop("activation", "None")

    results["parameters"] = export_parameters(
        results.pop("priors_parameters", results.get("parameters")),
        net_width=net_width,
        prior_type=prior_type,
        activation=activation,
    )

    if pickle_objects:
        results = pack_resursively(results)

    with open(json_filename, "w") as f:
        json.dump(results, f, default=str)

    return results


def load_from_json(
    json_filename, target_net_width=None, parse_pickles=True, framework="pyro"
):
    with open(json_filename, "r") as f:
        results = json.load(f)

    target_net_width = _extract_value(results, "net_width", target_net_width)

    if "parameters" in results:
        results["parameters"] = import_parameters(
            results["parameters"],
            target_net_width=target_net_width,
            framework=framework,
        )

    if parse_pickles:
        results = unpack_recursively(results)

    return results


def load_parameters_from_json(json_filename, net_width=None, **kwargs):
    """Load prior parameters from json file and converts to pyro distributions."""
    return load_from_json(
        json_filename, target_net_width=net_width, parse_pickles=False, **kwargs
    )["parameters"]
