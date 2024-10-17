import torch
import torch.nn as nn
import functools
import logging


class RBF(torch.nn.Module):
    def forward(self, x):
        return torch.exp(-(x**2))


class Hypernet(torch.nn.Module):
    def __init__(self, indim, outdim, activations=[RBF()], hidden=[64]):
        super().__init__()
        dims = [indim] + hidden + [outdim]
        activations += [None]
        print(f"[Hypernet] dims = {dims}")

        layers = []
        assert len(dims) - 1 == len(activations)
        for i, (layer_in, activation) in enumerate(zip(dims[:-1], activations)):
            layer_out = dims[i + 1]
            print(f"Adding ({layer_in}->{layer_out}) -> {activation}")
            layers.append(torch.nn.Linear(layer_in, dims[i + 1]))
            if activation is not None:
                layers.append(activation)
        self.f = torch.nn.Sequential(*layers)

    def forward(self, condition):
        return self.f(condition)


def create_hypernet(indim, outdim, arch_no=0):
    if arch_no == 0:
        return Hypernet(
            activations=[RBF(), RBF(), RBF()],
            hidden=[128, 32, 8],
            indim=indim,
            outdim=outdim,
        )
    elif arch_no == 1:
        return Hypernet(
            activations=[RBF(), RBF()],
            hidden=[128, 64],
            indim=indim,
            outdim=outdim,
        )
    raise ValueError(f"Unsupported hypernet arch_no={arch_no}!")


def calc_total_numel(target_param_shapes):
    """Computes the total number of parameters.

    Args:
        target_param_shapes (List[Tuple]): List of tensor shapes.
    """
    return sum(functools.reduce(lambda a, b: a * b, s) for s in target_param_shapes)


def split_and_reshape_tensor(flat_tensor, target_param_shapes):
    """
    Reshape a flat tensor into a list of tensors with specified shapes.

    Parameters:
    - flat_tensor (torch.Tensor): The input flat tensor.
    - target_param_shapes (List[Tuple]): List of target shapes for the output tensors.

    Returns:
    - List[torch.Tensor]: List of tensors reshaped to the target shapes.
    """
    if not isinstance(flat_tensor, torch.Tensor):
        raise TypeError("flat_tensor must be a torch.Tensor")

    # Compute total number of elements required by the target shapes
    total_elements_needed = sum(
        [torch.prod(torch.tensor(shape)) for shape in target_param_shapes]
    )

    # Check if the total number of elements matches the flat tensor length
    if total_elements_needed != flat_tensor.numel():
        raise ValueError(
            "The total number of elements in target_param_shapes does not match the length of flat_tensor"
        )

    # Reshape the tensor according to each shape in target_param_shapes
    flat_tensor = flat_tensor.flatten()
    offset = 0
    reshaped_tensors = []
    for shape in target_param_shapes:
        num_elements = torch.prod(torch.tensor(shape))
        # Slice the flat tensor from offset to offset + num_elements and reshape
        reshaped_tensor = flat_tensor[offset : offset + num_elements].reshape(shape)
        reshaped_tensors.append(reshaped_tensor)
        offset += num_elements

    return reshaped_tensors


class ConditioningHypernetA(Hypernet):
    """Wraps Hypernet so its API would match ConditioningHypernetB."""

    def __init__(self, indim, target_param_shapes, hidden=[64], activations=None):
        # Validate and setup activations
        if activations is None:
            logging.info(
                "[ConditioningHypernetB] Default to RBF if no activations are provided"
            )
            activations = [RBF()] * len(hidden)
        if len(hidden) != len(activations):
            raise ValueError(
                "Each hidden layer size must have a corresponding activation function"
            )
        super().__init__(
            activations=activations,
            hidden=hidden,
            indim=indim,
            outdim=calc_total_numel(target_param_shapes),
        )
        self.indim = indim
        self.target_param_shapes = target_param_shapes

    def forward(self, condition):
        assert (
            self.indim == condition.numel()
        ), f"[ConditioningHypernetA] Expected input with {self.indim} elements. Got {condition.shape}."

        return split_and_reshape_tensor(
            super().forward(condition.flatten()), self.target_param_shapes
        )


class ConditioningHypernetB(nn.Module):
    """
    A neural network that generates parameters for another network based on a given condition.

    Attributes:
        indim (int): The dimensionality of the conditioning input.
        target_param_shapes (list of tuples): A list of shapes for the target network's parameters.
        hidden (list of int): A list of sizes for each hidden layer.
        activations (list of torch.nn.modules.activation): A list of activation functions for each hidden layer.
    """

    def __init__(self, indim, target_param_shapes, hidden=[128, 128], activations=None):
        """
        Initializes the HyperNetwork.

        Args:
            indim (int): The dimensionality of the conditioning input.
            target_param_shapes (list of tuples): A list of shapes for the target network's parameters.
            hidden (list of int): A list of sizes for each hidden layer.
            activations (list of torch.nn.modules.activation): A list of activation functions for each hidden layer.
        """
        super(ConditioningHypernetB, self).__init__()
        self.condition_dim = indim
        self.target_param_shapes = target_param_shapes

        # Validate and setup activations
        if activations is None:
            logging.info(
                "[ConditioningHypernetB] Default to RBF if no activations are provided"
            )
            activations = [RBF()] * len(hidden)
        if len(hidden) != len(activations):
            raise ValueError(
                "Each hidden layer size must have a corresponding activation function"
            )

        # Construct hidden layers
        layers = []
        input_size = indim
        for size, activation in zip(hidden, activations):
            layers.append(nn.Linear(input_size, size))
            layers.append(activation)
            input_size = size
        self.hidden_layers = nn.Sequential(*layers)

        self.output_layers = nn.ModuleList()
        for shape in target_param_shapes:
            self.output_layers.append(
                nn.Linear(hidden[-1], int(torch.prod(torch.tensor(shape))))
            )

    def forward(self, condition):
        """
        Forward pass through the HyperNetwork.

        Args:
            condition (Tensor): The conditioning input tensor. Must be at least 2D.

        Returns:
            list of Tensors: A list of parameters for the target network.
        """
        # Reshape to shape = batch dim x num_conditions dim
        if len(condition.shape) == 0:
            condition = condition.unsqueeze(0).unsqueeze(0)
        elif len(condition.shape) == 1:
            condition = condition.unsqueeze(0)

        hidden_rep = self.hidden_layers(condition)
        params = []
        for layer in self.output_layers:
            param = layer(hidden_rep)
            params.append(
                param.view(param.size(0), *self.target_param_shapes[len(params)])
            )

        return params


def create_conditioning_hypernet(indim, target_param_shapes, arch_no=0):
    if arch_no == 0:
        return ConditioningHypernetB(
            activations=[RBF(), RBF(), RBF()],
            hidden=[128, 32, 8],
            indim=indim,
            target_param_shapes=target_param_shapes,
        )

    elif arch_no == 1:
        return ConditioningHypernetB(
            activations=[RBF(), RBF()],
            hidden=[128, 64],
            indim=indim,
            target_param_shapes=target_param_shapes,
        )

    elif arch_no == 2:
        return ConditioningHypernetB(
            activations=[nn.ReLU(), nn.ReLU()],
            hidden=[128, 64],
            indim=indim,
            target_param_shapes=target_param_shapes,
        )

    elif arch_no == 3:
        return ConditioningHypernetB(
            activations=[nn.ReLU()],
            hidden=[64],
            indim=indim,
            target_param_shapes=target_param_shapes,
        )

    elif arch_no == 4:
        return ConditioningHypernetB(
            activations=[RBF(), nn.SiLU(), nn.SiLU()],
            hidden=[128, 64, 64],
            indim=indim,
            target_param_shapes=target_param_shapes,
        )

    elif arch_no == 5:
        return ConditioningHypernetB(
            activations=[RBF(), RBF(), RBF()],
            hidden=[128, 64, 64],
            indim=indim,
            target_param_shapes=target_param_shapes,
        )

    elif arch_no == 6:
        return ConditioningHypernetB(
            activations=[RBF(), RBF(), RBF()],
            hidden=[128, 64, 32],
            indim=indim,
            target_param_shapes=target_param_shapes,
        )

    raise ValueError(f"Unsupported hypernet arch_no={arch_no}!")


if __name__ == "__main__":
    from activations.learnable_conditioned_nn_activations import (
        NNActivationConditioned,
        NNActivationConditionedEachRow,
    )

    print("NNActivationConditioned")

    activation = NNActivationConditioned(
        hidden_layers=2, width=5, indim=1, activation=nn.ReLU()
    )
    print(
        "activation.get_conditional_params_shapes() = ",
        activation.get_conditional_params_shapes(),
    )
    hypernetwork = ConditioningHypernetB(
        indim=3,
        target_param_shapes=activation.get_conditional_params_shapes(),
        hidden=[128, 128],
    )

    # Example usage
    input_data = torch.arange(-10, 10.0, 1)  # Example input (10 samples, 1-dimensional)
    condition = torch.tensor([1.0, 2.0, 3])  # Example \lambda values (3-dimensional)

    params = hypernetwork(condition)
    print("hypernetwork params = ", [p.shape for p in params])

    output = activation(input_data, params)
    print(output.shape)

    print("NNActivationConditionedEachRow")

    activation = NNActivationConditionedEachRow(
        hidden_layers=2, width=5, indim=1, activation=nn.ReLU()
    )
    print(
        "activation.get_conditional_params_shapes() = ",
        activation.get_conditional_params_shapes(),
    )
    hypernetwork = ConditioningHypernetB(
        indim=3,
        target_param_shapes=activation.get_conditional_params_shapes(),
        hidden=[128, 128],
    )
    # Example usage
    input_data = torch.arange(-10, 10.0, 1)  # Example input (20 samples, 1-dimensional)
    condition = torch.randn(20, 3)  # Example \lambda values (20 samples, 3-dimensional)

    params = hypernetwork(condition)
    print("hypernetwork params = ", [p.shape for p in params])

    output = activation(input_data, params)
    print(output.shape)

    ###############################################
