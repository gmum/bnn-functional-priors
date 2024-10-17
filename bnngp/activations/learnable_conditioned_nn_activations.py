import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .learnable_activations import _disable_bias, _disable_weight


class NNActivationConditioned(nn.Module):
    """
    A neural network whose parameters come from an external input (e.g. from a hypernetwork).

    Attributes:
        hidden_layers (int): The number of hidden layers in the network.
        width (int): The width of each hidden layer.
        indim (int): The input dimensionality.
        activation (nn.Module): The activation function.
    """

    def __init__(self, hidden_layers=1, width=5, indim=1, activation=nn.ReLU()):
        """
        Args:
            hidden_layers (int): The number of hidden layers in the network.
            width (int): The width of each hidden layer.
            indim (int): The input dimensionality.
            activation (nn.Module): The activation function.
        """
        super().__init__()
        self.width = width
        self.activation = activation
        self.indim = indim
        self.hidden_layers = hidden_layers
        self.num_params = len(self.get_conditional_params_shapes())

    def get_conditional_params_shapes(self):
        """
        Gets the shapes of the parameters for the target network.

        Returns:
            list of tuples: The shapes of the target network's parameters.
        """
        width, indim, hidden_layers = self.width, self.indim, self.hidden_layers
        return (
            [
                (width, indim),  # fci weight
                (width,),  # fci bias
            ]
            + [(width, width) for _ in range(hidden_layers)]  # fc weights
            + [(width,) for _ in range(hidden_layers)]  # fc biases
            + [(indim, width), (indim,)]  # fco weight  # fco bias
        )

    def forward(self, a, conditional_params):
        """
        Forward pass through the conditioned network.

        Args:
            a (Tensor): The input data.

        Returns:
            Tensor: The output of the network.
        """
        params = conditional_params
        assert len(params) == self.num_params, (
            "Passed parameters do not match architecture of the network: "
            f"{len(params)}!={self.num_params}"
        )

        if len(params[0].shape) > 2 and params[0].shape[0] == 1:
            # Remove the batch dimension added by the hypernetwork
            params = [param.squeeze(0) for param in params]

        # Extract parameters
        fci_weight = params[0]
        fci_bias = params[1]
        fc_weights = params[2 : 2 + self.num_params // 2 - 1]
        fc_biases = params[2 + self.num_params // 2 - 1 : -2]
        fco_weight = params[self.num_params - 2]
        fco_bias = params[self.num_params - 1]

        # Forward pass with conditioned parameters
        a = a.unsqueeze(-1)

        a = F.linear(a, fci_weight, fci_bias)
        a = self.activation(a)

        for weight, bias in zip(fc_weights, fc_biases):
            a = F.linear(a, weight, bias)
            a = self.activation(a)

        a = F.linear(a, fco_weight, fco_bias)

        a = a.squeeze(-1)
        return a


class NNActivationConditionedEachRow(NNActivationConditioned):
    def forward(self, a, params):
        """
        Forward pass through the conditioned network, with different conditioning for each row of input data.

        Args:
            a (Tensor): The input data.

        Returns:
            Tensor: The output of the network.
        """
        # Forward pass with conditioned parameters
        outputs = []
        for i in range(a.size(0)):
            single_a = a[i].unsqueeze(0).unsqueeze(-1)  # Make it 2D
            single_params = [
                param[i] for param in params
            ]  # Get parameters for the current instance

            output = super().forward(single_a, single_params)
            outputs.append(output)

        return torch.cat(outputs, dim=0).squeeze(-1)


class PeriodicNNActivationConditioned(nn.Module):
    def __init__(self, input_size=1, n_nodes=10, train_in=False):
        super().__init__()
        self.input_size = 1

        # Linear transformation from input to hidden layer
        self.fc1 = nn.Linear(input_size, n_nodes)
        if not train_in:  # turn of any training -> just broadcast
            _disable_weight(self.fc1)
            _disable_bias(self.fc1)

        self.hidden_size = n_nodes
        self.num_params = len(self.get_conditional_params_shapes())

    def get_conditional_params_shapes(self):
        """
        Gets the shapes of the parameters for the target network.

        Returns:
            list of tuples: The shapes of the target network's parameters.
        """
        width = self.hidden_size
        return [(self.input_size, width), (width,)]

    def forward(self, x, conditional_params):
        fc2_weight = conditional_params[0]
        frequencies = conditional_params[1]

        # Remove superflous dimensions
        if len(fc2_weight.shape) == 3:
            fc2_weight = fc2_weight[0, ...]
        if len(frequencies.shape) == 2:
            frequencies = frequencies[0, ...]

        x = x.unsqueeze(-1)

        # Pass through the first linear layer
        x = self.fc1(x)

        # Apply Fourier-based activations with increasing frequencies
        activations = []
        for i in range(self.hidden_size):
            if i % 2 == 0:
                # Apply sine activation for even-indexed neurons
                activations.append(torch.sin(2 * math.pi * frequencies[i] * x[..., i]))
            else:
                # Apply cosine activation for odd-indexed neurons
                activations.append(torch.cos(2 * math.pi * frequencies[i] * x[..., i]))

        # Stack activations back into a tensor
        x = torch.stack(activations, dim=-1)

        # Pass through the output layer
        x = torch.matmul(x, fc2_weight.T)  # x = self.fc2(x)

        x = x.squeeze(-1)
        return x
