import torch
import torch.nn as nn
import math


def rbf(a):
    return torch.exp(-(a**2))


class ActivationsCombination(nn.Module):
    def get_conditional_params_shapes(self):
        return []

    def __init__(self, nodes_per_activation=2):
        super().__init__()
        self.nodes_per_activation = nodes_per_activation

        self.activations = [nn.ReLU(), nn.Sigmoid(), nn.Tanh(), torch.cos, rbf]
        width = nodes_per_activation * len(self.activations)

        self.fc1 = nn.Linear(1, width)
        self.fc2 = nn.Linear(width, 1)

    def forward(self, a, conditional_params=None):
        a = a.unsqueeze(-1)

        a = self.fc1(a)
        activities = []
        for i, activation in enumerate(self.activations):
            s = self.nodes_per_activation * i
            e = self.nodes_per_activation * (i + 1)
            activities.append(activation(a[..., s:e]))
        a = torch.concat(activities, dim=-1)
        a = self.fc2(a)

        a = a.squeeze(-1)
        return a


class NNActivation(nn.Module):
    def get_conditional_params_shapes(self):
        return []

    def __init__(self, hidden_layers=1, width=5, indim=1, activation=nn.ReLU()):
        super().__init__()
        self.width = width
        self.activation = activation

        self.fci = nn.Linear(indim, width)
        self.fc = torch.nn.Sequential(
            *[nn.Linear(width, width) for _ in range(hidden_layers)]
        )
        self.fco = nn.Linear(width, indim)

    def forward(self, a, conditional_params=None):
        a = a.unsqueeze(-1)

        a = self.fci(a)
        a = self.activation(a)

        for fc in self.fc:
            a = fc(a)
            a = self.activation(a)

        a = self.fco(a)

        a = a.squeeze(-1)
        return a


def _disable_weight(linear):
    with torch.no_grad():
        linear.weight.fill_(1.0)
    linear.weight.requires_grad = False


def _disable_bias(linear):
    with torch.no_grad():
        linear.bias.fill_(0.0)
    linear.bias.requires_grad = False


class PeriodicNN(nn.Module):
    def __init__(
        self,
        input_size=1,
        n_nodes=10,
        output_size=1,
        # by default we train only frequencies and amplitudes (aka fourier2 activation)
        train_freqs=True,
        train_in=False,
        train_out="weight",
    ):
        super(PeriodicNN, self).__init__()

        # Linear transformation from input to hidden layer
        self.fc1 = nn.Linear(input_size, n_nodes)
        if not train_in:
            _disable_weight(self.fc1)
            _disable_bias(self.fc1)
        elif train_in == "bias":
            _disable_weight(self.fc1)
        elif train_in == "weight":
            _disable_bias(self.fc1)

        # Linear transformation from hidden layer to output
        self.fc2 = nn.Linear(n_nodes, output_size)
        if not train_out:
            _disable_weight(self.fc2)
            _disable_bias(self.fc2)
        elif train_out == "bias":
            _disable_weight(self.fc2)
        elif train_out == "weight":
            _disable_bias(self.fc2)

        # Define the frequencies for the Fourier basis activations
        frequencies = torch.arange(1, n_nodes + 1).float()
        self.frequencies = (
            torch.nn.Parameter(frequencies) if train_freqs else frequencies
        )

        self.hidden_size = n_nodes

    def forward(self, x, conditional_params=None):
        x = x.unsqueeze(-1)

        # Pass through the first linear layer
        x = self.fc1(x)

        # Apply Fourier-based activations with increasing frequencies
        activations = []
        for i in range(self.hidden_size):
            if i % 2 == 0:
                # Apply sine activation for even-indexed neurons
                activations.append(
                    torch.sin(2 * math.pi * self.frequencies[i] * x[..., i])
                )
            else:
                # Apply cosine activation for odd-indexed neurons
                activations.append(
                    torch.cos(2 * math.pi * self.frequencies[i] * x[..., i])
                )

        # Stack activations back into a tensor
        x = torch.stack(activations, dim=-1)

        # Pass through the output layer
        x = self.fc2(x)

        x = x.squeeze(-1)
        return x


# For backward compatibility
class FourierNN(PeriodicNN):
    def __init__(
        self,
        *args,
        hidden_size=10,
        **kwargs,
    ):
        return super().__init__(*args, n_nodes=hidden_size, **kwargs)


class NN2Fourier(nn.Module):
    def get_conditional_params_shapes(self):
        return []

    def __init__(
        self, indim=1, output_size=1, hidden_layers=0, width=5, n_nodes=10, activation=nn.SiLU()
    ):
        super().__init__()
        self.width = width or n_nodes//2
        self.activation = activation

        self.fci = nn.Linear(indim, width)
        self.fc = torch.nn.Sequential(
            *[nn.Linear(width, width) for _ in range(hidden_layers)]
        )
        self.fco = nn.Linear(width, n_nodes)
        _disable_weight(self.fco)
        _disable_bias(self.fco)

        # Define the frequencies for the Fourier basis activations
        self.n_nodes = n_nodes                
        frequencies = torch.arange(1, n_nodes + 1).float()
        self.frequencies = torch.nn.Parameter(frequencies)
        
        # Linear transformation from hidden layer to output
        self.fc2 = nn.Linear(n_nodes, output_size)
        _disable_bias(self.fc2)        
        
    def forward(self, a, conditional_params=None):
        a = a.unsqueeze(-1)

        a = self.fci(a)
        a = self.activation(a)

        for fc in self.fc:
            a = fc(a)
            a = self.activation(a)

        a = self.fco(a)
        
        #######################################################################
        
        # Apply Fourier-based activations with increasing frequencies
        activations = []
        for i in range(self.n_nodes):
            if i % 2 == 0:
                # Apply sine activation for even-indexed neurons
                activations.append(
                    torch.sin(2 * math.pi * self.frequencies[i] * a[..., i])
                )
            else:
                # Apply cosine activation for odd-indexed neurons
                activations.append(
                    torch.cos(2 * math.pi * self.frequencies[i] * a[..., i])
                )

        # Stack activations back into a tensor
        a = torch.stack(activations, dim=-1)

        # Pass through the output layer
        a = self.fc2(a)        

        a = a.squeeze(-1)
        return a
