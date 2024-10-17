import torch
import numpy as np


class SingleHiddenLayerWideNNWithGaussianPrior(torch.nn.Module):
    """
    A single hidden layer neural network with Gaussian priors on weights and biases.

    Attributes:
        ma (float): Mean for bias of first layer.
        mu (float): Mean for weight of first layer.
        mb (float): Mean for bias of output layer.
        mv (float): Mean for weight of output layer.
        sa (float): Standard deviation for bias of first layer.
        su (float): Standard deviation for weight of first layer.
        sb (float): Standard deviation for bias of output layer.
        wv (float): Standard deviation for weight of output layer.
        width (int): Number of neurons in the hidden layer.
        activation (torch.nn.Module): Activation function.
        fc1 (torch.nn.Linear): Fully connected layer 1.
        fc2 (torch.nn.Linear): Fully connected layer 2.
    """

    def __init__(
        self,
        ma,
        mu,
        mb,
        mv,
        sa,
        su,
        sb,
        wv,
        width=1000,
        activation=torch.nn.Tanh(),
        indim=1,
        outdim=1,
    ):
        """
        Initializes the network with Gaussian priors on the weights and biases.

        Args:
            ma (float): Mean for bias of first layer.
            mu (float): Mean for weight of first layer.
            mb (float): Mean for bias of output layer.
            mv (float): Mean for weight of output layer.
            sa (float): Standard deviation for bias of first layer.
            su (float): Standard deviation for weight of first layer.
            sb (float): Standard deviation for bias of output layer.
            wv (float): Standard deviation for weight of output layer.
            width (int, optional): Number of neurons in the hidden layer. Default is 1000.
            activation (torch.nn.Module, optional): Activation function. Default is torch.nn.Tanh().
            indim (int, optional): Input dimension. Default is 1.
            outdim (int, optional): Output dimension. Default is 1.
        """
        super().__init__()
        self.width = width
        self.activation = activation

        if outdim != 1:
            raise ValueError(
                "[SingleHiddenLayerWideNNWithGaussianPrior] Only outdim=1 is supported."
            )

        self.ma, self.mu, self.mb, self.mv = ma, mu, mb, mv
        self.sa, self.su, self.sb, self.wv = sa, su, sb, wv

    def forward(self, x, n_samples=1):
        """
        Forward pass through the network. Generates n_samples using different
        prior samples for each forward pass.

        Args:
            x (torch.Tensor): Input tensor.
            n_samples (int, optional): Number of samples to generate. Default is 1.

        Returns:
            torch.Tensor: The output tensor(s) of the network. If n_samples > 1,
            returns a tensor of shape (n_samples, *output_shape).
        """
        # Sample weights and biases for all n_samples at once
        fc1_weight_samples = torch.normal(
            self.mu, self.su, size=(n_samples, self.width, x.size(-1))
        )
        fc1_bias_samples = torch.normal(self.ma, self.sa, size=(n_samples, self.width))

        fc2_weight_samples = torch.normal(
            self.mv,
            self.wv / np.sqrt(self.width),
            size=(n_samples, x.size(-1), self.width),
        )
        fc2_bias_samples = torch.normal(self.mb, self.sb, size=(n_samples, x.size(-1)))

        # Expand input x to (n_samples, batch_size, input_dim)
        x = x.unsqueeze(0).expand(n_samples, *x.shape)

        # Perform the forward pass with sampled weights and biases
        out = torch.bmm(fc1_weight_samples, x.transpose(1, 2)).transpose(
            1, 2
        ) + fc1_bias_samples.unsqueeze(1)
        out = self.activation(out)
        out = torch.bmm(
            out, fc2_weight_samples.transpose(1, 2)
        ) + fc2_bias_samples.unsqueeze(1)

        return out.squeeze(0) if n_samples == 1 else out.squeeze().T


class SingleHiddenLayerWideNNWithGaussianPriorLegacy(torch.nn.Module):
    def __init__(
        self,
        ma,
        mu,
        mb,
        mv,
        sa,
        su,
        sb,
        wv,
        width=1000,
        activation=torch.nn.Tanh(),
        indim=1,
        outdim=1,
    ):
        super().__init__()

        if outdim != 1:
            raise ValueError(
                "[SingleHiddenLayerWideNNWithGaussianPrior] Only outdim=1 is supported."
            )

        self.width = width
        self.fc1 = torch.nn.Linear(indim, width)
        self.fc2 = torch.nn.Linear(width, outdim)
        self.activation = activation

        def sample_from_prior(self):
            # override the weights
            self.fc1.weight.data.normal_(mean=mu, std=su)
            self.fc1.bias.data.normal_(mean=ma, std=sa)

            self.fc2.weight.data.normal_(mean=mv, std=wv / np.sqrt(self.width))
            self.fc2.bias.data.normal_(mean=mb, std=sb)

        self.sample_from_prior = sample_from_prior

    def forward(self, x):
        self.sample_from_prior(self)

        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class SingleHiddenLayerWideNNWithUniformPrior(torch.nn.Module):
    def __init__(
        self,
        la,
        ua,
        lu,
        uu,
        lb,
        ub,
        lv,
        uv,
        width=1000,
        activation=torch.nn.Tanh(),
        indim=1,
        outdim=1,
    ):
        super().__init__()
        self.width = width
        self.fc1 = torch.nn.Linear(indim, width)
        self.fc2 = torch.nn.Linear(width, outdim)
        self.activation = activation

        def sample_from_prior(self):
            # override the weights
            self.fc1.weight.data.uniform_(lu, uu)
            self.fc1.bias.data.uniform_(la, ua)

            self.fc2.weight.data.uniform_(lv, uv)
            self.fc2.bias.data.uniform_(lb, ub)

        self.sample_from_prior = sample_from_prior

    def forward(self, x):
        self.sample_from_prior(self)

        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    lengthscale_precision = 1.0

    batch_size = 123

    generator = SingleHiddenLayerWideNNWithGaussianPrior(
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        outdim=1,
    )

    # import grids

    # train_grid_x = grids.create_uniform_grid(-10, 10, 100, n_dims=3)
    train_grid_x = torch.arange(-10.0, 10.0, 0.1)[:, None]

    print("train_grid_x.shape=", train_grid_x.shape)
    print("a sample from generator=", generator(train_grid_x).shape)

    batch_target = torch.hstack(
        [generator(train_grid_x) for _ in range(batch_size)]
    ).mT  # shape = (batch_size, train_grid_x.len)
    print(f"{batch_size} samples: {batch_target.shape}.T")

    batch_target = generator(train_grid_x, n_samples=batch_size).T
    print(f"{batch_size} samples at once: {batch_target.shape}.T")

    train_grid_x = torch.tensor([-10.0, -5.0, 0.0, 5.0, 10.0])[:, None]
    print("3 samples on grid with 5 elements: ", generator(train_grid_x, 3))
