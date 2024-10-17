""" Custom neural network activation functions. """

import torch


class StepActivationHardTanH(torch.nn.Module):
    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, x):
        y = torch.nn.functional.hardtanh(x, -self.threshold, self.threshold)
        y = y + 0.5 * (torch.sign(x + 1.0e-7).detach() + 1.0) - y.detach()
        return 2.0 * y - 1.0


class StepActivation(torch.nn.Module):
    def forward(self, x):
        y = torch.sigmoid(x)
        y = y + 0.5 * (torch.sign(x + 1.0e-7).detach() + 1.0) - y.detach()
        return 2.0 * y - 1.0


def sigma(x, nu_ind=2, ell=0.5):
    """Implements the Matern activation function denoted as sigma(x) in Equation 9.
    sigma(x) corresponds to a Matern kernel, with specified smoothness
    parameter nu and length-scale parameter ell.

    Args:
      x: Input to the activation function
      device: A torch.device object
      nu_ind: Index for choosing Matern smoothness (look at nu_list below)
      ell: Matern length-scale, only 0.5 and 1 available with precalculated scaling coefficients

    Source: https://github.com/AaltoML/stationary-activations/blob/main/notebooks/banana_classification.ipynb
    """
    nu_list = [
        1 / 2,
        3 / 2,
        5 / 2,
        7 / 2,
        9 / 2,
    ]  # list of available smoothness parameters
    nu = torch.tensor(nu_list[nu_ind])  # smoothness parameter
    lamb = torch.sqrt(2 * nu) / ell  # lambda parameter
    v = nu + 1 / 2
    # Precalculated scaling coefficients for two different lengthscales (q divided by Gammafunction at nu + 0.5)
    ell05A = [
        4.0,
        19.595917942265423,
        65.31972647421809,
        176.69358285524189,
        413.0710073859664,
    ]
    ell1A = [
        2.0,
        4.898979485566356,
        8.16496580927726,
        11.043348928452618,
        12.90846898081145,
    ]
    if ell == 0.5:
        A = ell05A[nu_ind]
    if ell == 1:
        A = ell1A[nu_ind]
    y = A * torch.sign(x) * torch.abs(x) ** (v - 1) * torch.exp(-lamb * torch.abs(x))
    y[x < 0] = 0  # Values at x<0 must all be 0
    return y


# default activations which may be used without creating new objects
step_activation_hard_tanh = StepActivationHardTanH()
step_activation = StepActivation()
tanh = torch.nn.Tanh()
relu = torch.nn.ReLU()


def matern_12_l05(x):
    return sigma(x, nu_ind=0, ell=0.5)


def matern_32_l05(x):
    return sigma(x, nu_ind=1, ell=0.5)


def matern_52_l05(x):
    return sigma(x, nu_ind=2, ell=0.5)


def matern_72_l05(x):
    return sigma(x, nu_ind=3, ell=0.5)


def matern_12_l1(x):
    return sigma(x, nu_ind=0, ell=1.0)


def matern_32_l1(x):
    return sigma(x, nu_ind=1, ell=1.0)


def matern_52_l1(x):
    return sigma(x, nu_ind=2, ell=1.0)


def matern_72_l1(x):
    return sigma(x, nu_ind=3, ell=1.0)
