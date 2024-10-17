"""Piecewise Linear activation function for a 2 Dimensional data (B,C,L) from paper: https://arxiv.org/pdf/2104.03693.pdf"""

import torch
from torch import nn


class PWLA2d(torch.nn.Module):
    """Piecewise Linear activation function for a 2 Dimensional data (B,C,L)
    from paper: https://arxiv.org/pdf/2104.03693.pdf

    Args:
        N = int - number of intervals contained in function
        momentum = float - strength of momentum during the statistics collection phase (Phase I in paper)
    """

    def __init__(self, N=16, momentum=0.9):
        super(PWLA2d, self).__init__()
        self.N = N
        self.momentum = momentum
        # self.Br = torch.nn.Parameter(torch.tensor(10.))
        # self.Bl = torch.nn.Parameter(torch.tensor(-10.))
        self.Br = torch.nn.Parameter(torch.tensor(5.0))
        self.Bl = torch.nn.Parameter(torch.tensor(-5.0))
        self.register_buffer("running_mean", torch.zeros(1))
        self.register_buffer("running_std", torch.ones(1))
        self.Kl = torch.nn.Parameter(torch.tensor(0.0))
        self.Kr = torch.nn.Parameter(torch.tensor(1.0))
        self.Yidx = torch.nn.Parameter(
            nn.functional.relu(
                torch.linspace(self.Bl.item(), self.Br.item(), self.N + 1)
            )
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_std.fill_(1)

    def forward(self, x, mode=1):
        # print(x.shape)
        # print(x)
        if mode == 0:
            mean = x.mean([0, 1, -1])  # {TODO}: Possibly split along channel axis
            std = x.std(
                [0, 1, -1], unbiased=True
            )  # {TODO}: Possibly split along channel axis
            self.running_mean = (self.momentum * self.running_mean) + (
                1.0 - self.momentum
            ) * mean  # .to(input.device)
            self.running_std = (self.momentum * self.running_std) + (
                1.0 - self.momentum
            ) * std
            return nn.functional.relu(x)
        else:
            d = (self.Br - self.Bl) / self.N  # Interval length
            """{TODO} Refactor code"""
            #             Bidx = torch.linspace(self.Bl.item(),self.Br.item(),self.N)#LEFT Interval boundaries
            DATAind = torch.clamp(
                torch.floor((x - self.Bl.item()) / d), 0, self.N - 1
            )  # Number of corresponding interval for X
            Bdata = self.Bl + DATAind * d  # LEFT Interval boundaries
            maskBl = x < self.Bl  # Mask for LEFT boundary
            maskBr = x >= self.Br  # Mask for RIGHT boundary
            maskOther = ~(maskBl + maskBr)  # Mask for INSIDE boundaries
            Ydata = self.Yidx[DATAind.type(torch.int64)]  # Y-value for data
            Kdata = (
                self.Yidx[(DATAind).type(torch.int64) + 1]
                - self.Yidx[DATAind.type(torch.int64)]
            ) / d  # SLOPE for data
            return (
                maskBl * ((x - self.Bl) * self.Kl + self.Yidx[0])
                + maskBr * ((x - self.Br) * self.Kr + self.Yidx[-1])
                + maskOther * ((x - Bdata) * Kdata + Ydata)
            )


class PWLA2dLearnableActivation(nn.Module):

    def get_conditional_params_shapes(self):
        return []

    def __init__(
        self,
        N=16,
        momentum=0.9,
        min_x=None,
        max_x=None,
    ):
        """
        clamp activations outside [min_x, max_x]
        """
        super().__init__()
        self.N = N
        self.momentum = momentum
        self.pwla2d = PWLA2d(N=self.N, momentum=self.momentum)
        self.min_x, self.max_x = min_x, max_x

    def forward(self, x, conditional_params=None):
        if self.min_x and self.max_x:
            x = torch.clamp(x, min=self.min_x, max=self.max_x)

        a = self.pwla2d(x.view(-1, 1))
        a = a.reshape(x.shape)

        return a
