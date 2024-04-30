import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5, 
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.weight_sigma = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.register_buffer('weight_epsilon', torch.empty((out_features, in_features), **factory_kwargs))

        self.bias_mu = Parameter(torch.empty(out_features, **factory_kwargs))
        self.bias_sigma = Parameter(torch.empty(out_features, **factory_kwargs))
        self.register_buffer('bias_epsilon', torch.empty(out_features, **factory_kwargs))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init * mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init * mu_range)

    def reset_noise(self) -> None:
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def scale_noise(self, size) -> None:
        epsilon = torch.randn(size)
        epsilon = epsilon.sign().mul(epsilon.abs().sqrt())
        return epsilon


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)
    
    