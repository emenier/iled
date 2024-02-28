import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Union, Any, Optional, Sequence, Callable


class Scaling(nn.Module):

    def __init__(self, scale, shift):
        super().__init__()

        self.scale = nn.parameter.Parameter(scale, requires_grad=False)
        self.shift = nn.parameter.Parameter(shift, requires_grad=False)

    def forward(self, x):

        return (x - self.shift) / self.scale


class InverseScaling(nn.Module):

    def __init__(self, scale, shift):
        super().__init__()
        self.scale = nn.parameter.Parameter(scale, requires_grad=False)
        self.shift = nn.parameter.Parameter(shift, requires_grad=False)

    def forward(self, x):

        return x * self.scale + self.shift


class ScalerBase(nn.Module):

    def __init__(self, scale, shift):
        super().__init__()
        self.scaling = Scaling(scale, shift)
        self.inverse_scaling = InverseScaling(scale, shift)


@dataclass
class MinMaxScalerConfig:
    data_min: Sequence[float]
    data_max: Sequence[float]

    def make(self):
        return MinMaxScaler(self)


class MinMaxScaler(ScalerBase):

    def __init__(self, config):

        self.config = config
        self.scale = config.data_max - config.data_min
        self.shift = config.data_min
        self.scale = torch.tensor(self.scale)
        self.shift = torch.tensor(self.shift)

        super().__init__(self.scale, self.shift)
