import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Callable

__all__ = ["CNN1DEncoderConfig", "CNN1DDecoderConfig", "CNN1DEncoder", "CNN1DDecoder"]


@dataclass
class CNN1DEncoderConfig:
    activation: Callable = nn.Identity()
    activation_output: Callable = nn.Identity()
    kernel_size: int = 5
    dim_input: int = None
    dim_latent: int = None
    layer_channels: Sequence[int] = ()
    bias: bool = True
    latent_centering: bool = False

    def make(self):
        return CNN1DEncoder(self)


@dataclass
class CNN1DDecoderConfig:
    activation: Callable = nn.Identity()
    activation_output: Callable = nn.Identity()
    kernel_size: int = 5
    dim_input: int = None
    dim_latent: int = None
    layer_channels: Sequence[int] = ()
    unflatten_shape: Sequence[int] = ()
    bias: bool = True

    def make(self):
        return CNN1DDecoder(self)


class LatentSpaceCentering(nn.Module):

    def __init__(self, c, momentum=0.05):
        super().__init__()
        self.means = nn.Parameter(torch.zeros(c), requires_grad=False)
        self.momentum = momentum
        # self.training = True
        self.c = c

    def forward(self, x, reverse=False):

        if reverse:
            return x + self.means
        else:
            if self.training:
                dims = [i for i in range(x.ndim - 1)]
                x_means = x.mean(dim=dims)
                self.means.data = (
                    1 - self.momentum
                ) * self.means + self.momentum * x_means

            return x - self.means.reshape(*[1 for i in range(x.ndim - 1)], self.c)


class Unpad1d(nn.Module):

    def __init__(self, left, right):
        super().__init__()
        self.left, self.right = left, right

    def forward(self, x):

        return x[..., self.left : -self.right]


class PaddingLayer1D(nn.Module):

    def __init__(self, dim_input, reverse=False):

        super().__init__()
        initial_padding = 2 ** int(np.ceil(np.log(dim_input) / np.log(2))) - dim_input

        if initial_padding > 0:
            initial_padding = initial_padding / 2
            self.left_padding, self.right_padding = int(np.floor(initial_padding)), int(
                np.ceil(initial_padding)
            )

            if reverse:
                self.layer = Unpad1d(self.left_padding, self.right_padding)
            else:
                self.layer = nn.ConstantPad1d(
                    padding=(self.left_padding, self.right_padding), value=0.0
                )
        else:
            self.layer = nn.Identity()

    def forward(self, x):

        return self.layer(x)


class Downsample(nn.Module):

    def __init__(self, input_c, output_c, kernel_size, act):
        super().__init__()
        self.filters = nn.Conv1d(
            input_c, output_c, kernel_size=kernel_size, stride=1, padding="same"
        )

        self.pooling = nn.AvgPool1d(kernel_size=(2,), stride=(2,), padding=(0,))

        self.act = act

    def forward(self, x):

        return self.act(self.pooling(self.filters(x)))


class Upsample(nn.Module):

    def __init__(self, input_c, output_c, kernel_size, act):
        super().__init__()

        self.upsampler = nn.Upsample(scale_factor=2, mode="linear")

        padding = int((kernel_size - 1) / 2)

        self.filters = nn.ConvTranspose1d(
            input_c, output_c, kernel_size=kernel_size, padding=(padding,)
        )

        self.act = act

    def forward(self, x):

        return self.act(self.filters(self.upsampler(x)))


class CNN1DEncoder(nn.Module):

    def __init__(self, config: CNN1DEncoderConfig):
        super().__init__()

        act = config.activation
        layers = []

        layers.append(PaddingLayer1D(config.dim_input))

        for i in range(len(config.layer_channels) - 1):
            layers.append(
                Downsample(
                    config.layer_channels[i],
                    config.layer_channels[i + 1],
                    kernel_size=config.kernel_size,
                    act=act,
                )
            )

        padded_size = (
            config.dim_input + layers[0].left_padding + layers[0].right_padding
        )
        final_nx = int(np.ceil(padded_size / 2 ** (len(config.layer_channels) - 1)))

        layers.append(nn.Flatten(start_dim=-2))
        layers.append(
            nn.Linear(final_nx * config.layer_channels[-1], config.dim_latent)
        )

        layers.append(config.activation_output)
        if config.latent_centering:
            layers.append(LatentSpaceCentering(config.dim_latent))

        self.layers = nn.ModuleList(layers)
        self.config = config
        # apply_custom_initialization(self)

    def __iter__(self):
        return iter(self.layers)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class CNN1DDecoder(nn.Module):

    def __init__(self, config: CNN1DDecoderConfig):
        super().__init__()

        act = config.activation
        layers = []

        lin_dim = config.unflatten_shape[0] * config.unflatten_shape[1]
        layers.append(nn.Linear(config.dim_latent, lin_dim))
        layers.append(nn.Unflatten(-1, tuple(config.unflatten_shape)))

        for i in range(len(config.layer_channels) - 1):
            if i == len(config.layer_channels) - 2:
                act = config.activation_output

            layers.append(
                Upsample(
                    config.layer_channels[i],
                    config.layer_channels[i + 1],
                    kernel_size=config.kernel_size,
                    act=act,
                )
            )

        layers.append(PaddingLayer1D(config.dim_input, reverse=True))

        self.layers = nn.ModuleList(layers)
        self.config = config
        # apply_custom_initialization(self)

    def __iter__(self):
        return iter(self.layers)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x
