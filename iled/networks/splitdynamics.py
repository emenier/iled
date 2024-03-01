from dataclasses import dataclass
from ..backprop import ACA_bptt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torch.nn.utils.parametrizations import orthogonal
import torch.nn.functional as F

from typing import Any, Optional, Sequence, Callable


@dataclass
class SplitDynamicsConfig:
    """Config class for the (continuous) split dynamics."""

    dim_latent: int
    dim_hidden: int

    activation: Callable

    linear_operator: str = "unconstrained"
    nl_operator: str = "unconstrained"

    nl_width: int = 32
    nl_n_hidden_layers: int = 3

    default_substeps: int = 1
    zero_init: bool = False

    def make(self):

        return SplitDynamics(self)


################################
# Operator Root
################################


class OperatorRoot(nn.Module):

    def get_operator_losses(self, *args, **kwargs):

        return None

    def decayable_parameters(self):

        return list(self.parameters())

    def non_decayable_parameters(self):

        return []

    def zero_initialize(self):
        raise NotImplementedError

################################
# WEIGHT PARAMETERIZATIONS
################################


class ImaginaryParametrization(nn.Module):

    def forward(self, W):

        return W - W.t()


class DiagonalParametrization(nn.Module):
    def forward(self, X):

        K = -torch.diag_embed(torch.abs(torch.diagonal(X)))

        return K


class DissipativeParametrization(nn.Module):

    def forward(self, W):

        return W - W.t() - torch.diag_embed(torch.abs(torch.diag(W)))


class BoundedParam(nn.Module):

    def forward(self, X):

        return torch.clamp(X, min=0, max=1)


################################
# LIFTERS (to memory space)
################################


class IdentityLifter(OperatorRoot):

    def __init__(self, dim, dim_hidden, act, n_layers=4):

        super().__init__()

        assert (
            dim_hidden > dim
        ), "[splitdynamics] Hidden dim lower than input dim, \
        can't use Identity lifter"

        inc = int((dim_hidden - 2 * dim) / n_layers)
        layers = []
        d = dim
        for l in range(n_layers - 1):
            layers.extend([nn.Linear(d, d + inc), act])
            d += inc
        layers.append(nn.Linear(d, dim_hidden - dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):

        ret = torch.cat([x, self.layers(x)], dim=-1)
        return ret

    def zero_initialize(self):

        self.layers[-1].weight.data.fill_(0.)
        self.layers[-1].bias.data.fill_(0.)

################################
# NON LINEAR PARTS
################################

class NLOperatorRoot(OperatorRoot):

    def zero_initialize(self):

        self.layers[-1].weight.data.fill_(0.)
        self.layers[-1].bias.data.fill_(0.)



class NonLinearMLP(NLOperatorRoot):
    """Considers the full state (z and memory)"""

    def __init__(self, dim_latent, dim_hidden, act, n_hidden_layers=4, width=32):
        super().__init__()

        layers = []
        in_dim = dim_latent + dim_hidden
        layers.append(nn.Linear(in_dim, width))

        layers.append(act)
        for n in range(n_hidden_layers):
            layers.extend([nn.Linear(width, width), act])
        layers.append(nn.Linear(width, dim_latent))

        self.layers = nn.Sequential(*layers)

        self.layers[-1].weight.data.fill_(0.0)
        self.layers[-1].bias.data.fill_(0.0)

    def forward(self, z, h):

        return self.layers(torch.cat([z, h], dim=-1))

class MemoryOnlyMLP(NLOperatorRoot):

    def __init__(self, dim_latent, dim_hidden, act, n_hidden_layers=4, width=32):
        super().__init__()
        layers = []
        in_dim = dim_hidden
        layers.append(nn.Linear(in_dim, width))

        layers.append(act)
        for n in range(n_hidden_layers):
            layers.extend([nn.Linear(width, width), act])
        layers.append(nn.Linear(width, dim_latent))

        self.layers = nn.Sequential(*layers)

        self.layers[-1].weight.data.fill_(0.0)
        self.layers[-1].bias.data.fill_(0.0)

    def forward(self, z, h):

        return self.layers(h)


nl_operator_class = {"unconstrained": NonLinearMLP, "memory_only": MemoryOnlyMLP}
################################
# LINEAR OPERATORS
################################


class LinearRoot(OperatorRoot):

    def non_decayable_parameters(self):
        return self.parameters()

    def decayable_parameters(self):
        return []


class UnconstrainedLinearOperator(LinearRoot):

    def __init__(self, dim, *args):
        super().__init__()
        self.operator = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.operator(x)

    def weight(self, *args):
        return self.operator.weight

    def zero_initialize(self):

        self.operator.weight.data.fill_(0.)


class ConservativeLinearOperator(LinearRoot):

    def __init__(self, dim, *args):
        super().__init__()
        self.operator = nn.Linear(dim, dim, bias=False)

        parametrize.register_parametrization(
            self.operator, "weight", ImaginaryParametrization()
        )

    def forward(self, x):
        return self.operator(x)

    def weight(self, *args):

        return self.operator.weight

    def zero_initialize(self):

        self.operator.parametrizations.weight.original.data.fill_(0.)

class DissipativeLinearOperator(LinearRoot):

    def __init__(self, dim, *args):
        super().__init__()
        self.operator = nn.Linear(dim, dim, bias=False)

        parametrize.register_parametrization(
            self.operator, "weight", DissipativeParametrization()
        )

    def forward(self, x):
        return self.operator(x)

    def weight(self, *args):

        return self.operator.weight

    def zero_initialize(self):

        self.operator.parametrizations.weight.original.data.fill_(0.)


class MaskedSubspaceLinearOperator(LinearRoot):

    def __init__(self, dim, *args, L1_loss_scale=1e-3, **kwargs):
        super().__init__()
        self.operator = nn.Linear(dim, dim, bias=False)

        self.mask = nn.Parameter(torch.ones(dim), requires_grad=True)
        parametrize.register_parametrization(self, "mask", BoundedParam())
        self.L1_loss_scale = L1_loss_scale

        self.projector = orthogonal(nn.Linear(dim, dim, bias=False))

    def forward(self, x):

        prod = self.weight().matmul(x.t())
        return prod.t()

    @property
    def subspace_operator(self):

        matrix_mask = torch.diag_embed(self.mask)
        return matrix_mask.matmul(self.operator.weight.matmul(matrix_mask))

    def weight(self, *args):

        return self.projector.weight.t().matmul(
            self.subspace_operator.matmul(self.projector.weight)
        )

    def get_operator_losses(self, *args, **kwargs):

        return self.L1_loss_scale * self.get_mask_L1() / len(self.mask)

    def get_mask_L1(self):
        return torch.abs(self.mask).sum()

    def decayable_parameters(self):
        return []  # list(self.operator.parameters())+list(self.projector.parameters())

    def non_decayable_parameters(self):
        return list(self.parameters())


linear_operator_classes = {
    "unconstrained": UnconstrainedLinearOperator,
    "conservative": ConservativeLinearOperator,
    "dissipative": DissipativeLinearOperator,
    "masked_subspace": MaskedSubspaceLinearOperator,
}

################################
# MEMORY OPERATORS
################################


def get_lambdas(dim_hidden, min_t, max_t):

    lambdas = 1 / (np.exp(np.linspace(np.log(min_t), np.log(max_t), dim_hidden)))

    return torch.tensor(lambdas).to(torch.get_default_dtype())


class DiagonalMemoryOperator(OperatorRoot):

    def __init__(self, dim_memory, min_t=1, max_t=10):
        super().__init__()

        self.operator = nn.Linear(dim_memory, dim_memory, bias=False)
        lambdas_init = get_lambdas(dim_memory, min_t, max_t)
        self.operator.weight.data = torch.diag_embed(lambdas_init)
        parametrize.register_parametrization(
            self.operator, "weight", DiagonalParametrization()
        )

    def forward(self, x):
        return self.operator(x)

    @property
    def lambdas(self):
        return torch.diag(self.operator.weight)

    @property
    def weight(self):
        return self.operator.weight

    def non_decayable_parameters(self):
        return self.parameters()

    def decayable_parameters(self):
        return []


################################
# Propagators
################################


class SplitDynamics(nn.Module):

    def __init__(self, config: SplitDynamicsConfig):
        super().__init__()

        act = config.activation

        self.z_dim, self.h_dim = config.dim_latent, config.dim_hidden

        linear_class = linear_operator_classes[config.linear_operator]
        self.linear_operator = linear_class(config.dim_latent)

        nl_class = nl_operator_class[config.nl_operator]
        self.nonlinear_operator = nl_class(
            config.dim_latent,
            config.dim_hidden,
            act,
            n_hidden_layers=config.nl_n_hidden_layers,
            width=config.nl_width,
        )

        self.memory_operator = DiagonalMemoryOperator(config.dim_hidden)

        self.lifter = IdentityLifter(config.dim_latent, config.dim_hidden, act)

        self._config = config
        self.default_substeps = config.default_substeps
        self.operators = [
            self.memory_operator,
            self.lifter,
            self.linear_operator,
            self.nonlinear_operator,
        ]

        if config.zero_init:
            self.zero_initialize()

    @property
    def config(self):
        return self._config

    def forward(self, z, h=None):

        return self.siRK3_step(
            None, z, h, substeps=self.default_substeps, dt=None
        )  # TODO : To fix, dt should not default to 1 (or None)

    def step(self, t, fullstate, dt):

        z, h = torch.split(fullstate, [self.z_dim, self.h_dim], dim=-1)

        return torch.cat([*self.siRK3_substep(t, z, h, dt=dt)], dim=-1)

    def siRK3_step(self, t, z, h, dt=1, substeps=1, **substep_kwargs):

        sub_dt = dt / substeps
        for i in range(substeps):

            z, h = self.siRK3_substep(t, z, h, dt=sub_dt, **substep_kwargs)
            if t is not None:
                t += sub_dt

        return z, h

    def siRK3_substep(self, t, z, h, dt=1):

        z, h = self.format_inputs(t, z, h)

        gamma = 0.5
        x0 = torch.cat([z, h], dim=-1)
        z_save, h_save = z.clone(), h.clone()
        x_shape = x0.shape
        x_save = x0.clone()
        x = x0

        for n in range(3):
            ddt = dt / (3 - n)
            z, h = torch.split(x, [self.z_dim, self.h_dim], dim=-1)

            z, h = self.format_inputs(t + ddt, z, h)

            numerator = (
                x_save
                + gamma
                * ddt
                * torch.cat(
                    [self.linear_operator(z_save), self.memory_operator(h_save)], dim=-1
                )
                + ddt
                * torch.cat([self.nonlinear_operator(z, h), self.lifter(z)], dim=-1)
            )

            numerator = numerator.reshape(*x_shape, 1)

            denom = self.get_inverted_operators(gamma, ddt)

            x = torch.matmul(denom, numerator).reshape(*x.shape)

        return torch.split(x, [self.z_dim, self.h_dim], dim=-1)

    def get_inverted_operators(self, gamma, dt):
        lin_weight = self.linear_operator.weight()
        mem_weight = self.memory_operator.weight
        dev, dtyp = lin_weight.device, lin_weight.dtype

        A_inv = torch.inverse(
            torch.eye(self.z_dim).to(dev, dtyp) - gamma * dt * lin_weight
        )
        Lambda_inv = torch.inverse(
            torch.eye(self.h_dim).to(dev, dtyp) - gamma * dt * mem_weight
        )
        return torch.block_diag(A_inv, Lambda_inv)

    def format_inputs(self, t, z, h):

        if h is None:
            with torch.no_grad():
                h = torch.zeros(z.shape[0], self.memory_operator.lambdas.shape[0]).to(
                    z.device, z.dtype
                )

        return z, h

    def rhs(self, t, fullstate):
        z, h = torch.split(fullstate, [self.z_dim, self.h_dim], dim=-1)
        z, h = self.format_inputs(t, z, h)
        dz_dt = self.linear_operator(z) + self.nonlinear_operator(z, h)
        dh_dt = self.lifter(z) + self.memory_operator(h)

        return torch.cat([dz_dt, dh_dt], dim=-1)

    def decayable_parameters(self):
        params_list = []

        for op in self.operators:
            params_list += list(op.decayable_parameters())
        return iter(params_list)

    def non_decayable_parameters(self):

        params_list = []

        for op in self.operators:
            params_list += list(op.non_decayable_parameters())
        return iter(params_list)

    def integrate(self, z, dt, substeps, n_warmup):

        options = ACA_bptt.get_integration_options(
            n0=n_warmup, n1=z.shape[1] - 1, dt=dt, substeps=substeps
        )

        if n_warmup > 0:
            h = self.get_initial_memory(z[:, : n_warmup + 1], dt, substeps=substeps)
        else:
            h = torch.randn(z.shape[0], self.h_dim).to(z.device, z.dtype)

        ics = torch.cat([z[:, n_warmup], h], dim=-1)

        results = ACA_bptt.odesolve_adjoint(ics, self, options)
        results = results.permute(1, 0, 2)
        memories = results[:, :, self.z_dim :]
        results = torch.cat([z[:, :n_warmup], results[..., : self.z_dim]], dim=1)
        return results, memories

    def get_initial_memory(self, z, dt, substeps=1):

        assert z.ndim == 3, "Memory init expects batched trajectories"

        z, h = self.format_inputs(None, z, None)

        nT = z.shape[1]
        interp_nT = substeps * (nT - 1) + 1

        z = nn.functional.interpolate(
            z.permute(0, 2, 1), size=interp_nT, mode="linear", align_corners=True
        )
        z = z.permute(0, 2, 1)

        times = (dt / substeps) * torch.arange(interp_nT).to(z.device, z.dtype)

        filtr = self.memory_operator.lambdas.reshape(-1, 1) * times
        filtr = torch.exp(filtr)
        filtr = torch.flip(filtr, dims=[-1]).t()

        (n_batch, n_times, z_dim) = z.shape

        lifted_trajectories = self.lifter(z.reshape(-1, z_dim))
        lifted_trajectories = lifted_trajectories.reshape(n_batch, n_times, self.h_dim)

        convoluted = lifted_trajectories * filtr
        trapezes = (
            (times[1:] - times[:-1]).reshape(1, -1, 1)
            * (convoluted[:, 1:] + convoluted[:, :-1])
            / 2
        )
        integration = trapezes.sum(1)

        return integration

    def evaluate_dynamics_parts(self, z, h):
        """Expects 1D latents"""

        initial_shape = z.shape
        z = z.flatten(end_dim=-2)
        h = h.flatten(end_dim=-2)

        z, h = self.format_inputs(None, z, h)

        linear_dynamics = self.linear_operator(z).reshape(*initial_shape)
        nl_dynamics = self.nonlinear_operator(z, h).reshape(*initial_shape)

        return linear_dynamics, nl_dynamics

    def get_dynamics_losses(self):

        return [op.get_operator_losses() for op in self.operators]

    def zero_initialize(self):

        self.linear_operator.zero_initialize()
        self.nonlinear_operator.zero_initialize()
        self.lifter.zero_initialize()