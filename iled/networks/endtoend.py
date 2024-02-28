from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

from typing import Any, Optional, Sequence, Callable, Union


@dataclass
class EndToEndConfig:

    n_warmup: int = 0
    data_dt: float = 1
    substeps: int = 1
    init_nTmax: int = 1
    ae_config: ... = None
    dynamics_config: ... = None

    def make(self):
        return EndToEndModel(self)


class EndToEndModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ae = config.ae_config.make()
        self.dynamics = config.dynamics_config.make()
        self.data_dt = config.data_dt
        self.substeps = config.substeps
        self.n_warmup = config.n_warmup
        assert (
            config.init_nTmax >= config.n_warmup
        ), "NTmax should be at superior to n_warmup"
        self.nTmax = config.init_nTmax
        self._config = config

    def forward(self, input_batch, NTmax=None, n_warmup=None, substeps=None):

        if NTmax is None:
            NTmax = self.get_nTmax()
        if n_warmup is None:
            n_warmup = self.get_n_warmup()
        if substeps is None:
            substeps = self.get_substeps()

        true_latents = self.ae.batch_transform(input_batch)

        reconstruction = self.ae.batch_inverse_transform(true_latents)

        if n_warmup < NTmax:
            latent_forecast, memories = self.dynamics.integrate(
                true_latents[:, :NTmax],
                dt=self.data_dt,
                substeps=substeps,
                n_warmup=n_warmup,
            )

            reconstructed_forecast = self.ae.batch_inverse_transform(latent_forecast)

            linear_part, nl_part = self.dynamics.evaluate_dynamics_parts(
                latent_forecast[:, n_warmup:].detach(), memories.detach()
            )

            dynamics_losses = self.dynamics.get_dynamics_losses()
        else:
            latent_forecast = true_latents[:,:NTmax]
            reconstructed_forecast = input_batch[:,:NTmax]
            dynamics_losses = torch.zeros(1)
            linear_part, nl_part = None, None
            memories = None

        # Returning dicts for DataParallel compatibility
        return {
            "true_latents": true_latents,
            "reconstruction": reconstruction,
            "latent_forecast": latent_forecast,
            "reconstructed_forecast": reconstructed_forecast,
            "memories": memories,
            "additional_losses": dynamics_losses,
            "dynamics_parts": (linear_part, nl_part),
        }

    def config(self):
        return self._config

    def decayable_parameters(self):
        list_param = list(self.ae.parameters()) + list(
            self.dynamics.decayable_parameters()
        )

        return iter(list_param)

    def non_decayable_parameters(self):
        return self.dynamics.non_decayable_parameters()

    def set_nTmax(self, new_nTmax):
        self.nTmax = new_nTmax

    def get_nTmax(self):

        return self.nTmax

    def get_n_warmup(self):
        return self.n_warmup

    def get_substeps(self):
        return self.substeps
