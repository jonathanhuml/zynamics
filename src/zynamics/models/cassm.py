"""CASSM-shaped dense computation-aware filtering prototype."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from pydantic import Field
from torch import Tensor, nn

from zynamics.models.base import BaseDynamicsModel, BaseModelConfig, OptimizationConfig
from zynamics.types import LossOutput, ModelOutput, observations_from_batch


@BaseModelConfig.register
class CASSMConfig(BaseModelConfig):
    """Config for the initial dense CASSM adapter."""

    name: Literal["cassm"] = "cassm"
    objective: str = "dense_filter_gaussian_nll"
    latent_dim: int = 3
    projection_dim: int = 8
    dt: float = 1.0
    init_obs_noise: float = 0.1
    jitter: float = 1e-5
    optimization: OptimizationConfig = Field(
        default_factory=lambda: OptimizationConfig(
            name="gradient",
            optimizer="AdamW",
            lr=1e-3,
            gradient_clip=1.0,
        )
    )

    def build(self, n_neurons: int, n_time: int) -> "CASSM":
        return CASSM(
            n_neurons=n_neurons,
            n_time=n_time,
            latent_dim=self.latent_dim,
            projection_dim=self.projection_dim,
            dt=self.dt,
            init_obs_noise=self.init_obs_noise,
            jitter=self.jitter,
            objective=self.objective,
        )


class CASSM(BaseDynamicsModel):
    """Dense prototype of a computation-aware state-space filter.

    This keeps the benchmark-facing API close to CASSM while staying compact.
    The upstream sparse projection/filter implementation can replace this core
    without changing configs, trainer code, or reporting.
    """

    def __init__(
        self,
        n_neurons: int,
        n_time: int,
        latent_dim: int = 3,
        projection_dim: int = 8,
        dt: float = 1.0,
        init_obs_noise: float = 0.1,
        jitter: float = 1e-5,
        objective: str = "dense_filter_gaussian_nll",
    ) -> None:
        super().__init__()
        if projection_dim > n_neurons:
            raise ValueError("projection_dim must be <= n_neurons for dense CASSM.")

        self.n_neurons = n_neurons
        self.n_time = n_time
        self.latent_dim = latent_dim
        self.projection_dim = projection_dim
        self.dt = float(dt)
        self.jitter = float(jitter)
        self.objective = objective
        self.state_dim = 2 * n_neurons

        self.raw_time_scale = nn.Parameter(torch.tensor(0.0))
        self.raw_kernel_scale = nn.Parameter(torch.tensor(0.0))
        self.raw_spatial_scale = nn.Parameter(torch.tensor(0.0))
        self.raw_obs_noise = nn.Parameter(
            torch.full((n_neurons,), _inv_softplus(init_obs_noise))
        )
        self.initial_state = nn.Parameter(torch.zeros(self.state_dim, 1))
        self.latent_locations = nn.Parameter(torch.randn(n_neurons, latent_dim) * 0.1)

        q, _ = torch.linalg.qr(torch.randn(n_neurons, projection_dim))
        self.projection = nn.Parameter(q.T.contiguous())

        observation_matrix = torch.zeros(n_neurons, self.state_dim)
        observation_matrix[torch.arange(n_neurons), 2 * torch.arange(n_neurons)] = 1.0
        self.register_buffer("observation_matrix", observation_matrix)

    def forward(self, x: Tensor) -> ModelOutput:
        loss, reconstruction, variance = self._filter(x.float())
        return ModelOutput(
            rates=reconstruction.clamp_min(0.0),
            reconstruction=reconstruction,
            extras={"loss": loss, "variance": variance},
        )

    def loss(
        self,
        batch: Tensor | dict[str, Tensor],
        output: ModelOutput,
        epoch: int = 0,
    ) -> LossOutput:
        total = output.extras["loss"]
        return LossOutput(
            total=total,
            named_terms={"nll": total},
            objective=self.objective,
        )

    def _filter(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if x.ndim != 3:
            raise ValueError("CASSM expects input shape (batch, time, neurons).")
        batch_size, n_time, n_neurons = x.shape
        if n_neurons != self.n_neurons:
            raise ValueError(f"Expected {self.n_neurons} neurons, got {n_neurons}.")

        transition, stationary_cov, process_cov = self._dynamics()
        obs_noise = F.softplus(self.raw_obs_noise) + self.jitter
        h = self.observation_matrix
        pmat = self.projection
        h_proj = pmat @ h
        r_proj = pmat @ torch.diag(obs_noise) @ pmat.T
        eye_proj = torch.eye(self.projection_dim, device=x.device, dtype=x.dtype)

        mean = self.initial_state.expand(batch_size, -1, -1)
        cov = stationary_cov

        losses = []
        means = []
        variances = []
        for t in range(n_time):
            pred_obs = (h @ mean).squeeze(-1)
            pred_var = torch.diagonal(h @ cov @ h.T, dim1=-2, dim2=-1) + obs_noise
            residual = x[:, t] - pred_obs
            losses.append(
                0.5
                * (
                    torch.log(2 * torch.tensor(math.pi, device=x.device) * pred_var)
                    + residual.pow(2) / pred_var
                ).mean()
            )

            projected_residual = pmat @ residual.unsqueeze(-1)
            innovation = h_proj @ cov @ h_proj.T + r_proj
            innovation = innovation + self.jitter * eye_proj
            chol = torch.linalg.cholesky(innovation)
            pht = cov @ h_proj.T
            gain = torch.cholesky_solve(pht.T, chol).T

            mean = mean + gain.unsqueeze(0) @ projected_residual
            cov = cov - gain @ h_proj @ cov
            cov = 0.5 * (cov + cov.T) + self.jitter * torch.eye(
                self.state_dim, device=x.device, dtype=x.dtype
            )

            means.append((h @ mean).squeeze(-1))
            variances.append(torch.diagonal(h @ cov @ h.T, dim1=-2, dim2=-1) + obs_noise)

            mean = transition @ mean
            cov = transition @ cov @ transition.T + process_cov
            cov = 0.5 * (cov + cov.T)

        return (
            torch.stack(losses).mean(),
            torch.stack(means, dim=1),
            torch.stack(variances, dim=1),
        )

    def _dynamics(self) -> tuple[Tensor, Tensor, Tensor]:
        dtype = self.initial_state.dtype
        device = self.initial_state.device
        ell = F.softplus(self.raw_time_scale) + 1e-4
        sigma2 = F.softplus(self.raw_kernel_scale) + 1e-4
        spatial_ell = F.softplus(self.raw_spatial_scale) + 1e-4

        lam = torch.sqrt(torch.tensor(3.0, dtype=dtype, device=device)) / ell
        delta_t = torch.tensor(self.dt, dtype=dtype, device=device)
        fmat = torch.stack(
            [
                torch.stack([torch.zeros_like(lam), torch.ones_like(lam)]),
                torch.stack([-lam.pow(2), -2.0 * lam]),
            ]
        ).squeeze(-1)
        transition_time = torch.matrix_exp(fmat * delta_t)

        stationary_time = torch.stack(
            [
                torch.stack([sigma2, torch.zeros_like(sigma2)]),
                torch.stack([torch.zeros_like(sigma2), lam.pow(2) * sigma2]),
            ]
        ).squeeze(-1)

        distances = torch.cdist(self.latent_locations, self.latent_locations).pow(2)
        spatial_cov = torch.exp(-0.5 * distances / spatial_ell.pow(2))
        spatial_cov = spatial_cov + self.jitter * torch.eye(
            self.n_neurons, device=device, dtype=dtype
        )

        transition = torch.kron(
            torch.eye(self.n_neurons, device=device, dtype=dtype),
            transition_time,
        )
        stationary_cov = torch.kron(spatial_cov, stationary_time)
        process_cov = stationary_cov - transition @ stationary_cov @ transition.T
        process_cov = 0.5 * (process_cov + process_cov.T)
        process_cov = process_cov + self.jitter * torch.eye(
            self.state_dim, device=device, dtype=dtype
        )
        return transition, stationary_cov, process_cov


def _inv_softplus(value: float) -> float:
    return math.log(math.exp(value) - 1.0)
