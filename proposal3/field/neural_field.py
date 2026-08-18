"""Learned, sensor-agnostic continuous scene field for P3.

This replaces the toy *linear* ``SceneField`` (Gaussian bumps x cosines, solved
by one least-squares) with a **learned neural field** ``NeuralSceneField`` that
maps continuous coordinates ``(x, y, lambda)`` to radiance.  The crucial design
choice that preserves P3's differentiated claim is that the field is
*sensor-agnostic*: it learns the underlying scene, and a sensor only ever
appears as the *operator* ``O_s`` (SRF integrate + blur/decimate) applied to the
rendered cube.  Because the field never sees which sensor it is being rendered
for, a field fitted on sensors A,B can be rendered for an unseen sensor C
zero-shot -- exactly the sensor-independence thesis, but now with a model
expressive enough for real (non-low-order) scenes.

The teacher/fit protocol below lets the selfcheck demonstrate the central point
of the P3 redesign: real scenes are NOT low-order Gaussian x cosine fields, so
the linear family underfits (large hold-out-sensor error) while the neural field
recovers them and keeps delta_sensor small.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


class _SIREN(nn.Module):
    """SIREN layer stack (sine activations, first-layer frequency scaling)."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, layers: int,
                 omega0: float = 30.0):
        super().__init__()
        self.omega0 = omega0
        self.first = nn.Linear(in_dim, hidden)
        self.mid = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(max(layers - 1, 0))])
        self.last = nn.Linear(hidden, out_dim)
        # SIREN init: uniform in [-c/lin, c/lin] for hidden layers.
        with torch.no_grad():
            for lin in (self.first, *self.mid, self.last):
                nin = lin.weight.shape[1]
                c = 1.0 if lin is self.last else 6.0 / omega0
                lin.weight.uniform_(-c / nin ** 0.5, c / nin ** 0.5)
                if lin.bias is not None:
                    lin.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.sin(self.omega0 * self.first(x))
        for lin in self.mid:
            x = torch.sin(lin(x))
        return self.last(x)


class NeuralSceneField(nn.Module):
    """Sensor-agnostic radiance field F(x, y, lambda; c).

    ``c`` is a per-scene latent code (learnable), ``(x, y, lambda)`` are
    continuous coordinates in [0, 1].  Output is a single radiance value, so the
    field is evaluated on a dense grid to produce the cube (N_lam, H, W).
    """

    def __init__(self, latent_dim: int = 32, hidden: int = 128, layers: int = 4,
                 omega0: float = 30.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = _SIREN(3 + latent_dim, hidden, 1, layers, omega0)

    def render(self, lam: torch.Tensor, hw: int,
               code: torch.Tensor) -> torch.Tensor:
        """Evaluate the field on the dense cube (N_lam, H, W)."""
        dev = code.device
        g = (torch.arange(hw, dtype=torch.float32, device=dev) + 0.5) / hw
        xy = torch.stack(torch.meshgrid(g, g, indexing="ij"), dim=-1)  # (H,W,2)
        xy = xy.reshape(-1, 2)                                        # (HW,2)
        code_t = code.reshape(1, self.latent_dim).expand(xy.shape[0], -1)
        out = []
        for li in lam:
            coord = torch.cat([xy, li.expand_as(xy[:, :1]), code_t], dim=-1)
            out.append(self.net(coord).reshape(hw, hw))
        return torch.stack(out, dim=0)  # (N_lam, H, W)


def render_observation(field: NeuralSceneField, code: torch.Tensor,
                       sensor, lam: torch.Tensor, hw: int) -> torch.Tensor:
    """O_s[F] for the neural field: render the cube, then apply the sensor."""
    cube = field.render(lam, hw, code)            # (N,H,W)
    return sensor.observe_cube(cube, lam)         # (M,h,w)


def fit_neural(field: NeuralSceneField, code: torch.Tensor,
               pairs: List[Tuple[object, torch.Tensor]],
               lam: torch.Tensor, hw: int,
               steps: int = 800, lr: float = 5e-3) -> List[float]:
    """Jointly fit (field, code) to observations from several sensors.

    `pairs` is a list of (sensor, y_obs).  The sensor operator is never learned
    -- only the scene field and its latent code are.  Returns the per-step loss
    for diagnostics.
    """
    params = list(field.parameters()) + [code]
    opt = torch.optim.Adam(params, lr=lr)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        total = 0.0
        # Separate backward per sensor: each builds its own graph and is freed
        # after its own backward, so gradients accumulate without retain_graph.
        for sensor, y_obs in pairs:
            y_obs = y_obs.detach()
            y_hat = render_observation(field, code, sensor, lam, hw)
            loss = ((y_hat - y_obs) ** 2).mean()
            loss.backward()
            total = total + float(loss.item())
        opt.step()
        losses.append(total)
    return losses
