"""Sensor operators O_s applied to a shared continuous scene field.

A sensor is defined by its spectral response (Gaussian bumps over lambda) and
its spatial sampling (blur + decimate).  Each sensor is a *linear operator*
on the field's coefficients, so fitting a field to several sensors is one
least-squares solve, and rendering it for a hold-out sensor is zero-shot.
"""

from __future__ import annotations

from typing import Optional

import torch

from proposal5.spectralflow.nullspace import DegradationOperator

from .field import SceneField


def gaussian_srf(centers: torch.Tensor, lam: torch.Tensor,
                 width: float) -> torch.Tensor:
    """(N_lam, M) response; each band normalised to sum 1 over lambda."""
    d = lam[:, None] - centers[None, :]
    r = torch.exp(-0.5 * (d / width) ** 2)
    return r / r.sum(0, keepdim=True).clamp_min(1e-12)


class Sensor:
    def __init__(self, name: str, scale: int, msi_bands: int,
                 srf_width: float, srf_lo: float = 0.12, srf_hi: float = 0.88,
                 psf_sigma: float = 1.2):
        self.name = name
        self.scale = scale
        self.msi_bands = msi_bands
        self.srf_width = srf_width
        self.srf_lo, self.srf_hi = srf_lo, srf_hi
        self.D = DegradationOperator(scale, sigma=psf_sigma)

    def centers(self) -> torch.Tensor:
        return torch.linspace(self.srf_lo, self.srf_hi, self.msi_bands)

    def observe(self, field: SceneField, lam: torch.Tensor,
                hw: int) -> torch.Tensor:
        """Y_s = O_s[F]: spectral integrate, then blur + decimate. (M, h, w)."""
        spec = gaussian_srf(self.centers(), lam, self.srf_width)  # (N,M)
        f = field.render(lam, hw)                                  # (N,H,W)
        y = torch.einsum("nm,nhw->mhw", spec, f)                   # (M,H,W)
        return self.D(y[None])[0]

    def linear_map(self, field: SceneField, lam: torch.Tensor,
                   hw: int) -> torch.Tensor:
        """Dense matrix A_s: (M*h*w, nz) with observe = A_s @ z."""
        nz = field.bands * field.modes ** 2
        cols = []
        z0 = field.Z.detach().clone()
        with torch.no_grad():
            for i in range(nz):
                z = torch.zeros(nz)
                z[i] = 1.0
                field.Z.copy_(z.reshape(field.bands, field.modes, field.modes))
                cols.append(self.observe(field, lam, hw).reshape(-1))
            field.Z.copy_(z0)
        return torch.stack(cols, dim=1)


def fit_field(fields_obs, fields_A):
    """Least-squares fit of a field to any number of sensors.

    fields_obs: list of observations [y_1, y_2, ...]; fields_A: their maps.
    Returns the flattened coefficient vector and the flat observations.
    """
    A = torch.cat([a for a in fields_A], dim=0)
    y = torch.cat([o.reshape(-1) for o in fields_obs], dim=0)
    return torch.linalg.lstsq(A, y).solution, y