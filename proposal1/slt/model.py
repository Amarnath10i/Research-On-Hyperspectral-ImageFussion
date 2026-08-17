"""SLTNet: geodesic-transport fusion network.

Two output parameterisations, selected by ``cfg.manifold``:

* Euclidean residual (stage 1 baseline):  out = Y_0 + r(X, M).  The network can
  freely change intensity and direction, so intensity can mask spectral error.
* Manifold transport (the contribution):  out = I_0 * Exp_{dir0}(v) where
  I_0 = ||Y_0|| and dir0 = Y_0 / I_0 come from the observation and v is a
  learned tangent vector.  The encoder is fed *scale-invariant* inputs (the
  direction, and optionally the normalised MSI guide), so the transport is
  illumination-equivariant by construction: scaling the observation scales the
  output intensity but leaves the spectrum (SAM) unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .manifold import exp_map, l2_normalize, tangent_projection


class _ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, 1, 1)
        self.norm = nn.GroupNorm(min(8, cout), cout)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm(self.conv1(x)))
        return self.conv2(h) + x if x.shape[1] == h.shape[1] else self.conv2(h)


class Encoder(nn.Module):
    def __init__(self, in_ch: int, width: int, depth: int = 3):
        super().__init__()
        blocks, c = [], in_ch
        for _ in range(depth):
            blocks.append(_ConvBlock(c, width))
            c = width
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SLTNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        guide_ch = cfg.msi_bands if cfg.use_msi_guide else 0
        self.encoder = Encoder(cfg.bands + guide_ch, cfg.width, cfg.depth)
        head_ch = cfg.bands if cfg.fix_intensity else cfg.bands + 1
        self.head = nn.Conv2d(cfg.width, head_ch, 1)
        self.scale = cfg.scale

    def forward(self, lr: torch.Tensor, msi: torch.Tensor = None):
        y0 = F.interpolate(lr, scale_factor=self.scale, mode="bicubic",
                           align_corners=False).clamp(0, 1)
        if self.cfg.manifold:
            i0 = y0.norm(2, 1, keepdim=True).clamp_min(1e-8)
            dir0 = y0 / i0
            enc_in = [dir0]
            if self.cfg.use_msi_guide:
                enc_in.append(l2_normalize(msi))
        else:
            i0, dir0 = None, None
            enc_in = [y0]
            if self.cfg.use_msi_guide:
                enc_in.append(msi)
        feats = self.encoder(torch.cat(enc_in, 1))
        raw = self.head(feats)

        if self.cfg.manifold:
            v = raw[:, : self.cfg.bands] if not self.cfg.fix_intensity else raw
            v = tangent_projection(v, dir0)
            n = v.norm(2, 1, keepdim=True)
            v = v / n.clamp_min(1e-12) * n.clamp_max(self.cfg.max_angle_rad)
            if self.cfg.geodesic:
                dir_out = exp_map(dir0, v)
            else:
                dir_out = l2_normalize(dir0 + v)      # naive chordal transport
            if self.cfg.fix_intensity:
                out = i0 * dir_out
            else:
                log_int = raw[:, self.cfg.bands: self.cfg.bands + 1].clamp(-0.7, 0.7)
                out = i0 * dir_out * torch.exp(log_int)
            out = out.clamp(0, 1)
            return {"out": out, "v": v, "dir0": dir0, "I0": i0, "y0": y0}
        else:
            out = (y0 + raw).clamp(0, 1)
            return {"out": out, "v": None, "dir0": None, "I0": None, "y0": y0}

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())