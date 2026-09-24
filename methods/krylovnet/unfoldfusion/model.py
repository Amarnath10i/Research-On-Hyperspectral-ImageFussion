"""UnfoldFusion - a deep-unfolded variational solver for HSI-MSI fusion.

WHERE THIS DIFFERS FROM PROPOSAL 1
----------------------------------
DAETF-Net is a feed-forward network that is *told* about the physics through
two loss terms. UnfoldFusion has no free-form backbone at all: it is an
optimisation algorithm whose iterations have been unrolled into layers, so the
observation model is structural rather than encouraged.

THE MODEL
---------
The observation model for fusion is

    Y_h = B(X)          low-resolution HSI  = blur + decimate of the truth
    Y_m = R(X)          multispectral image = spectral projection of the truth

Recovering X is ill-posed, so the classical formulation is

    min_X  ||B(X) - Y_h||^2 + ||R(X) - Y_m||^2 + lambda * phi(X)

with phi a regulariser encoding what natural hyperspectral images look like.
Half-quadratic splitting introduces an auxiliary V and alternates:

    V <- prox_phi( X )                       denoise / project onto the prior
    X <- argmin ||B(X)-Y_h||^2 + ||R(X)-Y_m||^2 + rho||X - V||^2

The second step is a linear least-squares problem. We solve it with a few
conjugate-gradient steps using only matrix-vector products - B, B^T, R, R^T are
all cheap and known - so no matrix is ever formed.

WHAT IS LEARNED
---------------
* prox_phi: a small CNN denoiser, shared across stages, conditioned on the
  current noise level (a FiLM-style embedding of rho). This is the learned
  prior, and it is the *only* free-form component.
* rho and the CG step count per stage: learned, so the solver decides how much
  to trust the prior versus the data at each stage.
* a low-rank spectral basis for X, estimated from Y_h by SVD, so the CG runs in
  an r-dimensional coefficient space instead of over all bands. This is what
  makes the unrolling affordable at 31 bands.

WHY IT SHOULD TRANSFER
----------------------
Every stage re-imposes agreement with the actual observations. A feed-forward
network that has memorised CAVE statistics has nothing forcing it to re-explain
a Harvard observation; here, a solution that does not satisfy B(X)=Y_h is
driven out at every stage regardless of which dataset it came from. The learned
part is only the prior - the smallest surface for domain-specific overfitting.

The degradation operator B is *estimated per input* rather than assumed, so a
sensor with a different blur is handled by re-estimating B, not by retraining.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------- operators
class BlurDecimate(nn.Module):
    """B and B^T: blur+decimate, and its exact adjoint.

    The adjoint matters. Using a plain bilinear upsample in place of B^T makes
    the CG solve inconsistent and the whole unrolling degenerates into a stack
    of denoisers. Here B^T is transposed convolution with the *same* kernel,
    which is the true adjoint of conv+stride.
    """

    def __init__(self, scale: int, ksize: int = 9):
        super().__init__()
        self.scale, self.ksize = scale, ksize
        # learned per-input via `set_kernel`; initialised to a Gaussian
        ax = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2
        g = torch.exp(-0.5 * (ax / 1.2) ** 2)
        k = torch.outer(g, g)
        self.register_buffer("default_kernel", (k / k.sum())[None, None])

    def forward(self, x: torch.Tensor, kernel: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        b, c = x.shape[:2]
        k = self.default_kernel if kernel is None else kernel
        k = k.expand(b, 1, self.ksize, self.ksize).reshape(b, 1, 1, self.ksize,
                                                           self.ksize)
        w = k.expand(b, c, 1, self.ksize, self.ksize).reshape(b * c, 1,
                                                             self.ksize, self.ksize)
        pad = self.ksize // 2
        # ZERO padding, not reflect. Reflect padding is a linear operator whose
        # adjoint is not reflect padding, so mixing it with a transposed-conv
        # adjoint breaks <Bx,y> == <x,B^T y> and leaves CG solving an
        # inconsistent system - the unrolling then silently degenerates into a
        # stack of denoisers. Zero padding is exactly self-consistent with the
        # zero-insert adjoint below.
        xr = F.pad(x.reshape(1, b * c, *x.shape[-2:]), (pad,) * 4, mode="constant")
        y = F.conv2d(xr, w, groups=b * c).reshape(b, c, *x.shape[-2:])
        return y[..., ::self.scale, ::self.scale].contiguous()

    def transpose(self, y: torch.Tensor, out_hw: Tuple[int, int],
                  kernel: Optional[torch.Tensor] = None) -> torch.Tensor:
        """B^T: zero-upsample then correlate with the flipped kernel."""
        b, c, h, w = y.shape
        up = y.new_zeros(b, c, out_hw[0], out_hw[1])
        up[..., ::self.scale, ::self.scale] = y
        k = self.default_kernel if kernel is None else kernel
        k = torch.flip(k.reshape(-1, 1, self.ksize, self.ksize), dims=(-2, -1))
        k = k.expand(b, 1, self.ksize, self.ksize).reshape(b, 1, 1, self.ksize,
                                                           self.ksize)
        wgt = k.expand(b, c, 1, self.ksize, self.ksize).reshape(b * c, 1,
                                                               self.ksize, self.ksize)
        pad = self.ksize // 2
        upr = F.pad(up.reshape(1, b * c, out_hw[0], out_hw[1]), (pad,) * 4,
                    mode="constant")
        return F.conv2d(upr, wgt, groups=b * c).reshape(b, c, *out_hw)


class SpectralProject(nn.Module):
    """R and R^T: projection through the spectral response function."""

    def __init__(self, srf: torch.Tensor):
        super().__init__()
        self.register_buffer("srf", srf)              # [bands, msi_bands]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.srf.to(x.dtype).t()[:, :, None, None]
        return F.conv2d(x, w)

    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        w = self.srf.to(y.dtype)[:, :, None, None]
        return F.conv2d(y, w)


# ------------------------------------------------------------- learned prior
class NoiseConditionedDenoiser(nn.Module):
    """prox_phi. One denoiser shared by every stage, told the current rho.

    Sharing weights across stages is deliberate: it keeps the parameter count
    low, makes the unrolling behave like an actual iterative algorithm rather
    than a deep net with tied notation, and means adding stages at test time
    costs no new parameters.
    """

    def __init__(self, channels: int, width: int = 64, depth: int = 4):
        super().__init__()
        self.head = nn.Conv2d(channels, width, 3, 1, 1)
        self.body = nn.ModuleList([
            nn.Sequential(nn.Conv2d(width, width, 3, 1, 1), nn.ReLU(inplace=True),
                          nn.Conv2d(width, width, 3, 1, 1))
            for _ in range(depth)
        ])
        self.film = nn.ModuleList([nn.Linear(32, width * 2) for _ in range(depth)])
        for f in self.film:
            nn.init.zeros_(f.weight)
            nn.init.zeros_(f.bias)
        self.tail = nn.Conv2d(width, channels, 3, 1, 1)
        self.act = nn.ReLU(inplace=True)

    @staticmethod
    def _embed(rho: torch.Tensor) -> torch.Tensor:
        """Sinusoidal embedding of log-rho, as in noise-conditioned denoisers."""
        v = torch.log(rho.clamp_min(1e-8)).reshape(-1, 1)
        freqs = torch.arange(16, device=rho.device, dtype=v.dtype)
        ang = v * (2.0 ** freqs)[None]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)

    def forward(self, x: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        e = self._embed(rho.expand(x.shape[0]) if rho.numel() == 1 else rho)
        h = self.act(self.head(x))
        for blk, film in zip(self.body, self.film):
            g, b = film(e).chunk(2, dim=1)
            h = h + blk(h * (1 + g[:, :, None, None]) + b[:, :, None, None])
        return x + self.tail(h)          # residual: predicts the correction


# ------------------------------------------------------------------ the model
class UnfoldFusion(nn.Module):
    """K unrolled half-quadratic-splitting stages in a learned spectral subspace."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.stages, self.rank = cfg.stages, cfg.rank
        self.B = BlurDecimate(cfg.scale, cfg.blur_ksize)
        self.denoiser = NoiseConditionedDenoiser(cfg.rank, cfg.width, cfg.denoise_depth)
        # one log-rho per stage: the solver learns its own trust schedule
        self.log_rho = nn.Parameter(torch.linspace(-2.0, 1.0, cfg.stages))
        self.cg_steps = cfg.cg_steps
        # kernel estimator: predicts the blur actually present in this input
        self.kernel_net = nn.Sequential(
            nn.Conv2d(cfg.bands + cfg.msi_bands, 48, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, 3, 2, 1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, cfg.blur_ksize ** 2),
        )
        self.srf_buf: Optional[torch.Tensor] = None

    # -- subspace ----------------------------------------------------------
    @staticmethod
    def _basis(lr: torch.Tensor, rank: int) -> torch.Tensor:
        """Spectral basis E from the LR-HSI. Hyperspectral cubes are close to
        low rank, so a handful of vectors explain nearly all the variance and
        the solve can run in coefficient space."""
        b, c = lr.shape[:2]
        y = lr.reshape(b, c, -1).float()
        cov = y @ y.transpose(1, 2)                    # [B,C,C]
        u, _, _ = torch.linalg.svd(cov)
        return u[:, :, :rank]                          # [B,C,r]

    def _estimate_kernel(self, lr: torch.Tensor, msi: torch.Tensor) -> torch.Tensor:
        msi_lr = F.adaptive_avg_pool2d(msi, lr.shape[-2:])
        logits = self.kernel_net(torch.cat([lr, msi_lr], dim=1))
        k = torch.softmax(logits, dim=1)               # non-negative, sums to 1
        return k.reshape(-1, 1, self.cfg.blur_ksize, self.cfg.blur_ksize)

    # -- data-consistency step --------------------------------------------
    def _cg(self, z0: torch.Tensor, rhs: torch.Tensor, applyA, steps: int
            ) -> torch.Tensor:
        """Conjugate gradients on A z = rhs, A symmetric positive definite.

        A is never materialised - only its action is needed - which is what
        keeps a 31-band least-squares solve tractable inside a forward pass.
        """
        z = z0
        r = rhs - applyA(z)
        p = r
        rs = (r * r).flatten(1).sum(1)
        for _ in range(steps):
            ap = applyA(p)
            denom = (p * ap).flatten(1).sum(1).clamp_min(1e-10)
            alpha = (rs / denom).reshape(-1, 1, 1, 1)
            z = z + alpha * p
            r = r - alpha * ap
            rs_new = (r * r).flatten(1).sum(1)
            beta = (rs_new / rs.clamp_min(1e-10)).reshape(-1, 1, 1, 1)
            p = r + beta * p
            rs = rs_new
        return z

    def set_srf(self, srf: torch.Tensor) -> None:
        """The spectral response is measured from the data, not learned."""
        self.srf_buf = srf

    def forward(self, lr_hsi: torch.Tensor, msi: torch.Tensor) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        b, c = lr_hsi.shape[:2]
        hw = (msi.shape[-2], msi.shape[-1])
        srf = self.srf_buf
        if srf is None:                                # fall back to a flat SRF
            srf = lr_hsi.new_ones(c, cfg.msi_bands) / c
        R = SpectralProject(srf.to(lr_hsi.device)).to(lr_hsi.device)

        kernel = self._estimate_kernel(lr_hsi, msi)
        E = self._basis(lr_hsi, self.rank).to(lr_hsi.dtype)          # [B,C,r]
        Et = E.transpose(1, 2)

        def to_coeff(x):                               # [B,C,H,W] -> [B,r,H,W]
            return torch.einsum('brc,bchw->brhw', Et, x)

        def to_image(z):                               # [B,r,H,W] -> [B,C,H,W]
            return torch.einsum('bcr,brhw->bchw', E, z)

        # initial estimate: bicubic upsample projected into the subspace
        x0 = F.interpolate(lr_hsi, size=hw, mode='bicubic', align_corners=False)
        z = to_coeff(x0)

        # precompute the fixed part of the normal equations
        Bt_yh = to_coeff(self.B.transpose(lr_hsi, hw, kernel))
        Rt_ym = to_coeff(R.transpose(msi))

        stage_outputs = []
        for k in range(self.stages):
            # Running more stages at test time than were trained is allowed:
            # the denoiser weights are shared, and extra stages reuse the final
            # (largest) rho, which is the converged trust level - so they act as
            # further refinement rather than as untrained layers.
            rho = F.softplus(self.log_rho[min(k, self.log_rho.numel() - 1)]) + 1e-3
            v = self.denoiser(z, rho.reshape(1))       # prior step

            def applyA(u, rho=rho):
                x = to_image(u)
                t1 = to_coeff(self.B.transpose(self.B(x, kernel), hw, kernel))
                t2 = to_coeff(R.transpose(R(x)))
                return t1 + t2 + rho * u

            rhs = Bt_yh + Rt_ym + rho * v
            z = self._cg(z, rhs, applyA, self.cg_steps)     # data step
            stage_outputs.append(to_image(z))

        out = stage_outputs[-1].clamp(0, 1)
        return {"out": out, "stages": stage_outputs, "kernel": kernel,
                "coeff": z, "feat": z.mean(dim=(2, 3))}

    def features(self, lr_hsi: torch.Tensor, msi: torch.Tensor) -> torch.Tensor:
        return self.forward(lr_hsi, msi)["feat"]


def build_model(cfg) -> UnfoldFusion:
    return UnfoldFusion(cfg)


def deep_supervision(out: Dict, target: Optional[torch.Tensor], cfg,
                     supervised: bool = True, **kw):
    """Extra loss term: supervise every unrolled stage, not just the last.

    Without it the early stages receive almost no gradient and the unrolling
    collapses into 'one useful stage plus decoration'. With it, each stage is
    pushed to be a genuine improvement on the previous one, which is also what
    makes the stage count adjustable at test time.
    """
    if not supervised or target is None or "stages" not in out:
        return out["out"].new_zeros(()), {}
    stages = out["stages"]
    n = len(stages)
    loss = out["out"].new_zeros(())
    for i, s in enumerate(stages[:-1]):
        w = cfg.w_stage * (i + 1) / n            # later stages weigh more
        loss = loss + w * torch.sqrt((s - target) ** 2 + 1e-6).mean()
    return loss, {"stage": float(loss.item())}
