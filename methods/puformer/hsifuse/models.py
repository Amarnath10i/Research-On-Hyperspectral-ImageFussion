"""PUFormer: Physics-Unfolded transFormer for HSI-MSI fusion, plus the SSRNet anchor.

Each of K stages:
    Z_k   = X_{k-1} - eta_H * Dt(D X - Y_H) - eta_M * Rt(R X - Y_M)     exact data-fidelity step
    X_k   = Z_k + P_k([Z_k, Y_M, Dt-residual], memory)                    learned prior (U-Net)
The prior is a 3-level U-Net of Restormer blocks: transposed (channel) attention,
which is attention across spectral feature maps, plus a gated depth-wise FFN.
Features of each stage's decoder are passed to the next stage (cross-stage memory).
The image path stays in float32; only the prior runs under autocast.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import Degradation, kron_prox, kron_prox_learned, null_gate


class LayerNorm2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w, self.b = nn.Parameter(torch.ones(c)), nn.Parameter(torch.zeros(c))

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        return (x - mu) / torch.sqrt(var + 1e-6) * self.w[:, None, None] + self.b[:, None, None]


class MDTA(nn.Module):
    """Multi-Dconv-head transposed attention (attention over channels)."""

    def __init__(self, c, heads):
        super().__init__()
        self.h = heads
        self.t = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(c, c * 3, 1, bias=False)
        self.dw = nn.Conv2d(c * 3, c * 3, 3, padding=1, groups=c * 3, bias=False)
        self.out = nn.Conv2d(c, c, 1, bias=False)

    def forward(self, x):
        n, c, hh, ww = x.shape
        q, k, v = self.dw(self.qkv(x)).chunk(3, 1)
        q, k, v = (t.reshape(n, self.h, c // self.h, hh * ww) for t in (q, k, v))
        q, k = F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1)
        a = (q @ k.transpose(-2, -1) * self.t).softmax(-1).to(v.dtype)
        return self.out((a @ v).reshape(n, c, hh, ww))


class GDFN(nn.Module):
    def __init__(self, c, mult=2.66):
        super().__init__()
        h = int(c * mult)
        self.inp = nn.Conv2d(c, h * 2, 1, bias=False)
        self.dw = nn.Conv2d(h * 2, h * 2, 3, padding=1, groups=h * 2, bias=False)
        self.out = nn.Conv2d(h, c, 1, bias=False)

    def forward(self, x):
        a, b = self.dw(self.inp(x)).chunk(2, 1)
        return self.out(F.gelu(a) * b)


class WindowMSA(nn.Module):
    """Ablation: spatial self-attention inside non-overlapping ws x ws windows (tokens = pixels), with the
    same 1x1 + depth-wise 3x3 projections as MDTA, so only the attention axis changes."""

    def __init__(self, c, heads, ws=8):
        super().__init__()
        self.h, self.ws = heads, ws
        self.qkv = nn.Conv2d(c, c * 3, 1, bias=False)
        self.dw = nn.Conv2d(c * 3, c * 3, 3, padding=1, groups=c * 3, bias=False)
        self.out = nn.Conv2d(c, c, 1, bias=False)

    def forward(self, x):
        n, c, hh, ww = x.shape
        s = min(self.ws, hh, ww)
        ph, pw = (-hh) % s, (-ww) % s
        t = self.dw(self.qkv(x))
        if ph or pw:
            t = F.pad(t, (0, pw, 0, ph), mode="replicate")
        H, W, d = hh + ph, ww + pw, c // self.h
        t = t.reshape(n, 3, self.h, d, H // s, s, W // s, s)
        t = t.permute(1, 0, 4, 6, 2, 5, 7, 3).reshape(3, -1, self.h, s * s, d)
        q, k, v = t[0], t[1], t[2]
        a = (q.float() @ k.float().transpose(-2, -1) * d ** -0.5).softmax(-1).to(v.dtype)
        o = (a @ v).reshape(n, H // s, W // s, self.h, s, s, d)
        o = o.permute(0, 3, 6, 1, 4, 2, 5).reshape(n, c, H, W)[..., :hh, :ww]
        return self.out(o)


class ConvMixer(nn.Module):
    """Ablation: attention replaced by a convolutional mixer (1x1 -> depth-wise 3x3 -> GELU -> 1x1)."""

    def __init__(self, c, heads=None):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(c, c * 2, 1, bias=False),
                               nn.Conv2d(c * 2, c * 2, 3, padding=1, groups=c * 2, bias=False), nn.GELU(),
                               nn.Conv2d(c * 2, c, 1, bias=False))

    def forward(self, x):
        return self.f(x)


MIXERS = dict(mdta=MDTA, window=WindowMSA, conv=ConvMixer)


class Block(nn.Module):
    def __init__(self, c, heads, mixer="mdta"):
        super().__init__()
        self.n1, self.att, self.n2, self.ffn = LayerNorm2d(c), MIXERS[mixer](c, heads), LayerNorm2d(c), GDFN(c)

    def forward(self, x):
        x = x + self.att(self.n1(x))
        return x + self.ffn(self.n2(x))


def stack(c, heads, n, mixer="mdta"):
    return nn.Sequential(*[Block(c, heads, mixer) for _ in range(n)])


class Prior(nn.Module):
    """3-level Restormer U-Net: in_ch -> residual on the HSI."""

    def __init__(self, in_ch, out_ch, w=48, depth=(2, 3, 4), heads=(1, 2, 4), mem=True, mixer="mdta", film=0,
                 lr_ch=0, xfer=0):
        super().__init__()
        # null-space transfer prior (both parts start at zero, i.e. at the plain U-Net):
        # lr_ch > 0  measurement injection: the LR-HSI enters the bottleneck, whose grid is the LR grid at x4
        # xfer > 0   spectral transfer branch: a per-pixel network maps the MSI's fine detail (in the hyperspectral
        #            bands) and the current spectra to the detail of the bands the MSI does not see
        self.inj = nn.Conv2d(lr_ch, 4 * w, 3, padding=1) if lr_ch else None
        if self.inj is not None:
            nn.init.zeros_(self.inj.weight), nn.init.zeros_(self.inj.bias)
        self.xfer = nn.Sequential(nn.Conv2d(xfer, 64, 1), nn.GELU(), nn.Conv2d(64, 64, 1), nn.GELU(),
                                  nn.Conv2d(64, out_ch, 1)) if xfer else None
        if self.xfer is not None:
            nn.init.zeros_(self.xfer[-1].weight), nn.init.zeros_(self.xfer[-1].bias)
        # film > 0: features are modulated (scale and shift per channel) by an observation-model embedding of that
        # size after the embedding, at the bottleneck input and after the refinement (zero init: no modulation)
        self.film = nn.Linear(film, 2 * (w + 4 * w + w)) if film else None
        if self.film is not None:
            nn.init.zeros_(self.film.weight), nn.init.zeros_(self.film.bias)
        self.w = w
        self.embed = nn.Conv2d(in_ch, w, 3, padding=1)
        self.mem = nn.Conv2d(w * 2, w, 1) if mem else None
        self.e1, self.d12 = stack(w, heads[0], depth[0], mixer), nn.Conv2d(w, w * 2, 4, 2, 1)
        self.e2, self.d23 = stack(w * 2, heads[1], depth[1], mixer), nn.Conv2d(w * 2, w * 4, 4, 2, 1)
        self.mid = stack(w * 4, heads[2], depth[2], mixer)
        self.u32, self.r2 = nn.ConvTranspose2d(w * 4, w * 2, 2, 2), nn.Conv2d(w * 4, w * 2, 1)
        self.dec2 = stack(w * 2, heads[1], depth[1], mixer)
        self.u21, self.r1 = nn.ConvTranspose2d(w * 2, w, 2, 2), nn.Conv2d(w * 2, w, 1)
        self.dec1 = stack(w, heads[0], depth[0], mixer)
        self.refine = stack(w, heads[0], 2, mixer)
        self.tail = nn.Conv2d(w, out_ch, 3, padding=1)
        nn.init.zeros_(self.tail.weight), nn.init.zeros_(self.tail.bias)  # start at the physics step

    def forward(self, x, memory=None, emb=None, lr=None, xin=None, deg=None):
        mod = [lambda t: t] * 3
        if self.film is not None and emb is not None:
            w = self.w
            g = self.film(emb).split([w, w, 4 * w, 4 * w, w, w], 1)
            mod = [lambda t, a=g[2 * i], b=g[2 * i + 1]: t * (1 + a[:, :, None, None].to(t.dtype))
                   + b[:, :, None, None].to(t.dtype) for i in range(3)]
        f = self.embed(x)
        if self.mem is not None and memory is not None:
            f = self.mem(torch.cat([f, memory], 1))
        e1 = self.e1(mod[0](f))
        e2 = self.e2(self.d12(e1))
        b = self.d23(e2)
        if self.inj is not None and lr is not None:
            if lr.shape[-2:] != b.shape[-2:]:
                lr = F.interpolate(lr, size=b.shape[-2:], mode="bilinear", align_corners=False)
            b = b + self.inj(lr.to(b.dtype))
        m = self.mid(mod[1](b))
        d2 = self.dec2(self.r2(torch.cat([self.u32(m), e2], 1)))
        d1 = self.dec1(self.r1(torch.cat([self.u21(d2), e1], 1)))
        d1 = mod[2](self.refine(d1))
        out = self.tail(d1)
        if self.xfer is not None and xin is not None:
            out = out + self.xfer(xin.to(out.dtype))
        return out, d1


class EigenBlock(nn.Module):
    """One block of the eigen-operator prior: global mixing in the spatial eigenbasis of the observation operator
    (U_r^T F U_c per channel) with per-channel filters generated from the axis eigenvalues (not from positions, so
    the block does not depend on the image size, the scale factor or the sensor), a local depth-wise path, and a
    gated feed-forward network; all three in residual branches."""

    def __init__(self, c):
        super().__init__()
        self.n1, self.n2 = LayerNorm2d(c), LayerNorm2d(c)
        self.filt = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, c))
        self.mix = nn.Conv2d(c, c, 1)
        self.local = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, groups=c), nn.GELU(), nn.Conv2d(c, c, 1))
        self.ffn = GDFN(c, 2.66)
        nn.init.zeros_(self.mix.weight), nn.init.zeros_(self.mix.bias)

    def forward(self, x, kf):
        y = self.n1(x)
        lr_, lc_ = kf.lam_r.to(y.dtype), kf.lam_c.to(y.dtype)
        h, w = lr_.numel(), lc_.numel()
        fr = torch.stack([torch.log(lr_ + 1e-6), (lr_ > 1e-8 * lr_.max()).to(y.dtype),
                          torch.linspace(0, 1, h, device=y.device, dtype=y.dtype)], 1)       # (H, 3)
        fc = torch.stack([torch.log(lc_ + 1e-6), (lc_ > 1e-8 * lc_.max()).to(y.dtype),
                          torch.linspace(0, 1, w, device=y.device, dtype=y.dtype)], 1)       # (W, 3)
        feat = torch.cat([fr[:, None].expand(h, w, 3), fc[None].expand(h, w, 3)], -1)       # (H, W, 6)
        g = self.filt(feat).permute(2, 0, 1)[None]                                           # (1, C, H, W)
        ur, uc = kf.ur.to(y.dtype), kf.uc.to(y.dtype)
        e = torch.einsum("hk,nchw,wl->nckl", ur, y, uc)                                     # eigen coefficients
        glob = torch.einsum("hk,nckl,wl->nchw", ur, e * g, uc)
        x = x + self.mix(glob) + self.local(y)
        return x + self.ffn(self.n2(x))


class EigenPrior(nn.Module):
    """Eigen-operator prior (replaces the U-Net): a full-resolution stack of EigenBlocks. Same interface as Prior;
    needs the operator (deg) to build its eigenbasis (cached per image size)."""

    def __init__(self, in_ch, out_ch, w=64, depth=8, mem=True, film=0):
        super().__init__()
        self.w = w
        self.embed = nn.Conv2d(in_ch, w, 3, padding=1)
        self.mem = nn.Conv2d(w * 2, w, 1) if mem else None
        self.film = nn.Linear(film, 2 * w) if film else None
        if self.film is not None:
            nn.init.zeros_(self.film.weight), nn.init.zeros_(self.film.bias)
        self.blocks = nn.ModuleList(EigenBlock(w) for _ in range(depth))
        self.tail = nn.Conv2d(w, out_ch, 3, padding=1)
        nn.init.zeros_(self.tail.weight), nn.init.zeros_(self.tail.bias)

    def forward(self, x, memory=None, emb=None, lr=None, xin=None, deg=None):
        from .ops import kron_factors
        f = self.embed(x)
        if self.mem is not None and memory is not None:
            f = self.mem(torch.cat([f, memory], 1))
        if self.film is not None and emb is not None:
            a, b = self.film(emb).split([self.w, self.w], 1)
            f = f * (1 + a[:, :, None, None].to(f.dtype)) + b[:, :, None, None].to(f.dtype)
        kf = kron_factors(deg, f.shape[-2], f.shape[-1], f.device)
        for blk in self.blocks:
            f = blk(f, kf)
        return self.tail(f), f


def msi_backproject(deg, ym):
    """R^+ Y_M: the minimum-norm spectra in the B hyperspectral bands that reproduce the HR-MSI exactly. A network
    that receives this instead of Y_M does not depend on the MSI sensor (number of bands or their responses)."""
    srf = deg.srf.to(torch.float32)                                   # (B, M)
    g = srf.T @ srf
    w = srf @ torch.linalg.inv(g + 1e-8 * torch.trace(g) * torch.eye(g.shape[0], device=g.device))
    return torch.einsum("nmhw,bm->nbhw", ym.float(), w).to(ym.dtype)


def noise_level(x):
    """Per image, the noise standard deviation relative to the image RMS: the median absolute diagonal second
    difference of every band (Donoho's robust estimator; 2 sigma for white noise), averaged over the bands."""
    x = x.float()
    d = x[..., 1:, 1:] - x[..., :-1, 1:] - x[..., 1:, :-1] + x[..., :-1, :-1]
    sd = d.abs().flatten(2).median(2).values / (0.6745 * 2.0)                 # (N, C)
    rms = x.pow(2).flatten(2).mean(2).sqrt()
    return (sd.mean(1) / rms.mean(1).clamp_min(1e-8)).clamp_min(1e-6)


class PUFormer(nn.Module):
    """Ablation switches (the defaults are the full model):
        physics=False    no data-consistency step (Z_k = X_{k-1}) and no residual input: a cascade of priors
        res_input=False  the prior does not get s^2 Dt(D Z - Y_H) (those channels are fed zeros)
        memory=False     no cross-stage memory
        mixer            'mdta' (transposed / channel attention), 'window' (8x8 spatial windows) or 'conv'
    Observation-model conditioning (omc=True): a code c = (sigma / sigma0 - 1, SRF shift / 12 nm, log10 n_H + 2,
    log10 n_M + 2) describes how the inputs were observed - the blur width and SRF shift of the operator in the data
    steps (given, or estimated from the test pair) and the noise level n of each observation, estimated from the
    image itself relative to its RMS. The data-step sizes become softplus(eta_k + u_k . c), the noise-dependent
    weighting of the two data terms, and an embedding of c modulates every prior (FiLM). All of it starts at zero,
    i.e. at the unconditioned model.
    """
    CODE = 4

    def __init__(self, deg: Degradation, bands=128, msi=8, stages=3, width=48,
                 depth=(2, 3, 4), heads=(1, 2, 4), physics=True, res_input=True, memory=True, mixer="mdta",
                 omc=False, dstep="grad", cg_iters=5, mres=False, reg=False, sa=False, nsp=False, ntp=False,
                 cwd=False, unc=False, prior="unet", nomsi=False, nohsi=False):
        super().__init__()
        self.deg, self.K, self.bands = deg, stages, bands
        # input ablations: without the HR-MSI the network is a single-image super-resolution network (data step on
        # the LR-HSI only, no MSI input); without the LR-HSI it is spectral super-resolution from the HR-MSI (start at
        # R^+ Y_M, data step on the HR-MSI only). Removed inputs are replaced by zeros everywhere.
        self.nomsi, self.nohsi = nomsi, nohsi
        # observability-aware uncertainty: a head on the last stage's features predicts the log-variance of the
        # error left in the fusion null space (the observed part is fixed by the data step); read model.last_logvar
        self.unc = unc
        self.last_logvar = None
        if unc:                               # on detached features: it cannot change the reconstruction
            self.unc_head = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
                                          nn.Conv2d(width, bands, 3, padding=1))
            nn.init.zeros_(self.unc_head[-1].weight), nn.init.constant_(self.unc_head[-1].bias, -9.0)
        self.ntp = ntp                        # null-space transfer prior (measurement injection + spectral transfer)
        # confidence-weighted data step: per stage, two small networks map the local disagreement with each
        # observation to a confidence in (0, 1); low-confidence measurements are replaced by the estimate's own
        # prediction before the data step (start: confidence ~0.98 everywhere)
        self.cwd = cwd
        if cwd:
            def conf():
                net = nn.Sequential(nn.Conv2d(2, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 16, 3, padding=1), nn.GELU(),
                                    nn.Conv2d(16, 1, 3, padding=1))
                nn.init.zeros_(net[-1].weight), nn.init.constant_(net[-1].bias, 4.0)
                return net
            self.conf_h = nn.ModuleList(conf() for _ in range(stages))
            self.conf_m = nn.ModuleList(conf() for _ in range(stages))
        self.reg = reg
        self.sa = sa                          # sensor-independent: the prior sees R^+ Y_M (B bands), not Y_M (M bands)
        # null-space-gated prior: the correction of stage k acts freely in the fusion null space and with weight
        # gamma_k in the observed subspace; gamma_k = sigmoid(g_k [+ w_k . c]) follows the noise when omc is on
        self.nsp = nsp
        if nsp:
            self.nsp_gate = nn.Parameter(torch.zeros(stages))
        if reg:                               # + column blur width and the two sub-pixel offsets
            self.CODE = 7
        self.physics, self.res_input = physics, res_input and physics
        self.dstep, self.cg_iters, self.mres = dstep, cg_iters, mres and physics
        self.amp = False                      # fp16 autocast for the priors only
        self.eta_h = nn.Parameter(torch.full((stages,), 1.0))
        self.eta_m = nn.Parameter(torch.full((stages,), 1.0))
        self.omc = omc
        if dstep == "kronlearn":            # learned spectral-domain gains of the closed-form data step, per stage
            def mlp():
                net = nn.Sequential(nn.Linear(1 + self.CODE, 16), nn.SiLU(), nn.Linear(16, 1))
                nn.init.zeros_(net[-1].weight), nn.init.zeros_(net[-1].bias)
                return net
            self.gain_s = nn.ModuleList(mlp() for _ in range(stages))
            self.gain_b = nn.ModuleList(mlp() for _ in range(stages))
        if omc:
            self.step_h = nn.Parameter(torch.zeros(stages, self.CODE))
            self.step_m = nn.Parameter(torch.zeros(stages, self.CODE))
            if nsp:
                self.nsp_code = nn.Parameter(torch.zeros(stages, self.CODE))
            self.code_net = nn.Sequential(nn.Linear(self.CODE, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU())
        in_ch = bands + (bands if sa else msi) + bands    # Z, Y_M (or R^+ Y_M), Dt(D Z - Y_H)
        in_ch += bands if self.mres else 0   # (B / M) Rt(R Z - Y_M)
        if prior == "eon":                    # eigen-operator prior instead of the U-Net
            self.priors = nn.ModuleList(
                EigenPrior(in_ch, bands, w=64, depth=8, mem=memory and k > 0, film=64 if omc else 0)
                for k in range(stages))
        else:
            self.priors = nn.ModuleList(
                Prior(in_ch, bands, width, depth, heads, mem=memory and k > 0, mixer=mixer, film=64 if omc else 0,
                      lr_ch=bands if ntp else 0, xfer=2 * bands if ntp else 0)
                for k in range(stages))
        if unc and prior == "eon":            # the uncertainty head reads the prior's feature width
            self.unc_head[0] = nn.Conv2d(64, width, 3, padding=1)

    def obs_code(self, yh, ym):
        """(N, 4) observation-model code: operator parameters of self.deg and the estimated noise levels."""
        n = yh.shape[0]
        s0 = self.deg.scale / 2.0
        sig, sh = getattr(self.deg, "sigma", s0), getattr(self.deg, "shift_nm", 0.0)
        c = [torch.full((n,), sig / s0 - 1.0, device=yh.device), torch.full((n,), sh / 12.0, device=yh.device),
             torch.log10(noise_level(yh)) + 2.0, torch.log10(noise_level(ym)) + 2.0]
        if self.reg:
            c += [torch.full((n,), getattr(self.deg, "sigma_c", sig) / s0 - 1.0, device=yh.device),
                  torch.full((n,), getattr(self.deg, "off_r", 0.0), device=yh.device),
                  torch.full((n,), getattr(self.deg, "off_c", 0.0), device=yh.device)]
        return torch.stack(c, 1)

    def confident(self, k, x, yh, ym):
        """Confidence-weighted observations for stage k: w * Y + (1 - w) * A x, with w in (0, 1) from the local
        disagreement |A x - Y| (band-averaged, relative to its median over the image). A correct estimate is still a
        fixed point (A x = Y gives back Y for any w)."""
        def feats(r):
            r = r.abs().mean(1, keepdim=True)
            r = r / (r.flatten(1).median(1).values.view(-1, 1, 1, 1) + 1e-8)
            return torch.cat([torch.log1p(r), r.clamp(max=20.0) / 20.0], 1)
        dx, rx = self.deg.D(x), self.deg.R(x)
        wh = torch.sigmoid(self.conf_h[k](feats(dx - yh)))
        wm = torch.sigmoid(self.conf_m[k](feats(rx - ym)))
        self.last_conf = (wh.detach(), wm.detach())       # 1 - w: where the inputs disagree (change / corruption map)
        return wh * yh + (1 - wh) * dx, wm * ym + (1 - wm) * rx

    def prox_cg(self, x, yh, ym, a, b):
        """Exact proximal data step: argmin_z 1/2||z - x||^2 + a s^2/2 ||D z - Y_H||^2 + b/2 ||R z - Y_M||^2, i.e.
        (I + a s^2 Dt D + b Rt R) z = x + a s^2 Dt Y_H + b Rt Y_M, by conjugate gradients started at x (the first
        iteration is a gradient step with an exact line search). With the exact operators the ground truth is a
        fixed point for every a, b (Proposition 1 extended to the proximal step)."""
        s2 = self.deg.scale ** 2
        op = lambda v: v + a * s2 * self.deg.Dt(self.deg.D(v)) + b * self.deg.Rt(self.deg.R(v))
        dot = lambda u, v: (u * v).sum((1, 2, 3), keepdim=True)
        z = x
        r = x + a * s2 * self.deg.Dt(yh) + b * self.deg.Rt(ym) - op(z)
        p, rs = r, dot(r, r)
        for _ in range(self.cg_iters):
            q = op(p)
            step = rs / dot(p, q).clamp_min(1e-30)
            z, r = z + step * p, r - step * q
            rs_new = dot(r, r)
            p, rs = r + rs_new / rs.clamp_min(1e-30) * p, rs_new
        return z

    def fidelity_grad(self, x, yh, ym):
        rh = self.deg.D(x) - yh
        rm = self.deg.R(x) - ym
        return self.deg.Dt(rh), self.deg.Rt(rm)

    def forward(self, yh, ym, return_all=False):
        s2 = self.deg.scale ** 2
        if self.nomsi:
            ym = torch.zeros_like(ym)
        if self.nohsi:
            yh = torch.zeros_like(yh)
            x = msi_backproject(self.deg, ym)
        else:
            x = F.interpolate(yh, scale_factor=self.deg.scale, mode="bicubic", align_corners=False)
        mem, outs = None, []
        code = emb = None
        if self.omc or self.dstep == "kronlearn":
            code = self.obs_code(yh, ym)
        if self.omc:
            emb = self.code_net(code)
        hp = None
        if self.ntp:                          # the MSI's fine detail (above the LR band), in the hyperspectral bands
            low = F.interpolate(self.deg.D(ym), scale_factor=self.deg.scale, mode="bicubic", align_corners=False)
            hp = msi_backproject(self.deg, ym - low)
        yh0, ym0 = yh, ym
        for k in range(self.K):
            if self.cwd and self.physics:
                yh, ym = self.confident(k, x, yh0, ym0)
            if self.physics:
                # Dt D has gain ~1/s^2 on smooth content under x s decimation, so the H step is scaled by s^2
                if self.omc:
                    a = F.softplus(self.eta_h[k] + code @ self.step_h[k]).view(-1, 1, 1, 1)
                    b = F.softplus(self.eta_m[k] + code @ self.step_m[k]).view(-1, 1, 1, 1)
                else:
                    a, b = F.softplus(self.eta_h[k]), F.softplus(self.eta_m[k])
                if self.nohsi:
                    a = a * 0.0
                if self.nomsi:
                    b = b * 0.0
                if self.dstep == "kronlearn" and getattr(self.deg, "gaussian", False):
                    z = kron_prox_learned(self.deg, x, yh, ym, a * s2, b, self.gain_s[k], self.gain_b[k], code)
                elif self.dstep in ("kron", "kronlearn") and getattr(self.deg, "gaussian", False):
                    z = kron_prox(self.deg, x, yh, ym, a * s2, b)       # closed form (separable exact operators)
                elif self.dstep in ("cg", "kron", "kronlearn"):
                    z = self.prox_cg(x, yh, ym, a, b)
                else:
                    gh, gm = self.fidelity_grad(x, yh, ym)
                    z = x - a * s2 * gh - b * gm
            else:
                z = x
            # the prior sees the raw observations' disagreement (yh0, ym0), also where the data step distrusted them
            rh = self.deg.Dt(self.deg.D(z) - yh0) * s2 if self.res_input and not self.nohsi else torch.zeros_like(z)
            inp = [z, msi_backproject(self.deg, ym0) if self.sa else ym0, rh]
            if self.mres:
                inp.append(torch.zeros_like(z) if self.nomsi else
                           self.deg.Rt(self.deg.R(z) - ym0) * (self.bands / ym0.shape[1]))
            xin = torch.cat([hp, z], 1) if self.ntp else None
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp and x.is_cuda):
                res, mem = self.priors[k](torch.cat(inp, 1), mem, emb, lr=yh0 if self.ntp else None, xin=xin,
                                         deg=self.deg)
            res = res.float()
            if self.nsp and self.physics and getattr(self.deg, "gaussian", False):
                g = self.nsp_gate[k] + (code @ self.nsp_code[k] if self.omc else 0.0)
                res = null_gate(self.deg, res, torch.sigmoid(torch.as_tensor(g)).reshape(-1, 1, 1, 1))
            x = z + res
            outs.append(x)
        if self.unc:
            self.last_logvar = self.unc_head(mem.detach().float()).clamp(-20.0, 5.0)
        return outs if return_all else x


class SSRNet(nn.Module):
    """SSR-NET (Zhang et al., TGRS 2021) - reference anchor, same as baselines/ssrnet."""

    def __init__(self, scale=4, msi=8, bands=128):
        super().__init__()
        self.scale, self.msi, self.bands = scale, msi, bands
        conv = lambda: nn.Sequential(nn.Conv2d(bands, bands, 3, padding=1), nn.ReLU())
        self.conv_fus, self.conv_spat, self.conv_spec = conv(), conv(), conv()

    def forward(self, yh, ym, return_all=False):
        x = F.interpolate(yh, scale_factor=self.scale, mode="bilinear")
        gap = self.bands / (self.msi - 1.0)
        x = x.clone()
        for i in range(self.msi - 1):
            x[:, int(gap * i)] = ym[:, i]
        x[:, self.bands - 1] = ym[:, self.msi - 1]
        x = self.conv_fus(x)
        x = x + self.conv_spat(x)
        x = x + self.conv_spec(x)
        return [x] if return_all else x


def build(name: str, deg: Degradation, **kw):
    """Band counts come from the spectral response (B x M) of the degradation."""
    bands, msi = deg.srf.shape
    if name == "puformer":
        return PUFormer(deg, bands=bands, msi=msi, **kw)
    if name == "ssrnet":
        return SSRNet(scale=deg.scale, msi=msi, bands=bands)
    from .external import build_external        # official code of published methods, cloned at run time
    return build_external(name, bands=bands, msi=msi, scale=deg.scale, deg=deg, sa=kw.get("sa", False))
