"""Quality metrics. Inputs are (C, H, W) tensors in [0, 1] (global-max normalised).

Headline PSNR (`PSNR`): 10 log10(1 / MSE) over the whole cube, i.e. peak = the dataset's global max (data
are divided by it). This matches TIP'26 Table IV: its RMSE column is in raw DN, and 20 log10(max_DN / RMSE_DN)
reproduces its PSNR column to within ~1 dB.
Also reported: `PSNR_psrt` (peak = each test image's own max, PSRT metrics.py), `MPSNR_peak1` (band-wise
mean, peak 1) and `SSIM_psrt` (PSRT cal_ssim.py: fixed C1 = 0.01^2, C2 = 0.03^2, i.e. data range 1 = the
dataset peak, zero padding). Headline `SSIM` uses data range = the image's own max, which is stricter on
dark images.
`SCC` and `Q2n` follow the pansharpening toolbox of Vivone et al. as shipped in DLPan-Toolbox
(Quality_Indices/SCC.m, q2n.m, onions_quality.m, onion_mult*.m, norm_blocco.m): SCC is the normalised
cross-correlation of Sobel gradient magnitudes; Q2n uses 32x32 blocks, shift 32, on the image in DN
(x DN_SCALE, rounded to uint16 as q2n.m does).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


DN_SCALE = 15133.0  # DN value that maps to 1 (Chikusei crop max); train.py sets it per dataset


def psnr(gt, x):
    return float(10 * torch.log10(1.0 / torch.mean((gt - x) ** 2)))


def psnr_psrt(gt, x):
    mse = torch.mean((gt - x) ** 2)
    return float(10 * torch.log10(gt.max() ** 2 / mse))


def mpsnr_peak1(gt, x):
    mse = torch.mean((gt - x) ** 2, dim=(1, 2)).clamp_min(1e-12)
    return float(torch.mean(10 * torch.log10(1.0 / mse)))


def sam(gt, x, eps=1e-8):
    g, f = gt.flatten(1), x.flatten(1)
    cos = (g * f).sum(0) / (g.norm(dim=0) * f.norm(dim=0) + eps)
    return float(torch.rad2deg(torch.acos(cos.clamp(-1, 1))).mean())


def ergas(gt, x, ratio=4):
    rmse = torch.sqrt(torch.mean((gt - x) ** 2, dim=(1, 2)))
    mean = gt.mean(dim=(1, 2)).clamp_min(1e-12)
    return float(100 / ratio * torch.sqrt(torch.mean((rmse / mean) ** 2)))


def rmse_dn(gt, x):
    return float(torch.sqrt(torch.mean((gt - x) ** 2))) * DN_SCALE


def cc(gt, x):
    g = gt.flatten(1) - gt.flatten(1).mean(1, keepdim=True)
    f = x.flatten(1) - x.flatten(1).mean(1, keepdim=True)
    return float(((g * f).sum(1) / (g.norm(dim=1) * f.norm(dim=1) + 1e-12)).mean())


def ssim(gt, x, win=11, sigma=1.5):
    """Band-wise Gaussian SSIM, data range = max(GT), averaged over bands."""
    L = gt.max()
    ax = torch.arange(win, dtype=gt.dtype, device=gt.device) - (win - 1) / 2
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    k = (g[:, None] * g[None, :])[None, None].expand(gt.shape[0], 1, win, win)
    f = lambda t: F.conv2d(t[None], k, groups=gt.shape[0])[0]
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    mu1, mu2 = f(gt), f(x)
    s11, s22, s12 = f(gt * gt) - mu1 ** 2, f(x * x) - mu2 ** 2, f(gt * x) - mu1 * mu2
    m = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (s11 + s22 + c2))
    return float(m.mean())


def ssim_psrt(gt, x, win=11, sigma=1.5):
    """SSIM exactly as PSRT's cal_ssim.py: data range 1, zero padding, mean over bands and pixels."""
    ax = torch.arange(win, dtype=gt.dtype, device=gt.device) - win // 2
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    k = (g[:, None] * g[None, :])[None, None].expand(gt.shape[0], 1, win, win)
    f = lambda t: F.conv2d(t[None], k, padding=win // 2, groups=gt.shape[0])[0]
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu1, mu2 = f(gt), f(x)
    s11, s22, s12 = f(gt * gt) - mu1 ** 2, f(x * x) - mu2 ** 2, f(gt * x) - mu1 * mu2
    m = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (s11 + s22 + c2))
    return float(m.mean())


def _sobel_mag(x):
    """SCC.m: Sobel gradient magnitude of I(2:end-1, 2:end-1), imfilter = correlation, zero padding."""
    x = x[:, 1:-1, 1:-1]
    sy = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]], dtype=x.dtype, device=x.device)
    k = torch.stack([sy, sy.T])[:, None].repeat(x.shape[0], 1, 1, 1)       # (2C, 1, 3, 3)
    g = F.conv2d(x[None], k, padding=1, groups=x.shape[0])[0].view(x.shape[0], 2, *x.shape[1:])
    return torch.sqrt(g[:, 0] ** 2 + g[:, 1] ** 2)


def scc(gt, x):
    a, b = _sobel_mag(x), _sobel_mag(gt)
    return float((a * b).sum() / torch.sqrt((a * a).sum()) / torch.sqrt((b * b).sum()))


# ----------------------------------------------------------------------------------------------- Q2n
def _conj(v):
    return torch.cat([v[..., :1], -v[..., 1:]], -1)


def onion_mult(o1, o2):
    """Literal port of onion_mult.m / onion_mult2D.m; hypercomplex components on the last axis."""
    n = o1.shape[-1]
    if n == 1:
        return o1 * o2
    h = n // 2
    a, b = o1[..., :h], _conj(o1[..., h:])
    c, d = o2[..., :h], _conj(o2[..., h:])
    if n == 2:
        return torch.cat([a * c - d * b, a * d + c * b], -1)
    return torch.cat([onion_mult(a, c) - onion_mult(d, _conj(b)), onion_mult(_conj(a), d) + onion_mult(c, b)], -1)


_TABLES: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def _onion_table(n: int):
    """e_i * e_j = sign[i, j] * e_idx[i, j] for the product above (checked: exactly one +-1 entry)."""
    if n not in _TABLES:
        e = torch.eye(n, dtype=torch.float64)
        prod = onion_mult(e[:, None, :].expand(n, n, n), e[None, :, :].expand(n, n, n))  # (i, j, k)
        idx = prod.abs().argmax(-1)
        sign = prod.gather(-1, idx[..., None])[..., 0]
        assert torch.all(sign.abs() == 1) and torch.all((prod != 0).sum(-1) == 1), "not a signed permutation"
        _TABLES[n] = (idx, sign)
    return _TABLES[n]


def _onion_mean_product(x, y):
    """mean over pixels of onion_mult(x_p, y_p). x, y: (B, P, n) -> (B, n), via the (n, n) cross moments."""
    n = x.shape[-1]
    idx, sign = _onion_table(n)
    idx, sign = idx.to(x.device), sign.to(x.dtype).to(x.device)
    m = torch.einsum("bpi,bpj->bij", x, y) / x.shape[1]
    out = torch.zeros(x.shape[0], n, dtype=x.dtype, device=x.device)
    return out.scatter_add_(1, idx.flatten()[None].expand(x.shape[0], -1), (m * sign).flatten(1))


def onions_quality(d1, d2, fast=True):
    """onions_quality.m on a batch of blocks. d1 = GT, d2 = fused, both (B, P, n) with P = block^2 pixels."""
    d1, d2 = d1.double(), d2.double()
    p, n = d1.shape[1], d1.shape[2]
    d2 = _conj(d2)
    s = d1.mean(1, keepdim=True)                                             # norm_blocco: mean2
    t = d1.std(1, keepdim=True)                                              # std2 (N-1)
    t = torch.where(t == 0, torch.full_like(t, 2.220446049250313e-16), t)
    d1 = (d1 - s) / t + 1
    first = torch.zeros(n, dtype=torch.bool, device=d1.device); first[0] = True
    d2_norm = torch.where(first, (d2 - s) / t + 1, -(((-d2) - s) / t + 1))
    d2_zero = torch.where(first, d2 - s + 1, -(-d2 - s + 1))                 # the `s == 0` branch
    d2 = torch.where(s == 0, d2_zero, d2_norm)
    m1, m2 = d1.mean(1), d2.mean(1)                                          # (B, n)
    mod_q1m, mod_q2m = m1.norm(dim=1), m2.norm(dim=1)
    f = p / (p - 1)
    int1 = f * (d1 ** 2).sum(2).mean(1)
    int2 = f * (d2 ** 2).sum(2).mean(1)
    termine2 = mod_q1m * mod_q2m
    termine4 = mod_q1m ** 2 + mod_q2m ** 2
    termine3 = int1 + int2 - f * (mod_q1m ** 2 + mod_q2m ** 2)
    mean_bias = 2 * termine2 / termine4
    if fast:
        qv = f * _onion_mean_product(d1, d2)
    else:
        qv = f * onion_mult(d1, d2).mean(1)
    qm = onion_mult(m1, m2)
    q = (qv - f * qm) * (mean_bias * 2 / termine3)[:, None]
    degenerate = torch.zeros_like(q); degenerate[:, -1] = mean_bias
    return torch.where((termine3 == 0)[:, None], degenerate, q)


def q2n(gt, x, block=32, shift=32, scale=None, fast=True):
    """q2n.m (Q_blocks_size = Q_shift = 32). gt, x: (C, H, W) in [0, 1]."""
    c, h, w = gt.shape
    sx, sy = math.ceil(h / shift), math.ceil(w / shift)
    e1, e2 = (sx - 1) * shift + block - h, (sy - 1) * shift + block - w
    imgs = []
    for im in (gt, x):
        im = im.double()
        if e1 or e2:  # mirror-extend right / bottom edges exactly as q2n.m
            im = torch.cat([im, im[:, :, w - e2:].flip(-1)], -1) if e2 else im
            im = torch.cat([im, im[:, h - e1:, :].flip(-2)], -2) if e1 else im
        im = torch.round((im * (scale or DN_SCALE)).clamp(0, 65535))          # uint16()
        nb = 2 ** math.ceil(math.log2(c))
        if nb != c:
            im = torch.cat([im, im.new_zeros(nb - c, *im.shape[1:])], 0)
        blocks = torch.stack([im[:, j * shift:j * shift + block, i * shift:i * shift + block]
                              for j in range(sx) for i in range(sy)])        # (B, n, bh, bw)
        imgs.append(blocks.flatten(2).transpose(1, 2))                       # (B, P, n)
    q = onions_quality(imgs[0], imgs[1], fast=fast)
    return float(q.norm(dim=1).mean())


ALL = dict(PSNR=psnr, PSNR_psrt=psnr_psrt, MPSNR_peak1=mpsnr_peak1, SSIM=ssim, SSIM_psrt=ssim_psrt, SAM=sam,
           ERGAS=ergas, RMSE_DN=rmse_dn, CC=cc, SCC=scc)
FULL = dict(ALL, Q2n=q2n)


def evaluate(gts, preds, full=False):
    """Mean of every metric over a list/stack of (C, H, W) images, in float64 on the inputs' device.

    `full` adds Q2n (slower; used for the final test only).
    """
    fns = FULL if full else ALL
    out = {k: 0.0 for k in fns}
    for g, p in zip(gts, preds):
        g, p = g.double(), p.double().clamp(0, 1)
        for k, fn in fns.items():
            out[k] += fn(g, p)
    n = len(gts)
    return {k: v / n for k, v in out.items()}
