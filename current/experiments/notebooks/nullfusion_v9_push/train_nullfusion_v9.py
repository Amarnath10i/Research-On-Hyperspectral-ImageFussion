import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────── helpers ────────────────────────

def window_partition(x, win):
    B, H, W, C = x.shape
    x = x.view(B, H // win, win, W // win, win, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win * win * C)

def window_reverse(windows, win, H, W):
    B = windows.shape[0] // (H // win * W // win)
    x = windows.view(B, H // win, W // win, win, win, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class ChannelAttention(nn.Module):
    def __init__(self, ch, reduction=4):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c = x.shape[:2]
        return x * self.fc(self.avg(x).view(b, c))


class SpatialAttention(nn.Module):
    def __init__(self, kernel=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel, padding=kernel//2, bias=False)
    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            ChannelAttention(ch),
            SpatialAttention()
        )
    def forward(self, x):
        return x + self.block(x)


# ──────────────── Swin Transformer Block ────────────────

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hid = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hid)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hid, dim)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class WindowAttention(nn.Module):
    def __init__(self, dim, win_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        num_patches = win_size * win_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*win_size-1)*(2*win_size-1), num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(win_size)
        coords_w = torch.arange(win_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = coords.view(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += win_size - 1
        relative_coords[:, :, 1] += win_size - 1
        relative_coords[:, :, 0] *= 2 * win_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        B_N, N, C = x.shape
        qkv = self.qkv(x).reshape(B_N, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_N // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_N, N, C)
        return self.proj(out)


class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, win_size, shift=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, win_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)
        self.shift = shift
        self.win_size = win_size

    def forward(self, x, mask_matrix):
        B, H, W, C = x.shape
        residual = x
        x = self.norm1(x)

        if self.shift:
            shifted_x = torch.roll(x, shifts=(-self.win_size//2, -self.win_size//2), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.win_size)
        attn_windows = self.attn(x_windows, mask=attn_windows if attn_mask is not None else None)
        shifted_x = window_reverse(attn_windows, self.win_size, H, W)

        if self.shift:
            x = torch.roll(shifted_x, shifts=(self.win_size//2, self.win_size//2), dims=(1, 2))
        else:
            x = shifted_x

        x = residual + x
        x = x + self.mlp(self.norm2(x))
        return x


# ──────────────── Swin Transformer Layer ────────────────

class SwinLayer(nn.Module):
    def __init__(self, dim, num_heads, win_size=8, depth=2):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(SwinBlock(dim, num_heads, win_size, shift=(i % 2 == 1)))
        self.win_size = win_size

    def _create_mask(self, H, W, device):
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (slice(0, -self.win_size),
                    slice(-self.win_size, -self.win_size//2),
                    slice(-self.win_size//2, None))
        w_slices = (slice(0, -self.win_size),
                    slice(-self.win_size, -self.win_size//2),
                    slice(-self.win_size//2, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.win_size)
        mask_windows = mask_windows.view(-1, self.win_size * self.win_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        B, C, H, W = x.shape
        x_t = x.permute(0, 2, 3, 1)
        mask = self._create_mask(H, W, x.device)
        for blk in self.blocks:
            x_t = blk(x_t, mask)
        return x_t.permute(0, 3, 1, 2).contiguous()


# ──────────────── UNet with Swin Encoder + CNN Decoder ────────────────

class SwinEncoder(nn.Module):
    def __init__(self, in_ch, dims, num_heads_list, depths, win_size=8):
        super().__init__()
        self.patch = nn.Conv2d(in_ch, dims[0], 3, stride=1, padding=1)
        self.swin_layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(len(dims)):
            self.swin_layers.append(
                SwinLayer(dims[i], num_heads_list[i], win_size, depth=depths[i]))
            if i < len(dims) - 1:
                self.downsamples.append(nn.Conv2d(dims[i], dims[i+1], 2, stride=2))

    def forward(self, x):
        features = []
        x = self.patch(x)
        for i, swin in enumerate(self.swin_layers):
            x = swin(x)
            features.append(x)
            if i < len(self.swin_layers) - 1:
                x = self.downsamples[i](x)
        return features


class Decoder(nn.Module):
    def __init__(self, dims, out_ch):
        super().__init__()
        self.upconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        for i in range(len(dims)-1, 0, -1):
            self.upconvs.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(dims[i], dims[i-1], 3, padding=1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True)))
            self.decoder_blocks.append(ResBlock(dims[i-1]))

        self.final = nn.Sequential(
            nn.Conv2d(dims[0], dims[0]//2, 3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dims[0]//2, out_ch, 3, padding=1))

    def forward(self, features):
        x = features[-1]
        for i, (upconv, blk) in enumerate(
                zip(self.upconvs, self.decoder_blocks)):
            x = upconv(x)
            skip = features[-(i+2)]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                                  align_corners=False)
            x = x + skip
            x = blk(x)
        return self.final(x)


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, dims=(48, 96, 192, 384),
                 num_heads_list=(3, 6, 12, 24), depths=(2, 2, 4, 2),
                 win_size=8):
        super().__init__()
        self.encoder = SwinEncoder(in_ch, dims, num_heads_list, depths, win_size)
        self.decoder = Decoder(dims, out_ch)

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)


# ──────────────── Spectral Response Function ────────────────

def make_srf_3gauss(nband, center=None, sigma=None, condition=4.51):
    if center is None:
        center = torch.linspace(0.2, 0.8, nband)
    if sigma is None:
        sigma = torch.full((nband,), 0.07)
    srf = torch.exp(-0.5 * ((center.unsqueeze(1) - center.unsqueeze(0))
                             / sigma.unsqueeze(1))**2)
    srf = srf / srf.sum(dim=0, keepdim=True)
    return srf * condition


# ──────────────── Null-Space Fusion ────────────────

class NullSpaceFusion(nn.Module):
    def __init__(self, msi_ch, hsi_ch, condition=4.51):
        super().__init__()
        self.srf = make_srf_3gauss(hsi_ch)
        self.pseudo_init = nn.Conv2d(msi_ch, hsi_ch, 1, bias=False)

        H = self.srf @ self.srf.T
        I = torch.eye(hsi_ch)
        self.register_buffer('H_inv', torch.linalg.inv(H + 0.01 * I))
        self.register_buffer('Ps', (I - self.srf.T @ self.H_inv @ self.srf))

    def forward(self, hsi_lr, msi_hr):
        b, _, h, w = msi_hr.shape
        pseudo = self.pseudo_init(msi_hr)
        scale = h / hsi_lr.shape[2]
        hsi_up = F.interpolate(hsi_lr, size=(h, w), mode='bilinear',
                               align_corners=False)

        residual_null = pseudo @ self.Ps
        correction = F.interpolate(
            hsi_up - F.interpolate(
                (residual_null @ self.Ps.transpose(0,1)),
                size=hsi_lr.shape[2:], mode='bilinear', align_corners=False),
            size=(h, w), mode='bilinear', align_corners=False)
        return hsi_up + correction


# ──────────────── Diffusion Null-Fusion ────────────────

class DiffusionNullFusion(nn.Module):
    def __init__(self, msi_ch, hsi_ch, condition=4.51,
                 dims=(48, 96, 192, 384), win_size=8):
        super().__init__()
        self.fusion = NullSpaceFusion(msi_ch, hsi_ch, condition)
        in_ch = msi_ch + hsi_ch + 64

        self.sensor_embed = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(hsi_ch, 64), nn.SiLU(), nn.Linear(64, 64))
        self.cond_to_ch = nn.Conv2d(64, in_ch, 1)

        self.unet = UNet(in_ch, hsi_ch, dims=dims,
                         num_heads_list=(dims[0]//16, dims[1]//16,
                                         dims[2]//16, dims[3]//16),
                         depths=(2, 2, 4, 2), win_size=win_size)

        T = 1000
        betas = torch.linspace(1e-4, 0.02, T)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.T = T
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('sqrt_alpha_bar', torch.sqrt(alpha_bar))
        self.register_buffer('sqrt_one_minus_alpha_bar',
                             torch.sqrt(1.0 - alpha_bar))

    def sensor_condition(self, hsi_lr):
        return self.sensor_embed(hsi_lr).unsqueeze(-1).unsqueeze(-1)

    def extract(self, t, shape):
        out = self.sqrt_alpha_bar[t]
        while out.dim() < len(shape):
            out = out.unsqueeze(-1)
        return out

    def q_sample(self, x0, t):
        noise = torch.randn_like(x0)
        a = self.extract(t, x0.shape)
        return a * x0 + (1 - a).sqrt() * noise, noise

    def forward(self, hsi_lr, msi_hr, sigma_range=(0.0, 0.05)):
        init_fused = self.fusion(hsi_lr, msi_hr)
        B, _, H, W = init_fused.shape

        s = self.sensor_condition(hsi_lr)
        s = F.interpolate(s, size=(H, W), mode='bilinear', align_corners=False)
        cond = self.cond_to_ch(torch.cat([init_fused, msi_hr, s], dim=1))

        t = (sigma_range[0] + (sigma_range[1] - sigma_range[0])
             * torch.rand(B, device=init_fused.device)).long()
        noisy, _ = self.q_sample(init_fused, t)
        pred = self.unet(torch.cat([noisy, cond], dim=1))

        with torch.no_grad():
            identity = self.unet(torch.cat([init_fused, cond], dim=1))

        return init_fused + (pred - identity)


# ──────────────── Pixel-Shuffle Upsampler ────────────────

class PixelShuffleUpsampler(nn.Module):
    def __init__(self, in_ch, out_ch, scale):
        super().__init__()
        layers = []
        for _ in range(int(torch.log2(torch.tensor(scale, dtype=torch.float)).item())):
            layers += [
                nn.Conv2d(in_ch, in_ch * 4, 3, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.PixelShuffle(2)]
        layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ──────────────── Complete Model ────────────────

class NullFusionModel(nn.Module):
    def __init__(self, msi_ch=4, hsi_ch=128, scale=4, condition=4.51):
        super().__init__()
        self.upsampler = PixelShuffleUpsampler(msi_ch, msi_ch, scale)
        self.diffusion_fusion = DiffusionNullFusion(
            msi_ch, hsi_ch, condition=condition,
            dims=(48, 96, 192, 384), win_size=8)
        self.refine = nn.Sequential(
            ResBlock(hsi_ch), ResBlock(hsi_ch),
            nn.Conv2d(hsi_ch, hsi_ch, 3, padding=1, bias=False))

    def forward(self, hsi_lr, msi_hr):
        msi_up = self.upsampler(msi_hr)
        fused = self.diffusion_fusion(hsi_lr, msi_up)
        return fused + self.refine(fused)


# ──────────────── Losses ────────────────

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def sam_loss(self, pred, target):
        pred_flat = pred.flatten(2)
        target_flat = target.flatten(2)
        dot = (pred_flat * target_flat).sum(dim=1)
        norm_p = pred_flat.norm(dim=1).clamp(min=self.eps)
        norm_t = target_flat.norm(dim=1).clamp(min=self.eps)
        return (torch.acos((dot / (norm_p * norm_t)).clamp(-1+1e-7, 1-1e-7))).mean()

    def forward(self, pred, target):
        mse = F.mse_loss(pred, target)
        sam = self.sam_loss(pred, target)
        return mse + self.alpha * sam, mse, sam
