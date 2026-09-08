"""Generate self-contained DACF Kaggle notebook for PaviaU + Chikusei."""

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "proposal9", "notebooks", "DACF_Kaggle_PaviaU_Chikusei.ipynb")


def _lines(text):
    return text.splitlines(keepends=True)

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(source.strip())}

def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": _lines(source.strip("\n"))}

def build():
    cells = []

    # Title
    cells.append(md("""
# DACF: Degradation-Adaptive Conditional Flow for HSI-MSI Fusion

Self-contained notebook. Datasets: **PaviaU** (103 bands) + **Chikusei** (128 bands).

## BEFORE RUNNING: Select GPU T4 x2
1. Click **Notebook menu** (top-right `...`)
2. Go to **Settings** -> **Accelerator**
3. Select **GPU T4 x2**
4. Then click **Run All**
""".strip()))

    # Environment
    cells.append(md("## 1. Environment"))
    cells.append(code("""
import os, sys, json, time, math, warnings, glob
warnings.filterwarnings('ignore')

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
print(f'python {sys.version.split()[0]}  torch {torch.__version__}  numpy {np.__version__}')

DEVICE = 'cpu'
GPU_OK = False
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    arch = f'sm_{p.major}{p.minor}'
    built = list(torch.cuda.get_arch_list())
    print(f'gpu     {p.name} {p.total_memory/2**30:.1f} GB {arch}')
    GPU_OK = arch in built
    if GPU_OK:
        DEVICE = 'cuda'
        torch.backends.cudnn.benchmark = True
        print(f'amp     fp16 with tensor cores')
    else:
        print(f'\\n*** WARNING: {p.name} ({arch}) NOT supported by this PyTorch build.')
        print(f'*** FIX: Notebook menu -> Settings -> Accelerator -> "GPU T4 x2"')

if DEVICE != 'cuda':
    raise SystemExit(
        '\\n*** ABORT: No compatible GPU detected. ***\\n'
        '*** Go to Notebook menu -> Settings -> Accelerator -> GPU T4 x2 ***'
    )

WORK = '/kaggle/working' if os.path.isdir('/kaggle/working') else '.'
os.chdir(WORK)
print(f'workdir {os.getcwd()}  device={DEVICE}')
!pip install -q h5py
print("GPU ready")
"""))

    # Data loading
    cells.append(md("## 2. Data loading"))
    cells.append(code('''
def load_mat(path):
    try:
        import scipy.io as sio
        data = sio.loadmat(path)
        for k in data:
            if not k.startswith("_"):
                arr = data[k]
                if isinstance(arr, np.ndarray) and arr.ndim == 3:
                    return arr.astype(np.float32)
    except Exception:
        pass
    import h5py
    with h5py.File(path, "r") as f:
        for k in f:
            if not k.startswith("_"):
                arr = np.array(f[k])
                if arr.ndim == 3:
                    return arr.astype(np.float32)
                if hasattr(arr, "dtype") and arr.dtype.names:
                    for name in arr.dtype.names:
                        sub = np.array(arr[name])
                        if sub.ndim == 3:
                            return sub.astype(np.float32)
    raise ValueError(f"Cannot load HSI from {path}")

def load_hsi_from_path(path):
    mat_files = glob.glob(os.path.join(path, "**", "*.mat"), recursive=True)
    hsi_list = []
    for f in mat_files:
        try:
            arr = load_mat(f)
        except Exception as e:
            print(f"  Skipping {os.path.basename(f)}: {e}")
            continue
        if arr.ndim == 3 and min(arr.shape) > 10:
            if arr.shape[-1] < arr.shape[0] and arr.shape[-1] < arr.shape[1]:
                arr = np.transpose(arr, (2, 0, 1))
            mx = float(arr.max())
            if mx > 1:
                arr = arr / mx
            hsi_list.append(arr.astype(np.float32))
            print(f"  Loaded {os.path.basename(f)}: {arr.shape}")
    return hsi_list

def make_srf(bands, msi_bands=3):
    centers = np.linspace(0, bands - 1, msi_bands)
    srf = np.zeros((msi_bands, bands), dtype=np.float32)
    sigma = bands / (2 * msi_bands)
    for i, c in enumerate(centers):
        srf[i] = np.exp(-0.5 * ((np.arange(bands) - c) / sigma) ** 2)
    srf = srf / srf.sum(axis=1, keepdims=True)
    return torch.from_numpy(srf)

def gaussian_kernel2d(k, sx, sy):
    ax = torch.arange(k, dtype=torch.float32) - k // 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 / (2 * sx**2) + yy**2 / (2 * sy**2)))
    return kernel / kernel.sum()

def blur_downsample(x, kernel, scale):
    B, C, H, W = x.shape
    k = kernel.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1).to(x.device, x.dtype)
    pad = kernel.shape[0] // 2
    blurred = F.conv2d(x.reshape(B * C, 1, H, W), k.unsqueeze(1), padding=pad, groups=C)
    return blurred.reshape(B, C, H, W)[:, :, ::scale, ::scale]

def all_metrics(pred, target):
    p = pred.detach().cpu().numpy().astype(np.float64)
    t = target.detach().cpu().numpy().astype(np.float64)
    mse = np.mean((p - t) ** 2)
    psnr = 10 * np.log10(1.0 / max(mse, 1e-10))
    mu_p, mu_t = p.mean(), t.mean()
    sig_p, sig_t = p.std(), t.std()
    sig_pt = np.mean((p - mu_p) * (t - mu_t))
    c1, c2 = (0.01) ** 2, (0.03) ** 2
    ssim = ((2*mu_p*mu_t+c1)*(2*sig_pt+c2)) / ((mu_p**2+mu_t**2+c1)*(sig_p**2+sig_t**2+c2))
    pp = p.reshape(p.shape[0], -1)
    tt = t.reshape(t.shape[0], -1)
    cos = np.sum(pp*tt, axis=0) / (np.linalg.norm(pp, axis=0)*np.linalg.norm(tt, axis=0)+1e-10)
    sam = np.mean(np.arccos(np.clip(cos, -1+1e-7, 1-1e-7))) * 180 / np.pi
    rmse = np.sqrt(np.mean((p - t) ** 2, axis=(1, 2)))
    ergas = 100 * np.sqrt(np.mean((rmse / (np.mean(t, axis=(1, 2)) + 1e-10)) ** 2))
    return {"PSNR": psnr, "SSIM": ssim, "SAM": sam, "ERGAS": ergas}

print("Utilities defined")
'''))

    # Model
    cells.append(md("## 3. DACF Model"))
    cells.append(code('''
class FiLMBlock(nn.Module):
    def __init__(self, in_ch, code_dim):
        super().__init__()
        self.gamma = nn.Linear(code_dim, in_ch)
        self.beta = nn.Linear(code_dim, in_ch)
        nn.init.zeros_(self.gamma.weight); nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight); nn.init.zeros_(self.beta.bias)
    def forward(self, x, code):
        g = self.gamma(code).unsqueeze(-1).unsqueeze(-1)
        b = self.beta(code).unsqueeze(-1).unsqueeze(-1)
        return x * (1 + g) + b

class CouplingLayer(nn.Module):
    def __init__(self, channels, code_dim, hidden=64):
        super().__init__()
        self.split = channels // 2
        self.rest = channels - self.split
        self.net = nn.Sequential(
            nn.Conv2d(self.split, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, self.rest * 2, 3, padding=1))
        self.film = FiLMBlock(hidden, code_dim)
    def forward(self, x, code, reverse=False):
        B, C, H, W = x.shape
        x1, x2 = x[:, :self.split], x[:, self.split:]
        h = self.net[:2](x1)
        h = self.film(h, code)
        h = self.net[2:](h)
        log_scale, shift = h[:, :self.rest], h[:, self.rest:]
        if not reverse:
            x2 = x2 * torch.exp(log_scale.tanh()) + shift
            log_det = log_scale.tanh().flatten(1).sum(1)
        else:
            x2 = (x2 - shift) * torch.exp(-log_scale.tanh())
            log_det = torch.zeros(B, device=x.device)
        return torch.cat([x1, x2], dim=1), log_det

class ConditionalFlow(nn.Module):
    def __init__(self, bands, code_dim=64, hidden=64, n_layers=8):
        super().__init__()
        self.layers = nn.ModuleList(
            [CouplingLayer(bands, code_dim, hidden) for _ in range(n_layers)])
    def forward(self, x, code, reverse=False):
        ld = torch.zeros(x.shape[0], device=x.device)
        if not reverse:
            for layer in self.layers:
                x, l = layer(x, code, False); ld = ld + l
        else:
            for layer in reversed(self.layers):
                x, l = layer(x, code, True); ld = ld + l
        return x, ld
    def sample(self, z, code, steps=4):
        y = z.clone()
        for k in range(steps):
            t = (k + 0.5) / steps
            y_t = (1 - t) * z + t * y
            y_next, _ = self.forward(y_t, code, False)
            y = (1 - t) * y + t * y_next
        return y

class DegradationEncoder(nn.Module):
    def __init__(self, bands, msi_bands=3, code_dim=64, hidden=32):
        super().__init__()
        self.enc_lr = nn.Sequential(
            nn.Conv2d(bands, hidden, 3, padding=1), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(8),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.enc_ms = nn.Sequential(
            nn.Conv2d(msi_bands, 16, 3, padding=1), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(hidden + 16, 32), nn.ReLU(), nn.Linear(32, code_dim))
    def forward(self, lr_hsi, msi):
        return self.head(torch.cat([self.enc_lr(lr_hsi), self.enc_ms(msi)], 1))

class NullProjector(nn.Module):
    def __init__(self, scale=4, kernel_size=9, sigma=1.2):
        super().__init__()
        self.scale = scale
        self.ks = kernel_size
        self.sigma = sigma
    def _kernel(self, device, dtype):
        ax = torch.arange(self.ks, device=device, dtype=dtype) - self.ks // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        k = torch.exp(-(xx**2/(2*self.sigma**2) + yy**2/(2*self.sigma**2)))
        return k / k.sum()
    def D(self, x):
        B, C, H, W = x.shape
        k = self._kernel(x.device, x.dtype).unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)
        pad = self.ks // 2
        blurred = F.conv2d(x.reshape(B*C, 1, H, W), k.unsqueeze(1), padding=pad, groups=C)
        return blurred.reshape(B, C, H, W)[:, :, ::self.scale, ::self.scale]
    def project(self, v):
        B, C, H, W = v.shape
        dv = self.D(v)
        k = self._kernel(v.device, v.dtype).unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)
        pad = self.ks // 2
        up = F.interpolate(dv, size=(H, W), mode="nearest")
        corr = F.conv2d(up.reshape(B*C, 1, H, W), k.unsqueeze(1), padding=pad, groups=C)
        return v - corr.reshape(B, C, H, W)

class DACFNet(nn.Module):
    def __init__(self, bands, msi_bands=3, scale=4, enc_hidden=32, code_dim=64,
                 flow_hidden=64, flow_layers=8, flow_steps=4):
        super().__init__()
        self.bands = bands; self.scale = scale; self.flow_steps = flow_steps
        self.encoder = DegradationEncoder(bands, msi_bands, code_dim, enc_hidden)
        self.flow = ConditionalFlow(bands, code_dim, flow_hidden, flow_layers)
        self.projector = NullProjector(scale)
        self.srf = None
    def set_srf(self, srf): self.srf = srf
    def sample(self, lr_hsi, msi, steps=None):
        if steps is None: steps = self.flow_steps
        code = self.encoder(lr_hsi, msi)
        hr = self.flow.sample(lr_hsi, code, steps=steps)
        hr = self.projector.project(hr - lr_hsi) + lr_hsi
        return hr
    def forward(self, lr_hsi, msi):
        return {"out": self.sample(lr_hsi, msi)}

print("DACF model defined")
'''))

    # Dataset
    cells.append(md("## 4. Dataset"))
    cells.append(code('''
class DACFDataset(torch.utils.data.Dataset):
    def __init__(self, hsi_list, bands, msi_bands, srf, scale=4, patch=64,
                 train=True, length=8000, sigma_range=(0.5, 3.0), noise_range=(0.0, 0.05)):
        self.hsi_list = hsi_list; self.bands = bands; self.scale = scale
        self.patch = patch; self.train = train
        self.length = length if train else len(hsi_list)
        self.srf = srf; self.sigma_range = sigma_range; self.noise_range = noise_range
    def __len__(self): return self.length
    def __getitem__(self, idx):
        if self.train:
            si = np.random.randint(len(self.hsi_list))
            hsi = self.hsi_list[si]; _, h, w = hsi.shape
            top = np.random.randint(0, h - self.patch + 1)
            left = np.random.randint(0, w - self.patch + 1)
            gt = torch.from_numpy(hsi[:, top:top+self.patch, left:left+self.patch])
            if np.random.random() < 0.5: gt = torch.flip(gt, [-1])
            if np.random.random() < 0.5: gt = torch.flip(gt, [-2])
            k = np.random.randint(4)
            if k: gt = torch.rot90(gt, k, (-2, -1))
        else:
            si = idx % len(self.hsi_list); hsi = self.hsi_list[si]
            _, h, w = hsi.shape; p = (min(h, w) // self.scale) * self.scale
            gt = torch.from_numpy(hsi[:, :p, :p])
        sx = np.random.uniform(*self.sigma_range)
        sy = sx if np.random.random() > 0.5 else np.random.uniform(*self.sigma_range)
        kernel = gaussian_kernel2d(9, sx, sy)
        lr = blur_downsample(gt, kernel, self.scale)
        noise = np.random.uniform(*self.noise_range) if self.train else 0.0
        if noise > 0: lr = (lr + torch.randn_like(lr) * noise).clamp(0, 1)
        msi = torch.einsum("mb,bhw->mhw", self.srf, gt).clamp(0, 1)
        return {"lr": lr, "msi": msi, "gt": gt, "kernel": kernel}

print("Dataset defined")
'''))

    # Download
    cells.append(md("## 5. Download datasets"))
    cells.append(code('''
import kagglehub

print("Downloading PaviaU...")
paviau_path = kagglehub.dataset_download("syamkakarla/pavia-university-hsi")
print(f"PaviaU: {paviau_path}")

print("\\nDownloading Chikusei...")
chikusei_path = kagglehub.dataset_download("mingliu123/chikusei")
print(f"Chikusei: {chikusei_path}")

# Show structure
for name, p in [("PaviaU", paviau_path), ("Chikusei", chikusei_path)]:
    for root, dirs, files in os.walk(p):
        depth = root.replace(p, "").count(os.sep)
        if depth <= 2:
            indent = "  " * depth
            print(f"{indent}{os.path.basename(root)}/")
            for f in sorted(files)[:5]:
                print(f"{indent}  {f}")
            if len(files) > 5:
                print(f"{indent}  ... ({len(files)} files)")
'''))

    # Load data
    cells.append(md("## 6. Load datasets"))
    cells.append(code('''
print("Loading PaviaU...")
paviau_hsi = load_hsi_from_path(paviau_path)
print(f"  Total: {len(paviau_hsi)} scenes, {paviau_hsi[0].shape[0]} bands")

print("\\nLoading Chikusei...")
chikusei_hsi = load_hsi_from_path(chikusei_path)
print(f"  Total: {len(chikusei_hsi)} scenes, {chikusei_hsi[0].shape[0]} bands")
'''))

    # Training function
    cells.append(md("## 7. Training function"))
    cells.append(code('''
def train_dacf(model, dataset, cfg, device, name="dacf"):
    srf = dataset.srf.to(device)
    dl = torch.utils.data.DataLoader(dataset, batch_size=cfg["batch"], shuffle=True,
                                      num_workers=0, pin_memory=(device=="cuda"))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    scaler = torch.amp.GradScaler(enabled=(device=="cuda"))
    os.makedirs(f"{name}_out", exist_ok=True)
    best_psnr = -1
    history = {"iter": [], "loss": [], "psnr": [], "ssim": [], "sam": [], "ergas": []}
    t0 = time.time(); model.train(); it = iter(dl)

    for step in range(cfg["iters"]):
        try: batch = next(it)
        except StopIteration: it = iter(dl); batch = next(it)
        lr_t = batch["lr"].to(device, non_blocking=True)
        msi = batch["msi"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)
        t = torch.rand(lr_t.shape[0], device=device)

        warmup = cfg.get("warmup", 500)
        if step < warmup: lr_now = cfg["lr"] * (step+1) / warmup
        else:
            frac = (step - warmup) / max(1, cfg["iters"] - warmup)
            lr_now = 1e-6 + 0.5*(cfg["lr"]-1e-6)*(1+math.cos(math.pi*frac))
        for p in opt.param_groups: p["lr"] = lr_now

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device, enabled=(device=="cuda")):
            code = model.encoder(lr_t, msi)
            y_t = (1-t.view(-1,1,1,1))*lr_t + t.view(-1,1,1,1)*gt
            pred_v, _ = model.flow.forward(y_t, code, reverse=False)
            target_v = gt - lr_t
            loss_flow = F.mse_loss(pred_v, target_v)
            with torch.no_grad():
                fused_2 = model.flow.sample(lr_t, code, steps=2)
                fused_2 = model.projector.project(fused_2 - lr_t) + lr_t
            d_fused = model.projector.D(fused_2)
            loss_cycle = torch.mean(torch.sqrt((d_fused - lr_t)**2 + 1e-6))
            s_fused = torch.einsum("mb,bhw->mhw", srf, fused_2)
            loss_srf = torch.mean(torch.sqrt((s_fused - msi)**2 + 1e-6))
            loss = loss_flow + 0.5*loss_cycle + 0.3*loss_srf
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()

        if step % cfg.get("log_every", 100) == 0:
            elapsed = time.time() - t0
            rate = (step+1) / max(elapsed, 1e-6)
            eta = (cfg["iters"]-step-1) / max(rate, 1e-6) / 60
            history["iter"].append(step); history["loss"].append(loss.item())
            print(f"[{step:6d}/{cfg['iters']}] loss={loss.item():.4f} "
                  f"lr={lr_now:.2e} {rate:.2f}it/s eta={eta:.0f}m")

        if (step+1) % cfg.get("val_every", 2000) == 0 or step == cfg["iters"]-1:
            model.eval()
            psnrs, ssims, sams, ergases = [], [], [], []
            n_val = min(len(dataset), cfg.get("val_scenes", 4))
            for vi in range(n_val):
                b = DACFDataset(dataset.hsi_list, dataset.bands, 3, dataset.srf, train=False)
                bt = b[vi]
                lr_v = bt["lr"].unsqueeze(0).to(device)
                msi_v = bt["msi"].unsqueeze(0).to(device)
                gt_v = bt["gt"].unsqueeze(0).to(device)
                with torch.no_grad(): out = model(lr_v, msi_v)["out"]
                m = all_metrics(out[0], gt_v[0])
                psnrs.append(m["PSNR"]); ssims.append(m["SSIM"])
                sams.append(m["SAM"]); ergases.append(m["ERGAS"])
            mean_p = np.mean(psnrs)
            print(f"  [val] PSNR={mean_p:.3f} SSIM={np.mean(ssims):.4f} "
                  f"SAM={np.mean(sams):.3f} ERGAS={np.mean(ergases):.3f}")
            history["psnr"].append(mean_p)
            history["ssim"].append(np.mean(ssims))
            history["sam"].append(np.mean(sams))
            history["ergas"].append(np.mean(ergases))
            if mean_p > best_psnr:
                best_psnr = mean_p
                torch.save({"state": model.state_dict(), "best_psnr": best_psnr},
                           f"{name}_out/best.pth")
                print(f"  [save] best.pth (PSNR={best_psnr:.3f})")
            model.train()
            if device == "cuda": torch.cuda.empty_cache()

    torch.save({"state": model.state_dict(), "best_psnr": best_psnr}, f"{name}_out/final.pth")
    history["best_psnr"] = best_psnr
    print(f"[done] best PSNR={best_psnr:.3f}")
    return history

print("Training function defined")
'''))

    # Phase 2 adaptation
    cells.append(md("## 8. Self-supervised adaptation (Phase 2)"))
    cells.append(code('''
def adapt(model, lr_hsi, msi, srf, device, iters=100, lr=1e-3):
    """Adapt degradation encoder only (flow frozen)."""
    opt = torch.optim.Adam(model.encoder.parameters(), lr=lr)
    model.train(); model.flow.eval(); model.projector.eval()
    for step in range(iters):
        opt.zero_grad(set_to_none=True)
        code = model.encoder(lr_hsi, msi)
        fused = model.flow.sample(lr_hsi, code, steps=model.flow_steps)
        fused = model.projector.project(fused - lr_hsi) + lr_hsi
        d_fused = model.projector.D(fused)
        loss_cycle = torch.mean(torch.sqrt((d_fused - lr_hsi)**2 + 1e-6))
        s_fused = torch.einsum("mb,bhw->mhw", srf, fused)
        loss_srf = torch.mean(torch.sqrt((s_fused - msi)**2 + 1e-6))
        loss = loss_cycle + 0.5 * loss_srf
        loss.backward(); opt.step()
    model.eval()
    if device == "cuda": torch.cuda.empty_cache()

print("Adaptation function defined")
'''))

    # Train PaviaU
    cells.append(md("## 9. Train on PaviaU (103 bands, x4)"))
    cells.append(code('''
bands_pv = paviau_hsi[0].shape[0]
cfg_pv = {"batch": 16, "iters": 20000, "lr": 2e-4, "warmup": 500,
          "val_every": 2000, "log_every": 100, "val_scenes": 4}

srf_pv = make_srf(bands_pv, 3)
pv_model = DACFNet(bands_pv, 3, scale=4, enc_hidden=32, code_dim=64,
                    flow_hidden=64, flow_layers=8, flow_steps=4).to(DEVICE)
pv_model.set_srf(srf_pv)
pv_dataset = DACFDataset(paviau_hsi, bands_pv, 3, srf_pv, train=True,
                          length=cfg_pv["iters"]*cfg_pv["batch"])

n_params = sum(p.numel() for p in pv_model.parameters())
print(f"PaviaU: {bands_pv} bands, {cfg_pv['iters']} iters, batch={cfg_pv['batch']}")
print(f"Model: {n_params/1e6:.2f} M params")

t0 = time.time()
pv_hist = train_dacf(pv_model, pv_dataset, cfg_pv, DEVICE, name="paviau")
print(f"\\nTraining time: {(time.time()-t0)/60:.1f} min")
'''))

    # Plot PaviaU
    cells.append(code('''
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
if pv_hist["iter"]:
    ax[0].plot(pv_hist["iter"], pv_hist["loss"])
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel("loss"); ax[0].grid(alpha=.3)
    ax[0].set_title("PaviaU training loss")
if pv_hist["psnr"]:
    vals = list(range(len(pv_hist["psnr"])))
    ax[1].plot(vals, pv_hist["psnr"], marker="o")
    ax[1].set_xlabel("validation step"); ax[1].set_ylabel("PSNR (dB)"); ax[1].grid(alpha=.3)
    ax[1].set_title(f"PaviaU validation (best={pv_hist.get('best_psnr',0):.2f})")
plt.tight_layout(); plt.savefig("paviau_curves.png", dpi=140); plt.show()
'''))

    # Evaluate PaviaU
    cells.append(md("## 10. Evaluate PaviaU"))
    cells.append(code('''
print("=== PaviaU Test (with Phase 2 adaptation) ===")
pv_model.eval()
psnrs, ssims, sams, ergases = [], [], [], []
srf_pv_t = srf_pv.to(DEVICE)
for i, gt_np in enumerate(paviau_hsi):
    p = 256
    gt_t = torch.from_numpy(gt_np[:, :p, :p]).unsqueeze(0).to(DEVICE)
    lr_t = blur_downsample(gt_t, gaussian_kernel2d(9, 1.2, 1.2), 4)
    msi_t = torch.einsum("mb,bhw->mhw", srf_pv_t, gt_t[0]).clamp(0, 1).unsqueeze(0)
    torch.cuda.empty_cache()
    adapt(pv_model, lr_t, msi_t, srf_pv_t, DEVICE, iters=100, lr=1e-3)
    with torch.no_grad(): out = pv_model(lr_t, msi_t)["out"]
    m = all_metrics(out[0], gt_t[0])
    psnrs.append(m["PSNR"]); ssims.append(m["SSIM"])
    sams.append(m["SAM"]); ergases.append(m["ERGAS"])
    print(f"  scene {i:2d}: PSNR={m['PSNR']:7.3f} SSIM={m['SSIM']:.4f} "
          f"SAM={m['SAM']:6.3f} ERGAS={m['ERGAS']:8.3f}")
    del gt_t, lr_t, msi_t, out; torch.cuda.empty_cache()
pv_mean = {"PSNR": np.mean(psnrs), "SSIM": np.mean(ssims),
           "SAM": np.mean(sams), "ERGAS": np.mean(ergases)}
print(f"  MEAN:     PSNR={pv_mean['PSNR']:7.3f} SSIM={pv_mean['SSIM']:.4f} "
      f"SAM={pv_mean['SAM']:6.3f} ERGAS={pv_mean['ERGAS']:8.3f}")
del pv_model; torch.cuda.empty_cache()
'''))

    # Train Chikusei
    cells.append(md("## 11. Train on Chikusei (128 bands, x4)"))
    cells.append(code('''
bands_ch = chikusei_hsi[0].shape[0]
cfg_ch = {"batch": 8, "iters": 20000, "lr": 2e-4, "warmup": 500,
          "val_every": 2000, "log_every": 100, "val_scenes": 4}

srf_ch = make_srf(bands_ch, 3)
ch_model = DACFNet(bands_ch, 3, scale=4, enc_hidden=32, code_dim=64,
                    flow_hidden=64, flow_layers=8, flow_steps=4).to(DEVICE)
ch_model.set_srf(srf_ch)
ch_dataset = DACFDataset(chikusei_hsi, bands_ch, 3, srf_ch, train=True,
                          length=cfg_ch["iters"]*cfg_ch["batch"])

n_params = sum(p.numel() for p in ch_model.parameters())
print(f"Chikusei: {bands_ch} bands, {cfg_ch['iters']} iters, batch={cfg_ch['batch']}")
print(f"Model: {n_params/1e6:.2f} M params")

t0 = time.time()
ch_hist = train_dacf(ch_model, ch_dataset, cfg_ch, DEVICE, name="chikusei")
print(f"\\nTraining time: {(time.time()-t0)/60:.1f} min")
'''))

    # Plot Chikusei
    cells.append(code('''
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
if ch_hist["iter"]:
    ax[0].plot(ch_hist["iter"], ch_hist["loss"])
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel("loss"); ax[0].grid(alpha=.3)
    ax[0].set_title("Chikusei training loss")
if ch_hist["psnr"]:
    vals = list(range(len(ch_hist["psnr"])))
    ax[1].plot(vals, ch_hist["psnr"], marker="o")
    ax[1].set_xlabel("validation step"); ax[1].set_ylabel("PSNR (dB)"); ax[1].grid(alpha=.3)
    ax[1].set_title(f"Chikusei validation (best={ch_hist.get('best_psnr',0):.2f})")
plt.tight_layout(); plt.savefig("chikusei_curves.png", dpi=140); plt.show()
'''))

    # Evaluate Chikusei
    cells.append(md("## 12. Evaluate Chikusei"))
    cells.append(code('''
print("=== Chikusei Test (with Phase 2 adaptation) ===")
ch_model.eval()
psnrs, ssims, sams, ergases = [], [], [], []
srf_ch_t = srf_ch.to(DEVICE)
for i, gt_np in enumerate(chikusei_hsi):
    p = 256
    gt_t = torch.from_numpy(gt_np[:, :p, :p]).unsqueeze(0).to(DEVICE)
    lr_t = blur_downsample(gt_t, gaussian_kernel2d(9, 1.2, 1.2), 4)
    msi_t = torch.einsum("mb,bhw->mhw", srf_ch_t, gt_t[0]).clamp(0, 1).unsqueeze(0)
    torch.cuda.empty_cache()
    adapt(ch_model, lr_t, msi_t, srf_ch_t, DEVICE, iters=100, lr=1e-3)
    with torch.no_grad(): out = ch_model(lr_t, msi_t)["out"]
    m = all_metrics(out[0], gt_t[0])
    psnrs.append(m["PSNR"]); ssims.append(m["SSIM"])
    sams.append(m["SAM"]); ergases.append(m["ERGAS"])
    print(f"  scene {i:2d}: PSNR={m['PSNR']:7.3f} SSIM={m['SSIM']:.4f} "
          f"SAM={m['SAM']:6.3f} ERGAS={m['ERGAS']:8.3f}")
    del gt_t, lr_t, msi_t, out; torch.cuda.empty_cache()
ch_mean = {"PSNR": np.mean(psnrs), "SSIM": np.mean(ssims),
           "SAM": np.mean(sams), "ERGAS": np.mean(ergases)}
print(f"  MEAN:     PSNR={ch_mean['PSNR']:7.3f} SSIM={ch_mean['SSIM']:.4f} "
      f"SAM={ch_mean['SAM']:6.3f} ERGAS={ch_mean['ERGAS']:8.3f}")
del ch_model; torch.cuda.empty_cache()
'''))

    # Comparison
    cells.append(md("## 13. Results Comparison"))
    cells.append(code('''
print("=" * 80)
print("DACF RESULTS vs SOTA (Wald x4 protocol)")
print("=" * 80)
print()
print(f\'{"Method":<30} {"Dataset":<18} {"PSNR":>8} {"SSIM":>8} {"SAM":>8} {"ERGAS":>8}\')
print("-" * 80)

sota = [
    ("Bicubic",                "PaviaU x4",       25.40, 0.713, 15.05, 8.19),
    ("GSA",                    "PaviaU x4",       25.99, 0.743, 14.24, 7.71),
    ("Subspace-LS",            "PaviaU x4",       26.70, 0.784, 12.78, 7.06),
    ("KrylovNet (Ours prev)",  "PaviaU x4",       34.48, 0.952,  4.46, 2.84),
    ("CoFusion",               "PaviaU x4",       38.32, 0.982,  2.56, 1.95),
    ("Bicubic",                "Chikusei x4",     33.58, 0.897, 14.25, 0.00),
    ("KrylovNet (Ours prev)",  "Chikusei x4",     43.69, 0.983,  6.07, 0.00),
]

for name, ds, psnr, ssim, sam, ergas in sota:
    print(f\'{name:<30} {ds:<18} {psnr:>8.2f} {ssim:>8.4f} {sam:>8.2f} {ergas:>8.2f}\')

print("-" * 80)
print(f\'{"DACF (Ours)":<30} {"PaviaU x4":<18} {pv_mean["PSNR"]:>8.2f} '
      f\'{pv_mean["SSIM"]:>8.4f} {pv_mean["SAM"]:>8.2f} {pv_mean["ERGAS"]:>8.2f}\')
print(f\'{"DACF (Ours)":<30} {"Chikusei x4":<18} {ch_mean["PSNR"]:>8.2f} '
      f\'{ch_mean["SSIM"]:>8.4f} {ch_mean["SAM"]:>8.2f} {ch_mean["ERGAS"]:>8.2f}\')
print("=" * 80)
'''))

    # Save
    cells.append(md("## 14. Save results"))
    cells.append(code('''
results = {
    "paviau": pv_mean, "chikusei": ch_mean,
    "paviau_history": pv_hist, "chikusei_history": ch_hist,
}
with open("dacf_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("Saved: dacf_results.json")
for d in ["paviau_out", "chikusei_out"]:
    if os.path.isdir(d):
        print(f"  {d}/: {os.listdir(d)}")
print(f"Figures: {[f for f in os.listdir('.') if f.endswith('.png')]}")
'''))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    nb = build()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    print(f"wrote {OUT}")
    print(f"  {len(nb['cells'])} cells ({n_code} code), "
          f"{os.path.getsize(OUT) / 1024:.0f} KB")
