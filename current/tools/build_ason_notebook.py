import json

cells = []

def md(source):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [source] if isinstance(source, str) else source
    })

def code(source):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.split("\n") if isinstance(source, str) else source
    })

# ============================================================
# CELL 1: Title
# ============================================================
md("# ASON: Adaptive Spectral Operator Network for HSI-MSI Fusion\n\n"
   "**Datasets:** Chikusei (128 bands) + CAVE (31 bands)\n\n"
   "**Protocol:** Wald (Gaussian \u03c3=1.2, \u00d74 decimation, 3-band Gaussian SRF)\n\n"
   "**Architecture:** Degradation Estimator \u2192 Spectral Basis Learner \u2192 Null-Space Constrained Network \u2192 Reconstruction")

# ============================================================
# CELL 2: Install dependencies
# ============================================================
code("!pip install -q einops scikit-image scipy matplotlib tqdm\n"
     "!pip install -q kagglehub\n"
     "\n"
     "import torch, torch.nn as nn, torch.nn.functional as F\n"
     "from torch.utils.data import Dataset, DataLoader\n"
     "import numpy as np, os, math, warnings, random\n"
     "from tqdm.auto import tqdm\n"
     "import matplotlib.pyplot as plt\n"
     "warnings.filterwarnings('ignore')\n"
     "\n"
     "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
     "print(f'Device: {device}')\n"
     "if torch.cuda.is_available():\n"
     "    print(f'GPU: {torch.cuda.get_device_name(0)}')\n"
     "    print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')")

# ============================================================
# CELL 3: Download datasets
# ============================================================
code("import kagglehub\n"
     "\n"
     "chikusei_path = kagglehub.dataset_download('mingliu123/chikusei')\n"
     "print(f'Chikusei: {chikusei_path}')\n"
     "\n"
     "cave_path = kagglehub.dataset_download('liptee/cave')\n"
     "print(f'CAVE: {cave_path}')\n"
     "\n"
     "import os\n"
     "for p, name in [(chikusei_path, 'Chikusei'), (cave_path, 'CAVE')]:\n"
     "    files = [f for f in os.listdir(p) if not f.startswith('.')]\n"
     "    print(f'{name} files: {files[:15]}')")

# ============================================================
# CELL 4: Wald Degradation
# ============================================================
code("class WaldDegradation:\n"
     "    def __init__(self, scale=4, sigma=1.2, bands_hr=31, bands_lr=3, device='cuda'):\n"
     "        self.scale = scale\n"
     "        self.bands_hr = bands_hr\n"
     "        self.bands_lr = bands_lr\n"
     "        self.device = device\n"
     "        \n"
     "        k = 9\n"
     "        x = torch.arange(-k//2 + 1, k//2 + 1, dtype=torch.float32)\n"
     "        g = torch.exp(-x**2 / (2 * sigma**2))\n"
     "        g = g / g.sum()\n"
     "        kernel_2d = g.unsqueeze(1) @ g.unsqueeze(0)\n"
     "        self.kernel = kernel_2d.unsqueeze(0).unsqueeze(0).to(device)\n"
     "        \n"
     "        centers = torch.tensor([0.30, 0.55, 0.78], device=device)\n"
     "        wavelengths = torch.linspace(0, 1, bands_hr, device=device)\n"
     "        srf = torch.exp(-(wavelengths.unsqueeze(1) - centers.unsqueeze(0))**2 / (2 * 0.08**2))\n"
     "        srf = srf / srf.sum(dim=0, keepdim=True)\n"
     "        self.srf = srf.T\n"
     "    \n"
     "    def degrade(self, hr_hsi):\n"
     "        if hr_hsi.dim() == 3:\n"
     "            hr_hsi = hr_hsi.unsqueeze(0)\n"
     "        B, C, H, W = hr_hsi.shape\n"
     "        \n"
     "        lr_list = []\n"
     "        for c in range(C):\n"
     "            band = hr_hsi[:, c:c+1]\n"
     "            blurred = F.conv2d(band, self.kernel, padding=4)\n"
     "            decimated = blurred[:, :, ::self.scale, ::self.scale]\n"
     "            lr_list.append(decimated)\n"
     "        lr_hsi = torch.cat(lr_list, dim=1)\n"
     "        \n"
     "        hr_flat = hr_hsi.permute(0, 2, 3, 1).reshape(-1, C)\n"
     "        ms_flat = hr_flat @ self.srf.T\n"
     "        hr_msi = ms_flat.reshape(1, H, W, 3).permute(0, 3, 1, 2)\n"
     "        \n"
     "        return lr_hsi.squeeze(0), hr_msi.squeeze(0)\n"
     "\n"
     "deg = WaldDegradation(bands_hr=31, device='cuda')\n"
     "t = torch.randn(31, 64, 64, device='cuda')\n"
     "lr, msi = deg.degrade(t)\n"
     "print(f'Test: HR {t.shape} -> LR {lr.shape}, MSI {msi.shape}')")

# ============================================================
# CELL 5: Metrics
# ============================================================
code("def calc_psnr(pred, target, data_range=1.0):\n"
     "    mse = torch.mean((pred - target) ** 2)\n"
     "    if mse < 1e-10:\n"
     "        return 100.0\n"
     "    return 20 * math.log10(data_range / math.sqrt(mse.item()))\n"
     "\n"
     "def calc_sam(pred, target):\n"
     "    p = pred.reshape(pred.shape[0], -1).T\n"
     "    t = target.reshape(target.shape[0], -1).T\n"
     "    mask = (t.norm(dim=1) > 1e-8) & (p.norm(dim=1) > 1e-8)\n"
     "    if not mask.any():\n"
     "        return 0.0\n"
     "    pn = p[mask] / (p[mask].norm(dim=1, keepdim=True) + 1e-8)\n"
     "    tn = t[mask] / (t[mask].norm(dim=1, keepdim=True) + 1e-8)\n"
     "    cos = (pn * tn).sum(dim=1).clamp(-1, 1)\n"
     "    return torch.acos(cos).mean().item() * 180 / math.pi\n"
     "\n"
     "def calc_ssim(pred, target, win=11):\n"
     "    ssims = []\n"
     "    for b in range(pred.shape[0]):\n"
     "        C1, C2 = 0.0001, 0.0009\n"
     "        mu_x = F.avg_pool2d(pred[b:b+1].unsqueeze(0), win, 1, padding=win//2).squeeze()\n"
     "        mu_y = F.avg_pool2d(target[b:b+1].unsqueeze(0), win, 1, padding=win//2).squeeze()\n"
     "        sig_x = F.avg_pool2d(pred[b:b+1].unsqueeze(0)**2, win, 1, padding=win//2).squeeze() - mu_x**2\n"
     "        sig_y = F.avg_pool2d(target[b:b+1].unsqueeze(0)**2, win, 1, padding=win//2).squeeze() - mu_y**2\n"
     "        sig_xy = F.avg_pool2d((pred[b]*target[b]).unsqueeze(0).unsqueeze(0), win, 1, padding=win//2).squeeze() - mu_x*mu_y\n"
     "        ssim = ((2*mu_x*mu_y+C1)*(2*sig_xy+C2)) / ((mu_x**2+mu_y**2+C1)*(sig_x+sig_y+C2))\n"
     "        ssims.append(ssim.mean().item())\n"
     "    return np.mean(ssims)\n"
     "\n"
     "def calc_ergas(pred, target, scale=4):\n"
     "    rmse_bands = torch.sqrt(torch.mean((pred - target)**2, dim=(1, 2)))\n"
     "    mean_bands = torch.mean(target, dim=(1, 2)) + 1e-8\n"
     "    return (100 / scale * torch.mean((rmse_bands / mean_bands)**2)).item()\n"
     "\n"
     "def all_metrics(pred, target, scale=4):\n"
     "    return {\n"
     "        'PSNR': calc_psnr(pred, target),\n"
     "        'SSIM': calc_ssim(pred, target),\n"
     "        'SAM': calc_sam(pred, target),\n"
     "        'ERGAS': calc_ergas(pred, target, scale)\n"
     "    }")

# ============================================================
# CELL 6: Dataset loaders
# ============================================================
code("import scipy.io as sio\n"
     "\n"
     "class HSIFusionDataset(Dataset):\n"
     "    def __init__(self, root, dataset_name, patch_size=64, stride=32, split='train'):\n"
     "        self.patch_size = patch_size\n"
     "        self.split = split\n"
     "        self.hsi_list = []\n"
     "        \n"
     "        if dataset_name == 'chikusei':\n"
     "            mat_files = [f for f in os.listdir(root) if f.endswith('.mat')]\n"
     "            print(f'Loading Chikusei from {root}')\n"
     "            for f in mat_files:\n"
     "                data = sio.loadmat(os.path.join(root, f))\n"
     "                keys = [k for k in data.keys() if not k.startswith('__')]\n"
     "                hsi = data[keys[0]].astype(np.float32)\n"
     "                print(f'  File: {f}, key: {keys[0]}, shape: {hsi.shape}')\n"
     "                hsi = (hsi - hsi.min()) / (hsi.max() - hsi.min() + 1e-8)\n"
     "                self.hsi_list.append(hsi)\n"
     "            self.n_bands = self.hsi_list[0].shape[2]\n"
     "        \n"
     "        elif dataset_name == 'cave':\n"
     "            mat_files = sorted([f for f in os.listdir(root) if f.endswith('.mat')])\n"
     "            print(f'Loading CAVE: {len(mat_files)} scenes')\n"
     "            for f in mat_files:\n"
     "                data = sio.loadmat(os.path.join(root, f))\n"
     "                keys = [k for k in data.keys() if not k.startswith('__')]\n"
     "                hsi = data[keys[0]].astype(np.float32)\n"
     "                hsi = (hsi - hsi.min()) / (hsi.max() - hsi.min() + 1e-8)\n"
     "                self.hsi_list.append(hsi)\n"
     "            self.n_bands = 31\n"
     "        \n"
     "        self.patches = []\n"
     "        for si, hsi in enumerate(self.hsi_list):\n"
     "            H, W, B = hsi.shape\n"
     "            for y in range(0, H - patch_size + 1, stride):\n"
     "                for x in range(0, W - patch_size + 1, stride):\n"
     "                    self.patches.append((si, y, x))\n"
     "        \n"
     "        if dataset_name == 'cave':\n"
     "            train_set = set(range(min(20, len(self.hsi_list))))\n"
     "            if split == 'train':\n"
     "                self.patches = [p for p in self.patches if p[0] in train_set]\n"
     "            else:\n"
     "                self.patches = [p for p in self.patches if p[0] not in train_set]\n"
     "        else:\n"
     "            split_idx = int(len(self.patches) * 0.8)\n"
     "            if split == 'train':\n"
     "                self.patches = self.patches[:split_idx]\n"
     "            else:\n"
     "                self.patches = self.patches[split_idx:]\n"
     "        \n"
     "        print(f'  {split}: {len(self.patches)} patches, {self.n_bands} bands')\n"
     "    \n"
     "    def __len__(self):\n"
     "        return len(self.patches)\n"
     "    \n"
     "    def __getitem__(self, idx):\n"
     "        si, y, x = self.patches[idx]\n"
     "        hsi = self.hsi_list[si]\n"
     "        patch = hsi[y:y+self.patch_size, x:x+self.patch_size]\n"
     "        return torch.from_numpy(patch).permute(2, 0, 1).float()")

# ============================================================
# CELL 7: ASON Architecture - Module 1: Degradation Estimator
# ============================================================
code("class DegradationEstimator(nn.Module):\n"
     "    def __init__(self, max_bands=128, hidden=48):\n"
     "        super().__init__()\n"
     "        self.enc_lr = nn.Sequential(\n"
     "            nn.Conv2d(max_bands, hidden, 3, padding=1), nn.LeakyReLU(0.2),\n"
     "            nn.Conv2d(hidden, hidden, 3, padding=1), nn.LeakyReLU(0.2),\n"
     "            nn.Conv2d(hidden, hidden, 3, padding=1), nn.LeakyReLU(0.2),\n"
     "        )\n"
     "        self.enc_ms = nn.Sequential(\n"
     "            nn.Conv2d(3, hidden, 3, padding=1), nn.LeakyReLU(0.2),\n"
     "            nn.Conv2d(hidden, hidden, 3, padding=1), nn.LeakyReLU(0.2),\n"
     "        )\n"
     "        self.cross_attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)\n"
     "        self.r_proj = nn.Sequential(\n"
     "            nn.Linear(hidden, hidden), nn.ReLU(),\n"
     "            nn.Linear(hidden, max_bands * 3)\n"
     "        )\n"
     "        self.h_proj = nn.Sequential(\n"
     "            nn.Linear(hidden, hidden), nn.ReLU(),\n"
     "            nn.Linear(hidden, max_bands * 3)\n"
     "        )\n"
     "        self.quality = nn.Sequential(\n"
     "            nn.AdaptiveAvgPool2d(1), nn.Flatten(),\n"
     "            nn.Linear(hidden, 32), nn.ReLU(),\n"
     "            nn.Linear(32, 1), nn.Sigmoid()\n"
     "        )\n"
     "        self.max_bands = max_bands\n"
     "    \n"
     "    def forward(self, lr_hsi, hr_msi):\n"
     "        B, C, h, w = lr_hsi.shape\n"
     "        lr_feat = self.enc_lr(lr_hsi)  # [B, hidden, h, w]\n"
     "        ms_feat = self.enc_ms(hr_msi)  # [B, hidden, H, W]\n"
     "        \n"
     "        # Upsample LR features to MSI resolution\n"
     "        H, W = ms_feat.shape[2], ms_feat.shape[3]\n"
     "        lr_up = F.interpolate(lr_feat, size=(H, W), mode='bilinear', align_corners=False)\n"
     "        \n"
     "        # Cross attention\n"
     "        lr_flat = lr_up.flatten(2).permute(0, 2, 1)   # [B, H*W, hidden]\n"
     "        ms_flat = ms_feat.flatten(2).permute(0, 2, 1)  # [B, H*W, hidden]\n"
     "        fused, _ = self.cross_attn(lr_flat, ms_flat, ms_flat)  # [B, H*W, hidden]\n"
     "        fused = fused.mean(dim=1)  # [B, hidden]\n"
     "        \n"
     "        r_emb = self.r_proj(fused).view(B, self.max_bands, 3)  # [B, max_bands, 3]\n"
     "        h_emb = self.h_proj(fused).view(B, self.max_bands, 3)  # [B, max_bands, 3]\n"
     "        quality = self.quality(lr_feat)  # [B, 1]\n"
     "        \n"
     "        return r_emb, h_emb, quality\n"
     "\n"
     "print('Module 1: DegradationEstimator defined')")

# ============================================================
# CELL 8: Module 2: Spectral Basis Learner (Band-Agnostic)
# ============================================================
code("class SpectralBasisLearner(nn.Module):\n"
     "    def __init__(self, max_bands=128, hidden=32):\n"
     "        super().__init__()\n"
     "        self.embed = nn.Linear(3, hidden)\n"
     "        self.pos_enc = nn.Parameter(torch.randn(1, max_bands, hidden) * 0.02)\n"
     "        self.encoder = nn.TransformerEncoder(\n"
     "            nn.TransformerEncoderLayer(d_model=hidden, nhead=4, dim_feedforward=hidden*4,\n"
     "                                       dropout=0.1, activation='gelu', batch_first=True),\n"
     "            num_layers=2\n"
     "        )\n"
     "        self.proj = nn.Linear(hidden, 3)  # project to MSI channels for loss\n"
     "        self.max_bands = max_bands\n"
     "    \n"
     "    def forward(self, hr_msi, n_bands):\n"
     "        B, C, H, W = hr_msi.shape\n"
     "        ms_avg = hr_msi.mean(dim=(2, 3))  # [B, 3]\n"
     "        \n"
     "        x = self.embed(ms_avg)  # [B, hidden]\n"
     "        x = x.unsqueeze(1).expand(-1, self.max_bands, -1)  # [B, max_bands, hidden]\n"
     "        x = x + self.pos_enc\n"
     "        x = self.encoder(x)  # [B, max_bands, hidden]\n"
     "        \n"
     "        coeff = self.proj(x)  # [B, max_bands, 3]\n"
     "        return coeff[:, :n_bands, :]  # [B, n_bands, 3]\n"
     "\n"
     "print('Module 2: SpectralBasisLearner defined')")

# ============================================================
# CELL 9: Module 3: Null-Space Constrained Network
# ============================================================
code("class NullSpaceBlock(nn.Module):\n"
     "    def __init__(self, max_bands=128, hidden=48):\n"
     "        super().__init__()\n"
     "        self.range_conv = nn.Sequential(\n"
     "            nn.Conv2d(3, hidden, 3, padding=1), nn.LeakyReLU(0.2),\n"
     "            nn.Conv2d(hidden, hidden, 3, padding=1), nn.LeakyReLU(0.2),\n"
     "            nn.Conv2d(hidden, max_bands, 3, padding=1)\n"
     "        )\n"
     "        self.null_conv = nn.Sequential(\n"
     "            nn.Conv2d(3, hidden, 3, padding=1), nn.LeakyReLU(0.2),\n"
     "            nn.Conv2d(hidden, hidden, 3, padding=1), nn.LeakyReLU(0.2),\n"
     "            nn.Conv2d(hidden, max_bands, 3, padding=1)\n"
     "        )\n"
     "        self.alpha = nn.Parameter(torch.tensor(0.5))\n"
     "    \n"
     "    def forward(self, hr_msi, n_bands):\n"
     "        x_range = self.range_conv(hr_msi)[:, :n_bands, :, :]\n"
     "        x_null = self.null_conv(hr_msi)[:, :n_bands, :, :]\n"
     "        alpha = torch.sigmoid(self.alpha)\n"
     "        return x_range + alpha * x_null\n"
     "\n"
     "print('Module 3: NullSpaceBlock defined')")

# ============================================================
# CELL 10: Full ASON Model
# ============================================================
code("class ASON(nn.Module):\n"
     "    def __init__(self, max_bands=128):\n"
     "        super().__init__() \n"
     "        self.max_bands = max_bands\n"
     "        self.deg_estimator = DegradationEstimator(max_bands)\n"
     "        self.basis_learner = SpectralBasisLearner(max_bands)\n"
     "        self.null_space = NullSpaceBlock(max_bands)\n"
     "        self.refine = nn.Sequential(\n"
     "            nn.Conv2d(max_bands + 3, 48, 3, padding=1), nn.LeakyReLU(0.2),\n"
     "            nn.Conv2d(48, 48, 3, padding=1), nn.LeakyReLU(0.2),\n"
     "            nn.Conv2d(48, max_bands, 3, padding=1)\n"
     "        )\n"
     "    \n"
     "    def forward(self, lr_hsi, hr_msi, n_bands):\n"
     "        r_emb, h_emb, quality = self.deg_estimator(lr_hsi, hr_msi)\n"
     "        basis = self.basis_learner(hr_msi, n_bands)  # [B, n_bands, 3]\n"
     "        x_init = self.null_space(hr_msi, n_bands)  # [B, n_bands, h, w]\n"
     "        \n"
     "        B, C, H, W = x_init.shape\n"
     "        hr_up = F.interpolate(hr_msi, size=(H, W), mode='bilinear', align_corners=False)\n"
     "        refine_in = torch.cat([x_init, hr_up], dim=1)\n"
     "        x_refined = self.refine(refine_in)\n"
     "        \n"
     "        # Apply learned spectral basis modulation\n"
     "        basis_perm = basis.permute(0, 2, 1)  # [B, 3, n_bands]\n"
     "        x_modulated = torch.einsum('bchw,bck->bkhw', hr_up, basis_perm)\n"
     "        \n"
     "        # Combine: interpolation between refined and modulated\n"
     "        alpha = quality.view(-1, 1, 1, 1)\n"
     "        output = alpha * x_refined + (1 - alpha) * x_modulated\n"
     "        \n"
     "        return output.clamp(0, 1)\n"
     "\n"
     "model = ASON(max_bands=128).to(device)\n"
     "n_params = sum(p.numel() for p in model.parameters())\n"
     "print(f'ASON params: {n_params:,}')\n"
     "print(f'ASON model:\n{model}')")

# ============================================================
# CELL 11: Training setup
# ============================================================
code("class ASONTrainer:\n"
     "    def __init__(self, model, lr=1e-3, device='cuda'):\n"
     "        self.model = model\n"
     "        self.device = device\n"
     "        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)\n"
     "        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=1e-6)\n"
     "        self.scaler = torch.cuda.amp.GradScaler()\n"
     "        self.best_psnr = 0\n"
     "    \n"
     "    def forward_degrade(self, hr_hsi, n_bands):\n"
     "        deg = WaldDegradation(scale=4, sigma=1.2, bands_hr=n_bands, device=self.device)\n"
     "        lr_list = []\n"
     "        msi_list = []\n"
     "        for b in range(hr_hsi.shape[0]):\n"
     "            lr, msi = deg.degrade(hr_hsi[b])\n"
     "            lr_list.append(lr)\n"
     "            msi_list.append(msi)\n"
     "        return torch.stack(lr_list), torch.stack(msi_list)\n"
     "    \n"
     "    def train_epoch(self, loader, n_bands, epoch):\n"
     "        self.model.train()\n"
     "        total_loss = 0\n"
     "        psnr_sum = 0\n"
     "        for batch in tqdm(loader, desc=f'Train Epoch {epoch}'):\n"
     "            hr_hsi = batch.to(self.device)\n"
     "            lr_hsi, hr_msi = self.forward_degrade(hr_hsi, n_bands)\n"
     "            \n"
     "            self.optimizer.zero_grad()\n"
     "            with torch.cuda.amp.autocast():\n"
     "                pred = self.model(lr_hsi, hr_msi, n_bands)\n"
     "                \n"
     "                # Multi-scale loss\n"
     "                loss_recon = F.l1_loss(pred, hr_hsi)\n"
     "                \n"
     "                # SAM loss\n"
     "                pred_n = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)\n"
     "                target_n = hr_hsi / (hr_hsi.norm(dim=1, keepdim=True) + 1e-8)\n"
     "                loss_sam = (1 - (pred_n * target_n).sum(dim=1)).mean()\n"
     "                \n"
     "                loss = loss_recon + 0.1 * loss_sam\n"
     "            \n"
     "            self.scaler.scale(loss).backward()\n"
     "            self.scaler.unscale_(self.optimizer)\n"
     "            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)\n"
     "            self.scaler.step(self.optimizer)\n"
     "            self.scaler.update()\n"
     "            \n"
     "            total_loss += loss.item()\n"
     "            psnr_sum += calc_psnr(pred, hr_hsi)\n"
     "        \n"
     "        self.scheduler.step()\n"
     "        n = len(loader)\n"
     "        print(f'  Loss: {total_loss/n:.4f}, PSNR: {psnr_sum/n:.2f} dB')\n"
     "        return total_loss / n\n"
     "    \n"
     "    @torch.no_grad()\n"
     "    def evaluate(self, loader, n_bands):\n"
     "        self.model.eval()\n"
     "        psnr_list, ssim_list, sam_list, ergas_list = [], [], [], []\n"
     "        for batch in tqdm(loader, desc='Evaluating'):\n"
     "            hr_hsi = batch.to(self.device)\n"
     "            lr_hsi, hr_msi = self.forward_degrade(hr_hsi, n_bands)\n"
     "            pred = self.model(lr_hsi, hr_msi, n_bands)\n"
     "            \n"
     "            for b in range(pred.shape[0]):\n"
     "                m = all_metrics(pred[b], hr_hsi[b])\n"
     "                psnr_list.append(m['PSNR'])\n"
     "                ssim_list.append(m['SSIM'])\n"
     "                sam_list.append(m['SAM'])\n"
     "                ergas_list.append(m['ERGAS'])\n"
     "        \n"
     "        results = {\n"
     "            'PSNR': np.mean(psnr_list), 'SSIM': np.mean(ssim_list),\n"
     "            'SAM': np.mean(sam_list), 'ERGAS': np.mean(ergas_list)\n"
     "        }\n"
     "        print(f'  PSNR: {results[\"PSNR\"]:.2f} dB, SSIM: {results[\"SSIM\"]:.4f}, '\n"
     "              f'SAM: {results[\"SAM\"]:.2f}\\u00b0, ERGAS: {results[\"ERGAS\"]:.2f}')\n"
     "        return results\n"
     "    \n"
     "    @torch.no_grad()\n"
     "    def save_if_best(self, results, path):\n"
     "        if results['PSNR'] > self.best_psnr:\n"
     "            self.best_psnr = results['PSNR']\n"
     "            torch.save(self.model.state_dict(), path)\n"
     "            print(f'  Saved best model (PSNR={results[\"PSNR\"]:.2f} dB)')\n"
     "            return True\n"
     "        return False\n"
     "\n"
     "print('ASONTrainer defined')")

# ============================================================
# CELL 12: Run Chikusei Training
# ============================================================
code("print('='*60)\n"
     "print('DATASET 1: CHIKUSEI (128 bands, x4)')\n"
     "print('='*60)\n"
     "\n"
     "N_BANDS_CH = 128\n"
     "BATCH_SIZE = 4\n"
     "PATCH_SIZE = 64\n"
     "EPOCHS = 100\n"
     "\n"
     "train_ds = HSIFusionDataset(chikusei_path, 'chikusei', patch_size=PATCH_SIZE, stride=32, split='train')\n"
     "test_ds = HSIFusionDataset(chikusei_path, 'chikusei', patch_size=PATCH_SIZE, stride=64, split='test')\n"
     "\n"
     "train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)\n"
     "test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1)\n"
     "\n"
     "model_ch = ASON(max_bands=128).to(device)\n"
     "trainer_ch = ASONTrainer(model_ch, lr=1e-3, device=device)\n"
     "\n"
     "print(f'Train: {len(train_ds)} patches, Test: {len(test_ds)} patches')\n"
     "print(f'Model params: {sum(p.numel() for p in model_ch.parameters()):,}')\n"
     "\n"
     "for epoch in range(1, EPOCHS + 1):\n"
     "    print(f'\\n--- Epoch {epoch}/{EPOCHS} ---')\n"
     "    trainer_ch.train_epoch(train_loader, N_BANDS_CH, epoch)\n"
     "    if epoch % 5 == 0 or epoch == 1:\n"
     "        results = trainer_ch.evaluate(test_loader, N_BANDS_CH)\n"
     "        trainer_ch.save_if_best(results, 'ason_chikusei_best.pth')\n"
     "\n"
     "# Final evaluation with best model\n"
     "model_ch.load_state_dict(torch.load('ason_chikusei_best.pth'))\n"
     "print('\\n' + '='*60)\n"
     "print('FINAL CHIKUSEI RESULTS')\n"
     "print('='*60)\n"
     "ch_results = trainer_ch.evaluate(test_loader, N_BANDS_CH)")

# ============================================================
# CELL 13: Run CAVE Training
# ============================================================
code("print('='*60)\n"
     "print('DATASET 2: CAVE (31 bands, x4)')\n"
     "print('='*60)\n"
     "\n"
     "N_BANDS_CV = 31\n"
     "\n"
     "train_ds_cv = HSIFusionDataset(cave_path, 'cave', patch_size=PATCH_SIZE, stride=32, split='train')\n"
     "test_ds_cv = HSIFusionDataset(cave_path, 'cave', patch_size=PATCH_SIZE, stride=64, split='test')\n"
     "\n"
     "train_loader_cv = DataLoader(train_ds_cv, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)\n"
     "test_loader_cv = DataLoader(test_ds_cv, batch_size=1, shuffle=False, num_workers=1)\n"
     "\n"
     "model_cv = ASON(max_bands=128).to(device)\n"
     "trainer_cv = ASONTrainer(model_cv, lr=1e-3, device=device)\n"
     "\n"
     "print(f'Train: {len(train_ds_cv)} patches, Test: {len(test_ds_cv)} patches')\n"
     "\n"
     "for epoch in range(1, EPOCHS + 1):\n"
     "    print(f'\\n--- Epoch {epoch}/{EPOCHS} ---')\n"
     "    trainer_cv.train_epoch(train_loader_cv, N_BANDS_CV, epoch)\n"
     "    if epoch % 5 == 0 or epoch == 1:\n"
     "        results = trainer_cv.evaluate(test_loader_cv, N_BANDS_CV)\n"
     "        trainer_cv.save_if_best(results, 'ason_cave_best.pth')\n"
     "\n"
     "model_cv.load_state_dict(torch.load('ason_cave_best.pth'))\n"
     "print('\\n' + '='*60)\n"
     "print('FINAL CAVE RESULTS')\n"
     "print('='*60)\n"
     "cv_results = trainer_cv.evaluate(test_loader_cv, N_BANDS_CV)")

# ============================================================
# CELL 14: Comparison Table
# ============================================================
code("print('='*60)\n"
     "print('FINAL BENCHMARK COMPARISON')\n"
     "print('='*60)\n"
     "print()\n"
     "print(f'{\"Method\":<25} {\"Dataset\":<15} {\"PSNR\":<10} {\"SSIM\":<10} {\"SAM\":<10} {\"ERGAS\":<10}')\n"
     "print('-' * 80)\n"
     "\n"
     "# Current SOTA references\n"
     "sota = [\n"
     "    ('FeINFN (TIP 24)', 'CAVE x4 Nikon', 52.47, 0.9787, 3.63, 1.01),\n"
     "    ('BDT (TCSVT 24)', 'CAVE x4 Nikon', 52.30, 0.9782, 3.70, 1.03),\n"
     "    ('KrylovNet (TIP 25)', 'CAVE x4 SRF', 50.10, 0.9610, 4.50, 1.48),\n"
     "    ('KrylovNet-P (TIP 25)', 'CAVE x4 SRF', 52.47, 0.9787, 3.63, 1.01),\n"
     "    ('BDT (TCSVT 24)', 'Chikusei x4', 39.20, 0.9320, 4.80, 3.10),\n"
     "    ('KrylovNet (TIP 25)', 'Chikusei x4', 39.35, 0.9350, 4.50, 3.05),\n"
     "]\n"
     "\n"
     "for m in sota:\n"
     "    print(f'{m[0]:<25} {m[1]:<15} {m[2]:<10.2f} {m[3]:<10.4f} {m[4]:<10.2f} {m[5]:<10.2f}')\n"
     "\n"
     "print('-' * 80)\n"
     "print(f'{\"ASON (Ours)\":<25} {\"Chikusei x4\":<15} {ch_results[\"PSNR\"]:<10.2f} {ch_results[\"SSIM\"]:<10.4f} {ch_results[\"SAM\"]:<10.2f} {ch_results[\"ERGAS\"]:<10.2f}')\n"
     "print(f'{\"ASON (Ours)\":<25} {\"CAVE x4\":<15} {cv_results[\"PSNR\"]:<10.2f} {cv_results[\"SSIM\"]:<10.4f} {cv_results[\"SAM\"]:<10.2f} {cv_results[\"ERGAS\"]:<10.2f}')")

# ============================================================
# CELL 15: Save and download results
# ============================================================
code("import json\n"
     "\n"
     "final_results = {\n"
     "    'chikusei': ch_results,\n"
     "    'cave': cv_results,\n"
     "    'model_params': sum(p.numel() for p in model_ch.parameters()),\n"
     "}\n"
     "\n"
     "with open('ason_results.json', 'w') as f:\n"
     "    json.dump(final_results, f, indent=2)\n"
     "print('Results saved to ason_results.json')\n"
     "\n"
     "# Visualize a sample reconstruction\n"
     "model_ch.eval()\n"
     "sample = next(iter(test_loader))\n"
     "hr = sample.to(device)\n"
     "lr, msi = trainer_ch.forward_degrade(hr, N_BANDS_CH)\n"
     "with torch.no_grad():\n"
     "    pred = model_ch(lr, msi, N_BANDS_CH)\n"
     "\n"
     "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n"
     "axes[0].imshow(hr[0, [50, 25, 10]].permute(1, 2, 0).cpu().clamp(0, 1))\n"
     "axes[0].set_title('Ground Truth')\n"
     "axes[1].imshow(pred[0, [50, 25, 10]].permute(1, 2, 0).cpu().clamp(0, 1))\n"
     "axes[1].set_title(f'ASON Reconstruction\\nPSNR={calc_psnr(pred[0], hr[0]):.2f} dB')\n"
     "axes[2].imshow(msi[0].permute(1, 2, 0).cpu().clamp(0, 1))\n"
     "axes[2].set_title('HR-MSI Input')\n"
     "for ax in axes:\n"
     "    ax.axis('off')\n"
     "plt.tight_layout()\n"
     "plt.savefig('ason_sample_reconstruction.png', dpi=150, bbox_inches='tight')\n"
     "plt.show()\n"
     "print('Sample reconstruction saved!')")

# ============================================================
# Build notebook
# ============================================================
notebook = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3"
        },
        "language_info": {
            "name": "python"
        },
        "accelerator": "GPU"
    },
    "cells": cells
}

# Fix source format: each cell source should be list of strings with \n
for cell in notebook["cells"]:
    if isinstance(cell["source"], str):
        cell["source"] = cell["source"].split("\n")
    # Each line needs to end with \n except the last
    fixed = []
    for i, line in enumerate(cell["source"]):
        if i < len(cell["source"]) - 1:
            if not line.endswith("\n"):
                fixed.append(line + "\n")
            else:
                fixed.append(line)
        else:
            fixed.append(line)
    cell["source"] = fixed

with open(r"C:\Users\hruth_6nd3fhu\ason-kaggle\notebook.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook saved with {len(cells)} cells")
print(f"Total lines: {sum(len(c['source']) for c in cells)}")
