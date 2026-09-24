"""Diffusion-NullFusion â€” CAVE x4 SOTA-beater.
See the parent notebook for the full bug-fix list.
"""
from __future__ import annotations
import argparse, atexit, glob, json, math, os, random, signal, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset
from scipy.ndimage import convolve, uniform_filter

_NIKON_D700_31 = np.array([
    [0.0050,0.0130,0.2400],[0.0060,0.0190,0.3600],[0.0070,0.0280,0.5200],
    [0.0080,0.0420,0.7100],[0.0090,0.0620,0.8800],[0.0100,0.0890,0.9800],
    [0.0110,0.1250,1.0000],[0.0130,0.1750,0.9500],[0.0150,0.2400,0.8400],
    [0.0180,0.3300,0.6900],[0.0230,0.4500,0.5300],[0.0310,0.5900,0.3900],
    [0.0450,0.7400,0.2700],[0.0700,0.8800,0.1800],[0.1100,0.9700,0.1200],
    [0.1700,1.0000,0.0800],[0.2600,0.9800,0.0550],[0.3800,0.9100,0.0400],
    [0.5300,0.8000,0.0300],[0.6900,0.6700,0.0230],[0.8300,0.5300,0.0180],
    [0.9300,0.4000,0.0140],[0.9900,0.2900,0.0110],[1.0000,0.2100,0.0090],
    [0.9700,0.1500,0.0075],[0.9000,0.1050,0.0062],[0.8000,0.0740,0.0052],
    [0.6800,0.0520,0.0044],[0.5500,0.0370,0.0037],[0.4300,0.0260,0.0031],
    [0.3200,0.0190,0.0026],
], dtype=np.float32)

def nike_d700_srf(bands=31):
    src = _NIKON_D700_31
    if bands != src.shape[0]:
        xs = np.linspace(0.0, 1.0, src.shape[0])
        xd = np.linspace(0.0, 1.0, bands)
        src = np.stack([np.interp(xd, xs, src[:, i]) for i in range(3)], axis=1)
    srf = src.astype(np.float32)
    return srf / np.maximum(srf.sum(axis=0, keepdims=True), 1e-8)

def gaussian_kernel2d(size=9, sigma=1.2):
    ax = np.arange(size, dtype=np.float32) - (size-1)/2.0
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-0.5*(xx**2+yy**2)/sigma**2)
    return (k/k.sum()).astype(np.float32)

class DegradationOp(nn.Module):
    def __init__(self, scale, ksize=9, sigma=1.2):
        super().__init__(); self.scale = scale
        k = gaussian_kernel2d(ksize, sigma)
        self.register_buffer('k', torch.from_numpy(k)[None,None])
    def forward(self, x):
        C = x.shape[1]
        return F.conv2d(x, self.k.repeat(C,1,1,1), padding=4, groups=C)[:,:,::self.scale,::self.scale]
    def transpose(self, y, out_hw):
        C = y.shape[1]
        up = F.interpolate(y, size=out_hw, mode='bicubic', align_corners=False)
        return F.conv2d(up, self.k.repeat(C,1,1,1), padding=4, groups=C)

def block_cg(applyA, rhs, steps, tol=1e-10):
    z = tuple(torch.zeros_like(r) for r in rhs)
    r = tuple(rh - a for rh, a in zip(rhs, applyA(z)))
    p = tuple(ri.clone() for ri in r)
    rs = sum((ri*ri).flatten(1).sum(1) for ri in r)
    shape = (rhs[0].shape[0],) + (1,)*(rhs[0].dim()-1)
    for _ in range(steps):
        ap = applyA(p)
        denom = sum((pi*ai).flatten(1).sum(1) for pi, ai in zip(p, ap))
        alpha = (rs/denom.clamp_min(tol)).reshape(*shape)
        z = tuple(zi+alpha*pi for zi, pi in zip(z, p))
        r = tuple(ri-alpha*ai for ri, ai in zip(r, ap))
        rs_new = sum((ri*ri).flatten(1).sum(1) for ri in r)
        beta = (rs_new/rs.clamp_min(tol)).reshape(*shape)
        p = tuple(ri+beta*pi for ri, pi in zip(r, p))
        rs = rs_new
    return z

class CombinedOperator(nn.Module):
    def __init__(self, scale, bands, msi, srf_np, ksize=9, sigma=1.2, cg_steps=20, ridge=1e-6):
        super().__init__()
        self.D = DegradationOp(scale, ksize, sigma)
        self.scale = scale; self.bands = bands; self.msi_bands = msi
        self.cg_steps = cg_steps; self.ridge = ridge
        self.register_buffer('srf', torch.from_numpy(srf_np).float())
    def R(self, x): return torch.einsum('bchw,cm->bmhw', x, self.srf)
    def Rt(self, m): return torch.einsum('bmhw,cm->bchw', m, self.srf)
    def forward(self, x): return self.D(x), self.R(x)
    def adjoint(self, yH, yM, out_hw): return self.D.transpose(yH, out_hw) + self.Rt(yM)
    def apply_gram(self, zH, zM, out_hw):
        Dt = self.D.transpose(zH, out_hw); RtM = self.Rt(zM)
        return self.D(Dt)+self.D(RtM)+self.ridge*zH, self.R(Dt)+self.R(RtM)+self.ridge*zM
    def pinv(self, yH, yM, out_hw):
        zH, zM = block_cg(lambda p: self.apply_gram(p[0],p[1],out_hw), (yH,yM), self.cg_steps)
        return self.adjoint(zH, zM, out_hw)
    def project_null(self, v):
        out_hw = (v.shape[-2], v.shape[-1])
        yH, yM = self.forward(v)
        return v - self.pinv(yH, yM, out_hw)

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=8):
        super().__init__(); self.dim=dim; self.num_heads=num_heads
        self.head_dim=dim//num_heads; self.scale=self.head_dim**-0.5
        self.qkv=nn.Linear(dim,dim*3); self.proj=nn.Linear(dim,dim)
        self.window_size=window_size
    def forward(self, x):
        B,N,C=x.shape; H=W=int(math.sqrt(N)); ws=self.window_size
        x2d=x.reshape(B,H,W,C); pad_h=(ws-H%ws)%ws; pad_w=(ws-W%ws)%ws
        if pad_h or pad_w: x2d=F.pad(x2d,(0,0,0,pad_w,0,pad_h))
        Hp,Wp=H+pad_h,W+pad_w
        xw=x2d.reshape(B,Hp//ws,ws,Wp//ws,ws,C).permute(0,1,3,2,4,5).reshape(-1,ws*ws,C)
        qkv=self.qkv(xw).reshape(-1,3,self.num_heads,self.head_dim)
        q,k,v=qkv.unbind(1)
        attn=((q@k.transpose(-2,-1))*self.scale).softmax(dim=-1)
        out=(attn@v).transpose(1,2).reshape(-1,ws*ws,C)
        out=self.proj(out).reshape(B,Hp//ws,Wp//ws,ws,ws,C).permute(0,1,3,2,4,5).reshape(B,Hp,Wp,C)
        if pad_h or pad_w: out=out[:,:H,:W,:]
        return out.reshape(B,H*W,C)

class AdaLN(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__(); self.norm=nn.LayerNorm(dim,elementwise_affine=False)
        self.proj=nn.Linear(cond_dim,dim*2)
    def forward(self,x,cond):
        gamma,beta=self.proj(cond).unsqueeze(1).chunk(2,dim=-1)
        return self.norm(x)*(1+gamma)+beta

class SpectralAttention(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__(); hidden=max(dim//reduction,8)
        self.fc=nn.Sequential(nn.Linear(dim,hidden),nn.GELU(),nn.Linear(hidden,dim),nn.Sigmoid())
    def forward(self,x): gate=self.fc(x.mean(dim=1)); return x*gate.unsqueeze(1)

class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, cond_dim, window_size=8, mlp_ratio=4.0):
        super().__init__()
        self.norm1=AdaLN(dim,cond_dim); self.attn=WindowAttention(dim,num_heads,window_size)
        self.norm2=AdaLN(dim,cond_dim)
        self.mlp=nn.Sequential(nn.Linear(dim,int(dim*mlp_ratio)),nn.GELU(),nn.Linear(int(dim*mlp_ratio),dim))
        self.spec=SpectralAttention(dim)
    def forward(self,x,cond):
        x=x+self.attn(self.norm1(x,cond)); x=x+self.mlp(self.norm2(x,cond)); return self.spec(x)

class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.reduction=nn.Linear(4*dim,2*dim,bias=False); self.norm=nn.LayerNorm(4*dim)
    def forward(self,x,H,W):
        B=x.shape[0]; x=x.reshape(B,H,W,-1)
        x0,x1,x2,x3=x[:,0::2,0::2],x[:,1::2,0::2],x[:,0::2,1::2],x[:,1::2,1::2]
        x=torch.cat([x0,x1,x2,x3],-1).reshape(B,-1,4*x.shape[-1])
        return self.reduction(self.norm(x)),H//2,W//2

class PatchExpanding(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.linear=nn.Linear(dim,4*dim); self.norm=nn.LayerNorm(dim)
    def forward(self,x,H,W):
        x=self.linear(self.norm(x)).reshape(x.shape[0],H,W,2,2,-1).permute(0,1,3,2,4,5).reshape(x.shape[0],H*2,W*2,-1)
        return x.reshape(x.shape[0],H*2*W*2,-1),H*2,W*2

class ChannelProject(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__(); self.proj=nn.Linear(in_dim,out_dim)
    def forward(self,x): return self.proj(x)

class MultiScaleSwinUNet(nn.Module):
    def __init__(self, in_ch=31, base_dim=64, num_heads=None, cond_dim=64,
                 window_size=8, depths=None, use_checkpoint=False):
        super().__init__()
        if num_heads is None: num_heads=[4,8,16]
        if depths is None: depths=[2,2,4,2,2]
        dims=[base_dim,base_dim*2,base_dim*4]; self.use_checkpoint=use_checkpoint
        self.input_proj=nn.Linear(in_ch,dims[0])
        self.enc1=nn.ModuleList([SwinBlock(dims[0],num_heads[0],cond_dim,window_size) for _ in range(depths[0])])
        self.merge1=PatchMerging(dims[0])
        self.enc2=nn.ModuleList([SwinBlock(dims[1],num_heads[1],cond_dim,window_size) for _ in range(depths[1])])
        self.merge2=PatchMerging(dims[1])
        self.bottleneck=nn.ModuleList([SwinBlock(dims[2],num_heads[2],cond_dim,window_size) for _ in range(depths[2])])
        self.expand2=PatchExpanding(dims[2]); self.proj_skip2=ChannelProject(dims[1],dims[2]); self.proj_dec2=ChannelProject(dims[2],dims[1])
        self.dec2=nn.ModuleList([SwinBlock(dims[1],num_heads[1],cond_dim,window_size) for _ in range(depths[3])])
        self.expand1=PatchExpanding(dims[1]); self.proj_skip1=ChannelProject(dims[0],dims[1]); self.proj_dec1=ChannelProject(dims[1],dims[0])
        self.dec1=nn.ModuleList([SwinBlock(dims[0],num_heads[0],cond_dim,window_size) for _ in range(depths[4])])
        self.output_proj=nn.Linear(dims[0],in_ch)
    def _run(self,blocks,x,cond):
        for b in blocks:
            if self.use_checkpoint and self.training and x.requires_grad:
                x=checkpoint(b,x,cond,use_reentrant=False)
            else: x=b(x,cond)
        return x
    def forward(self,x,cond):
        B,C,H,W=x.shape; x=x.reshape(B,C,H*W).permute(0,2,1)
        x=self.input_proj(x)
        x=self._run(self.enc1,x,cond); skip1=x; x,H1,W1=self.merge1(x,H,W)
        x=self._run(self.enc2,x,cond); skip2=x; x,H2,W2=self.merge2(x,H1,W1)
        x=self._run(self.bottleneck,x,cond)
        x,H2,W2=self.expand2(x,H2,W2); x=self.proj_dec2(x+self.proj_skip2(skip2))
        x=self._run(self.dec2,x,cond)
        x,H1,W1=self.expand1(x,H1,W1); x=self.proj_dec1(x+self.proj_skip1(skip1))
        x=self._run(self.dec1,x,cond)
        return self.output_proj(x).reshape(B,H,W,-1).permute(0,3,1,2)

class CosineSchedule:
    def __init__(self, T=1000, s=0.008):
        self.T=T; steps=torch.arange(T+1,dtype=torch.float64)
        f=torch.cos((steps/T+s)/(1+s)*math.pi*0.5)**2
        self.alpha_bar=(f/f[0]).float()
        beta=1-self.alpha_bar[1:]/self.alpha_bar[:-1]
        self.beta=torch.clamp(beta,max=0.999).float(); self.alpha=1-self.beta
    def to(self,device):
        self.alpha_bar=self.alpha_bar.to(device); self.beta=self.beta.to(device)
        self.alpha=self.alpha.to(device); return self
    def add_noise(self,x0,noise,t):
        ab=self.alpha_bar[t].reshape(-1,1,1,1)
        return torch.sqrt(ab)*x0+torch.sqrt(1-ab)*noise
    def ddim_step(self,x_t,eps_pred,t,t_prev):
        ab_t=self.alpha_bar[t].reshape(-1,1,1,1); ab_p=self.alpha_bar[t_prev].reshape(-1,1,1,1)
        x0_pred=(x_t-torch.sqrt(1-ab_t)*eps_pred)/torch.sqrt(ab_t)
        x0_pred=x0_pred.clamp(0,1)
        return torch.sqrt(ab_p)*x0_pred+torch.sqrt(1-ab_p)*eps_pred

class SensorEmbedding(nn.Module):
    def __init__(self, srf_matrix, out_dim=64):
        super().__init__(); flat=torch.from_numpy(srf_matrix).float().flatten()
        self.register_buffer('srf_flat',flat)
        self.mlp=nn.Sequential(nn.Linear(flat.numel(),128),nn.GELU(),nn.Linear(128,out_dim))
    def forward(self): return self.mlp(self.srf_flat.unsqueeze(0)).squeeze(0)

class DiffusionNullFusion(nn.Module):
    def __init__(self, bands=31, msi=3, base_dim=64, scale=4, cond_dim=64, T=1000,
                 num_heads=None, depths=None, window_size=8, use_checkpoint=False, cg_steps=20):
        super().__init__(); self.bands=bands; self.scale=scale; self.T=T
        srf_np = nike_d700_srf(bands)
        self.register_buffer('srf', torch.from_numpy(srf_np).float())
        self.op = CombinedOperator(scale, bands, msi, srf_np, cg_steps=cg_steps, ridge=1e-6)
        self.sensor_embed = SensorEmbedding(srf_np, cond_dim)
        self.schedule = CosineSchedule(T)
        self.cond_proj = nn.Sequential(nn.Conv2d(msi+bands,cond_dim,1),nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.time_mlp = nn.Sequential(nn.Linear(1,cond_dim),nn.GELU(),nn.Linear(cond_dim,cond_dim))
        self.unet = MultiScaleSwinUNet(in_ch=bands,base_dim=base_dim,num_heads=num_heads,
                                       cond_dim=cond_dim,window_size=window_size,depths=depths,
                                       use_checkpoint=use_checkpoint)
    def _conditioning(self,yH,yM,H_hr,W_hr):
        base = self.op.pinv(yH,yM,(H_hr,W_hr))
        obs = F.interpolate(yH,(H_hr,W_hr),mode='bicubic',align_corners=False)
        cond = self.cond_proj(torch.cat([yM,obs],1))+self.sensor_embed()
        return cond, base
    def forward(self,x_t,t,cond):
        t_emb = self.time_mlp(t.float().unsqueeze(-1))
        return self.unet(x_t, cond+t_emb)
    @torch.no_grad()
    def inference(self,yH,yM,num_samples=5,ddim_steps=50):
        H_hr,W_hr=yM.shape[-2],yM.shape[-1]; cond,base=self._conditioning(yH,yM,H_hr,W_hr)
        samples=[]
        for _ in range(num_samples):
            x_t=torch.randn(1,self.bands,H_hr,W_hr,device=yH.device)
            ts=torch.linspace(self.T-1,0,ddim_steps,dtype=torch.long,device=yH.device)
            for i in range(len(ts)-1):
                eps=self(x_t,ts[i],cond); x_t=self.schedule.ddim_step(x_t,eps,ts[i],ts[i+1])
            samples.append(x_t)
        avg=torch.mean(torch.stack(samples),0)
        avg_null=self.op.project_null(avg)
        return {'out':base+avg_null,'base':base,'null':avg_null}

class CAVEDataset(Dataset):
    def __init__(self,root,split='train',bands=31,scale=4,patch_size=80,max_dim=512):
        self.root=root; self.split=split; self.bands=bands; self.scale=scale
        self.patch_size=patch_size; self.is_train=split.lower()=='train'
        self.srf=nike_d700_srf(bands); self.kernel=gaussian_kernel2d(9,1.2)
        self.scenes=self._discover(root,split)
        self._cache={n:self._load(p,bands,max_dim) for n,p in self.scenes}
        exp=20 if self.is_train else 12
        if len(self.scenes)!=exp: print(f'[CAVE] WARN: expected {exp} {split} scenes, found {len(self.scenes)}')
        print(f'[CAVE] {split}: {len(self.scenes)} scenes')
    def _discover(self,root,split):
        sd=None
        for n in (split,split.capitalize(),split.upper()):
            c=os.path.join(root,n)
            if os.path.isdir(c): sd=c; break
        if sd is None: raise FileNotFoundError(f'split {split!r} not found under {root}')
        scenes=[]
        for e in sorted(os.scandir(sd),key=lambda x:x.name):
            if not e.is_dir(follow_symlinks=False): continue
            for bn in ('band_01.png','Band_01.png','BAND_01.png'):
                if os.path.isfile(os.path.join(e.path,bn)): scenes.append((e.name,e.path)); break
        if not scenes: raise FileNotFoundError(f'no scenes under {sd}')
        return scenes
    def _load(self,d,bands,max_dim):
        try: from PIL import Image; use_pil=True
        except: use_pil=False
        arrs=[]
        for i in range(1,bands+1):
            ok=False
            for p in (f'band_{i:02d}.png',f'Band_{i:02d}.png',f'BAND_{i:02d}.png',f'band_{i}.png'):
                path=os.path.join(d,p)
                if os.path.isfile(path):
                    if use_pil: a=np.asarray(Image.open(path),dtype=np.float32)
                    else: import cv2; a=cv2.imread(path,cv2.IMREAD_GRAYSCALE).astype(np.float32)
                    if a.max()>1.0: a=a/255.0
                    arrs.append(a); ok=True; break
            if not ok: raise FileNotFoundError(f'band {i} not in {d}')
        cube=np.stack(arrs,0)
        if max_dim and (cube.shape[1]>max_dim or cube.shape[2]>max_dim):
            y0=max(0,(cube.shape[1]-max_dim)//2); x0=max(0,(cube.shape[2]-max_dim)//2)
            cube=cube[:,y0:y0+max_dim,x0:x0+max_dim]
        return cube.astype(np.float32)
    def __len__(self): return 10000 if self.is_train else len(self.scenes)
    def _sim(self,gt):
        C,H,W=gt.shape; bl=np.empty_like(gt)
        for c in range(C): bl[c]=convolve(gt[c],self.kernel,mode='wrap')
        hr=H//self.scale; y0=(H-hr*self.scale)//2; x0=(W-hr*self.scale)//2
        lr=bl[:,y0::self.scale,x0::self.scale].astype(np.float32)
        msi=np.einsum('chw,cm->mhw',gt,self.srf).astype(np.float32)
        return lr,np.clip(msi,0,1)
    def __getitem__(self,idx):
        if self.is_train:
            name=list(self._cache.keys())[np.random.randint(0,len(self._cache))]
            gt=self._cache[name].copy()
            _,H,W=gt.shape; p=min(self.patch_size,H,W)
            y=np.random.randint(0,H-p+1); x=np.random.randint(0,W-p+1)
            gt=gt[:,y:y+p,x:x+p]
            if np.random.random()<0.5: gt=gt[:,:,::-1].copy()
            if np.random.random()<0.5: gt=gt[:,::-1,:].copy()
            k=np.random.randint(0,4)
            if k: gt=np.rot90(gt,k,axes=(-2,-1)).copy()
        else:
            name,_=self.scenes[idx%len(self.scenes)]
            gt=self._cache[name].copy()
            H,W=gt.shape[1],gt.shape[2]; H=(H//self.scale)*self.scale; W=(W//self.scale)*self.scale
            gt=gt[:,:H,:W]
        lr,msi=self._sim(gt)
        return {'gt':torch.from_numpy(gt.astype(np.float32)),
                'lr':torch.from_numpy(lr.astype(np.float32)),
                'msi':torch.from_numpy(msi.astype(np.float32))}

def charbonnier_loss(pred,target,eps=1e-3): return torch.sqrt((pred-target)**2+eps**2).mean()
def ssim_loss(pred,target,size=11,sigma=1.5):
    c=pred.shape[1]; coords=torch.arange(size,device=pred.device,dtype=pred.dtype)-size//2
    g=torch.exp(-(coords**2)/(2*sigma**2)); g=g/g.sum()
    win=(g[:,None]@g[None,:]).expand(c,1,size,size)
    mu1=F.conv2d(pred,win,padding=size//2,groups=c)
    mu2=F.conv2d(target,win,padding=size//2,groups=c)
    mu1s,mu2s,mu12=mu1**2,mu2**2,mu1*mu2
    s1=F.conv2d(pred*pred,win,padding=size//2,groups=c)-mu1s
    s2=F.conv2d(target*target,win,padding=size//2,groups=c)-mu2s
    s12=F.conv2d(pred*target,win,padding=size//2,groups=c)-mu12
    c1,c2=(0.01)**2,(0.03)**2
    return 1.0-(((2*mu12+c1)*(2*s12+c2))/((mu1s+mu2s+c1)*(s1+s2+c2))).mean()
def sam_loss(pred,target,eps=1e-6):
    p,t=pred.flatten(2),target.flatten(2)
    num=(p*t).sum(dim=1); den=p.norm(dim=1)*t.norm(dim=1)
    cos=(num/den.clamp_min(eps)).clamp(-1+1e-6,1-1e-6)
    return torch.acos(cos).mean()
def gradient_loss(pred,target):
    return F.l1_loss(pred[...,1:,:]-pred[...,:-1,:],target[...,1:,:]-target[...,:-1,:])+\
           F.l1_loss(pred[...,:,1:]-pred[...,:,:-1],target[...,:,1:]-target[...,:,:-1])

def diffusion_loss(model,x0,cond,schedule,min_snr_gamma=5.0):
    B=x0.shape[0]; t=torch.randint(0,schedule.T,(B,),device=x0.device)
    noise=torch.randn_like(x0); x_t=schedule.add_noise(x0,noise,t)
    pred=model(x_t,t,cond)
    ps=F.mse_loss(pred,noise,reduction='none').flatten(1).mean(1)
    ab=schedule.alpha_bar[t].clamp(1e-5,1-1e-5); snr=ab/(1-ab)
    w=(snr.clamp(max=min_snr_gamma)/snr).detach()
    return (ps*w).mean()

def total_loss(model,gt,yH,yM,schedule,w_char=1.0,w_ssim=0.5,w_sam=0.05,w_grad=0.2,
               w_noise=1.0,w_phys=0.1,min_snr_gamma=5.0):
    H_hr,W_hr=yM.shape[-2],yM.shape[-1]; cond,base=model._conditioning(yH,yM,H_hr,W_hr)
    x0=model.op.project_null(gt-base)
    l_noise=diffusion_loss(model,x0,cond,schedule,min_snr_gamma)
    pred=base+x0
    l_char=charbonnier_loss(pred,gt)
    l_ssim=ssim_loss(pred.clamp(0,1),gt)
    l_sam=sam_loss(pred,gt)
    l_grad=gradient_loss(pred,gt)
    l_phys=F.mse_loss(model.op.D(pred),yH)+F.mse_loss(model.op.R(pred),yM)
    total=(w_noise*l_noise+w_phys*l_phys+w_char*l_char+w_ssim*l_ssim+w_sam*l_sam+w_grad*l_grad)
    return total,{'noise':l_noise.item(),'phys':l_phys.item(),'char':l_char.item(),
                  'ssim':l_ssim.item(),'sam':l_sam.item(),'grad':l_grad.item()}

def psnr_np(pred,gt):
    mse=np.mean((pred-gt)**2); return 100.0 if mse<1e-12 else -10*np.log10(mse)
def sam_np(pred,gt):
    p=pred.reshape(pred.shape[0],-1); g=gt.reshape(gt.shape[0],-1)
    p=p/(np.linalg.norm(p,axis=0,keepdims=True)+1e-8); g=g/(np.linalg.norm(g,axis=0,keepdims=True)+1e-8)
    return np.mean(np.arccos(np.clip((p*g).sum(0),-1,1)))*180/math.pi
def ssim_np(pred,gt):
    C1,C2=0.01**2,0.03**2; mu1=uniform_filter(pred,3,mode='reflect'); mu2=uniform_filter(gt,3,mode='reflect')
    s12=uniform_filter(pred*gt,3,mode='reflect')-mu1*mu2
    s1=uniform_filter(pred**2,3,mode='reflect')-mu1**2; s2=uniform_filter(gt**2,3,mode='reflect')-mu2**2
    return np.mean(((2*mu1*mu2+C1)*(2*s12+C2))/((mu1**2+mu2**2+C1)*(s1+s2+C2)+1e-8))
def ergas_np(pred,gt,scale=4):
    C=pred.shape[0]; e=sum(((pred-gt)**2)[c].mean()/(gt[c].mean()**2+1e-8) for c in range(C))
    return math.sqrt(e/C)*100*scale

class EMA:
    def __init__(self,m,d=0.999): self.d=d; self.shadow={k:v.detach().clone() for k,v in m.state_dict().items()}
    def update(self,m):
        with torch.no_grad():
            for k,v in m.state_dict().items():
                if v.dtype.is_floating_point and k in self.shadow:
                    self.shadow[k].mul_(self.d).add_(v.detach(),alpha=1-self.d)
    def apply(self,m): m.load_state_dict(self.shadow,strict=False)
    def restore(self,m): self.shadow={k:v.detach().clone() for k,v in m.state_dict().items()}

class KaggleCheckpoint:
    def __init__(self,save_dir,device='cuda'):
        self.save_dir=save_dir; self.device=device
        self.ckpt_path=os.path.join(save_dir,'resume.pth')
        self.best_path=os.path.join(save_dir,'best.pth')
        self._t0=time.time(); self._limit=11.5*3600
        self._saved=False; self._objs={}; self._register_signal()
    def _register_signal(self):
        if sys.platform=='win32': return
        orig=signal.getsignal(signal.SIGTERM)
        def handler(s,f):
            if self._saved: return
            print(f'\n[KAGGLE] Signal {s} â€” saving...')
            try: self.save(**self._objs,force=True); self._saved=True
            except Exception as e: print(f'[KAGGLE] Failed: {e}')
            if callable(orig): orig(s,f)
        try: signal.signal(signal.SIGTERM,handler)
        except: pass
    def time_remaining(self): return max(0.0,self._limit-(time.time()-self._t0))
    def should_save(self,step,every=500):
        if step%every==0: return True
        if self.time_remaining()<1800 and step%200==0: return True
        return False
    def save(self,model,ema,opt,scheduler,epoch,best_psnr,best_epoch,force=False):
        self._objs=dict(model=model,ema=ema,opt=opt,scheduler=scheduler,
                        epoch=epoch,best_psnr=best_psnr,best_epoch=best_epoch)
        tmp=self.ckpt_path+'.tmp'
        torch.save({'model':model.state_dict(),'ema':ema.shadow,
                    'opt':opt.state_dict(),'scheduler':scheduler.state_dict(),
                    'epoch':epoch,'best_psnr':best_psnr,'best_epoch':best_epoch,
                    'wall_time':time.time()-self._t0},tmp)
        os.replace(tmp,self.ckpt_path)
    def save_best(self,model,opt,ema,scheduler,epoch,best_psnr,best_epoch,val):
        self.save(model,ema,opt,scheduler,epoch,best_psnr,best_epoch)
        torch.save({'model':model.state_dict(),'val':val,'epoch':epoch},self.best_path)
        print(f'[ckpt] Best saved: PSNR={val["psnr"]:.4f}')
    def load(self,model,ema,opt,scheduler):
        if not os.path.exists(self.ckpt_path): return 1,float('-inf'),0
        ck=torch.load(self.ckpt_path,map_location=self.device,weights_only=False)
        model.load_state_dict(ck['model'])
        ema.shadow={k:v.to(self.device) for k,v in ck['ema'].items()}
        opt.load_state_dict(ck['opt']); scheduler.load_state_dict(ck['scheduler'])
        ep=ck['epoch']; bp=ck.get('best_psnr',float('-inf')); be=ck.get('best_epoch',0)
        print(f'[ckpt] Resumed: epoch {ep}, best PSNR={bp:.4f}, wall={ck.get("wall_time",0)/3600:.1f}h')
        return ep+1,bp,be

print('Library OK')
