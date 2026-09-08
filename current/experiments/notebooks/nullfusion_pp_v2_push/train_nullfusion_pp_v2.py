"""NullFusion++ v2 â€” SOTA-beater for Chikusei x4.
Run on Kaggle:  python train_nullfusion_pp_v2.py --root /kaggle/input/chikusei
"""
from __future__ import annotations
import argparse, glob, json, math, os, random, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.io import loadmat
from scipy.ndimage import convolve, uniform_filter
from torch.utils.data import Dataset

# ---- SRF & degradation ----
def chikusei_srf(bands=128):
    wl = np.linspace(363.0, 1018.0, 128)
    raw = np.stack([np.exp(-((wl-620)**2)/(2*80**2)), np.exp(-((wl-540)**2)/(2*70**2)), np.exp(-((wl-460)**2)/(2*60**2))], axis=1).astype(np.float32)
    return raw / np.maximum(raw.sum(axis=0, keepdims=True), 1e-8)

def gaussian_kernel2d(size=9, sigma=1.2):
    ax = np.arange(size, dtype=np.float32) - (size-1)/2
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-0.5*(xx**2+yy**2)/sigma**2)
    return (k / k.sum()).astype(np.float32)

class DegradationOp(nn.Module):
    def __init__(self, scale, ksize=9, sigma=1.2):
        super().__init__(); self.scale = scale
        k = gaussian_kernel2d(ksize, sigma)
        self.register_buffer('k', torch.from_numpy(k)[None,None])
    def forward(self, x):
        C = x.shape[1]
        return F.conv2d(x, self.k.repeat(C,1,1,1), padding=4, groups=C)[:,:,::self.scale,::self.scale]
    def transpose(self, y, hw):
        C = y.shape[1]
        up = F.interpolate(y, size=hw, mode='bicubic', align_corners=False)
        return F.conv2d(up, self.k.repeat(C,1,1,1), padding=4, groups=C)

# ---- CG solver ----
def scalar_cg(applyA, rhs, steps, tol=1e-10):
    z = torch.zeros_like(rhs); r = rhs - applyA(z); p = r.clone()
    rs = (r*r).flatten(1).sum(1)
    for _ in range(steps):
        ap = applyA(p); denom = (p*ap).flatten(1).sum(1)
        alpha = (rs/denom.clamp_min(tol)).reshape(-1,1,1,1)
        z = z + alpha*p; r = r - alpha*ap
        rs_new = (r*r).flatten(1).sum(1)
        beta = (rs_new/rs.clamp_min(tol)).reshape(-1,1,1,1)
        p = r + beta*p; rs = rs_new
    return z

class RangeNullProjector(nn.Module):
    def __init__(self, scale, cg_steps=40, ridge=1e-4):
        super().__init__()
        self.D = DegradationOp(scale); self.scale = scale
        self.cg_steps = cg_steps; self.ridge = ridge
    def _normal_op(self, out_hw):
        def applyA(z): return self.D(self.D.transpose(z, out_hw)) + self.ridge * z
        return applyA
    def pinv(self, yH, out_hw):
        return self.D.transpose(scalar_cg(self._normal_op(out_hw), yH, self.cg_steps), out_hw)
    def project_null(self, v, out_hw=None):
        if out_hw is None: out_hw = (v.shape[-2], v.shape[-1])
        return v - self.pinv(self.D(v), out_hw)

# ---- Blocks ----
class RCAB(nn.Module):
    def __init__(self, ch, red=16):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(ch,ch,3,1,1), nn.GELU(), nn.Conv2d(ch,ch,3,1,1))
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(ch,ch//red), nn.ReLU(), nn.Linear(ch//red,ch), nn.Sigmoid())
    def forward(self, x): return x + self.body(x) * self.ca(x).unsqueeze(-1).unsqueeze(-1)

class SpatialAttn(nn.Module):
    def __init__(self, ch, window=8):
        super().__init__(); self.window=window
        self.qkv = nn.Conv2d(ch, ch*3, 1); self.proj = nn.Conv2d(ch, ch, 1); self.scale = ch**-0.5
    def forward(self, x):
        B,C,H,W = x.shape; ws=self.window
        ph,pw = (ws-H%ws)%ws, (ws-W%ws)%ws
        xp = F.pad(x,(0,pw,0,ph)) if (ph or pw) else x
        _,_,Hp,Wp = xp.shape; q,k,v = self.qkv(xp).chunk(3, dim=1)
        nH,nW = Hp//ws, Wp//ws
        q = q.reshape(B,C,nH,ws,nW,ws).permute(0,2,4,3,5,1).reshape(-1,ws*ws,C)
        k = k.reshape(B,C,nH,ws,nW,ws).permute(0,2,4,3,5,1).reshape(-1,ws*ws,C)
        v = v.reshape(B,C,nH,ws,nW,ws).permute(0,2,4,3,5,1).reshape(-1,ws*ws,C)
        attn = ((q @ k.transpose(-2,-1)) * self.scale).softmax(dim=-1)
        out = (attn @ v).reshape(B,nH,nW,ws,ws,C).permute(0,5,3,1,4,2).reshape(B,C,Hp,Wp)
        if ph or pw: out = out[:,:,:H,:W]
        return x + self.proj(out)

class CrossAttn(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__(); self.heads=heads; self.d=ch//heads
        self.q=nn.Conv2d(ch,ch,1); self.kv=nn.Conv2d(ch,ch*2,1); self.proj=nn.Conv2d(ch,ch,1); self.scale=self.d**-0.5
    def forward(self, q, ctx):
        B,C,Hq,Wq = q.shape
        if ctx.shape[-2:]!=(Hq,Wq): ctx=F.adaptive_avg_pool2d(ctx,(Hq,Wq))
        qh = self.q(q).reshape(B,self.heads,self.d,Hq*Wq).transpose(2,3)
        k,v = self.kv(ctx).chunk(2, dim=1)
        k = k.reshape(B,self.heads,self.d,Hq*Wq).transpose(2,3)
        v = v.reshape(B,self.heads,self.d,Hq*Wq).transpose(2,3)
        attn = ((qh @ k.transpose(-2,-1)) * self.scale).softmax(dim=-1)
        return q + self.proj((attn @ v).transpose(2,3).reshape(B,C,Hq,Wq))

class WindowAttention(nn.Module):
    def __init__(self, ch, ws=8, heads=4):
        super().__init__(); self.ws=ws; self.heads=heads; self.d=ch//heads
        self.qkv=nn.Conv2d(ch,ch*3,1); self.proj=nn.Conv2d(ch,ch,1); self.scale=self.d**-0.5
    def forward(self, x):
        B,C,H,W=x.shape; ws=self.ws
        ph,pw=(ws-H%ws)%ws,(ws-W%ws)%ws
        xp=F.pad(x,(0,pw,0,ph)) if(ph or pw) else x
        _,_,Hp,Wp=xp.shape; q,k,v=self.qkv(xp).chunk(3,dim=1)
        nH,nW=Hp//ws,Wp//ws
        q=q.reshape(B,C,nH,ws,nW,ws).permute(0,2,4,3,5,1).reshape(-1,ws*ws,self.heads,self.d)
        k=k.reshape(B,C,nH,ws,nW,ws).permute(0,2,4,3,5,1).reshape(-1,ws*ws,self.heads,self.d)
        v=v.reshape(B,C,nH,ws,nW,ws).permute(0,2,4,3,5,1).reshape(-1,ws*ws,self.heads,self.d)
        attn=((q@k.transpose(-2,-1))*self.scale).softmax(dim=-1)
        out=(attn@v).reshape(-1,ws*ws,C).reshape(B,nH,nW,ws,ws,C).permute(0,5,3,1,4,2).reshape(B,C,Hp,Wp)
        if ph or pw: out=out[:,:,:H,:W]
        return x+self.proj(out)

class SwinBlock(nn.Module):
    def __init__(self, ch, ws=8, heads=4, mlp_ratio=4):
        super().__init__()
        self.norm1=nn.LayerNorm(ch); self.attn=WindowAttention(ch,ws,heads)
        self.norm2=nn.LayerNorm(ch); self.mlp=nn.Sequential(nn.Linear(ch,ch*mlp_ratio), nn.GELU(), nn.Linear(ch*mlp_ratio,ch))
    def forward(self, x):
        B,C,H,W=x.shape; xt=x.permute(0,2,3,1).reshape(-1,C)
        residual=self.attn(self.norm1(xt.reshape(B,H,W,C)).permute(0,3,1,2)).reshape(-1,C)
        xt=xt+residual; xt=xt+self.mlp(self.norm2(xt))
        return x+xt.reshape(B,H,W,C).permute(0,3,1,2)

class DataConsistencyLayer(nn.Module):
    def __init__(self, bands, width):
        super().__init__()
        self.prior=nn.Sequential(nn.Conv2d(bands,width,3,1,1), RCAB(width), RCAB(width), nn.Conv2d(width,bands,3,1,1))
        self.alpha=nn.Parameter(torch.zeros(1)); self.beta=nn.Parameter(torch.zeros(1))
    def forward(self, x, obs_spec, obs_spat):
        r=self.prior(x)
        return x + torch.sigmoid(self.alpha)*r + torch.sigmoid(self.beta)*(obs_spec + obs_spat - x)

# ---- Wavelet ----
def _haar(x):
    B,C,H,W=x.shape
    if H%2: x=F.pad(x,(0,0,0,1),mode='reflect')
    if W%2: x=F.pad(x,(0,1,0,0),mode='reflect')
    B,C,H,W=x.shape; x=x.reshape(B,C,H//2,2,W//2,2)
    LL=(x[:,:,:,0,:,0]+x[:,:,:,1,:,0]+x[:,:,:,0,:,1]+x[:,:,:,1,:,1])/4
    LH=(x[:,:,:,0,:,0]-x[:,:,:,1,:,0]+x[:,:,:,0,:,1]-x[:,:,:,1,:,1])/4
    HL=(x[:,:,:,0,:,0]+x[:,:,:,1,:,0]-x[:,:,:,0,:,1]-x[:,:,:,1,:,1])/4
    HH=(x[:,:,:,0,:,0]-x[:,:,:,1,:,0]-x[:,:,:,0,:,1]+x[:,:,:,1,:,1])/4
    return LL,LH,HL,HH

def _ihaar(LL,LH,HL,HH):
    B,C,h,w=LL.shape; x=torch.zeros(B,C,h*2,w*2,device=LL.device,dtype=LL.dtype)
    x[:,:,0::2,0::2]=LL+LH+HL+HH; x[:,:,1::2,0::2]=LL-LH+HL-HH
    x[:,:,0::2,1::2]=LL+LH-HL-HH; x[:,:,1::2,1::2]=LL-LH-HL+HH
    return x

class WaveletBranch(nn.Module):
    def __init__(self, bands, width, depth=1):
        super().__init__()
        self.enc1=nn.Sequential(nn.Conv2d(bands*3,width,3,1,1),nn.GELU(),RCAB(width),nn.Conv2d(width,bands*3,3,1,1))
        self.enc2=nn.Sequential(nn.Conv2d(bands*3,width,3,1,1),nn.GELU(),RCAB(width),nn.Conv2d(width,bands*3,3,1,1))
        self.enc3=nn.Sequential(nn.Conv2d(bands*3,width,3,1,1),nn.GELU(),RCAB(width),nn.Conv2d(width,bands*3,3,1,1))
    def forward(self, x):
        LL1,LH1,HL1,HH1=_haar(x)
        c1=self.enc1(torch.cat([LH1,HL1,HH1],1)).chunk(3,1)
        corr1=_ihaar(torch.zeros_like(LL1),c1[0],c1[1],c1[2])
        LL2,LH2,HL2,HH2=_haar(LL1)
        c2=self.enc2(torch.cat([LH2,HL2,HH2],1)).chunk(3,1)
        corr2=F.interpolate(_ihaar(torch.zeros_like(LL2),c2[0],c2[1],c2[2]),size=corr1.shape[-2:],mode='bilinear',align_corners=False)
        LL3,LH3,HL3,HH3=_haar(LL2)
        c3=self.enc3(torch.cat([LH3,HL3,HH3],1)).chunk(3,1)
        corr3=F.interpolate(_ihaar(torch.zeros_like(LL3),c3[0],c3[1],c3[2]),size=corr1.shape[-2:],mode='bilinear',align_corners=False)
        return corr1+corr2+corr3

# ---- Main model ----
class NullFusionPlusV2(nn.Module):
    def __init__(self, bands=128, msi=3, width=32, scale=4, dict_g=48, dict_m=32, dict_f=24):
        super().__init__(); self.bands=bands; W=width
        srf_t=torch.from_numpy(chikusei_srf(bands)).float()
        self.projector=RangeNullProjector(scale, cg_steps=8, ridge=1e-4)
        self.register_buffer('srf', srf_t)
        self.register_buffer('srfinv', torch.linalg.pinv(srf_t))
        self.hsi_stem=nn.Sequential(nn.Conv2d(bands,W,3,1,1),RCAB(W))
        self.msi_stem=nn.Sequential(nn.Conv2d(msi,W,3,1,1),RCAB(W))
        self.msi_detail=nn.Conv2d(msi,W,3,1,1)
        self.cross1=CrossAttn(W,4); self.cross2=CrossAttn(W,4)
        self.fuse_in=nn.Conv2d(W*2,W,1)
        self.enc1=nn.Sequential(RCAB(W))
        self.down1=nn.Conv2d(W,W*2,3,2,1)
        self.enc2=nn.Sequential(RCAB(W*2))
        self.down2=nn.Conv2d(W*2,W*4,3,2,1)
        self.bottleneck=nn.Sequential(SwinBlock(W*4,ws=8,heads=min(W//8,8)), SwinBlock(W*4,ws=8,heads=min(W//8,8)))
        self.up2=nn.ConvTranspose2d(W*4,W*2,2,2)
        self.dec2=nn.Sequential(RCAB(W*2))
        self.up1=nn.ConvTranspose2d(W*2,W,2,2)
        self.dec1=nn.Sequential(RCAB(W))
        self.sa1=SpatialAttn(W,8); self.sa2=SpatialAttn(W*2,8)
        self.unfold1=DataConsistencyLayer(bands,W)
        self.unfold2=DataConsistencyLayer(bands,W)
        self.prior_in=nn.Conv2d(W+W+bands,W,3,1,1)
        pb=[]
        for i in range(6):
            pb.append(RCAB(W))
            if i%2==1: pb.append(SpatialAttn(W,8))
        pb.append(nn.Conv2d(W,bands,3,1,1))
        self.prior=nn.Sequential(*pb)
        self.prior_proj=nn.Conv2d(bands,W,1)
        self.Dg=nn.Parameter(torch.zeros(bands,dict_g))
        self.Dm=nn.Parameter(torch.zeros(bands,dict_m))
        self.Df=nn.Parameter(torch.zeros(bands,dict_f))
        self.scale_w=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(W,32),nn.GELU(),nn.Linear(32,3))
        self.code_g=nn.Conv2d(W,dict_g,3,1,1)
        self.code_m=nn.Conv2d(W,dict_m,3,1,1)
        self.code_f=nn.Conv2d(W,dict_f,3,1,1)
        self.wavelet=WaveletBranch(bands,W//2,depth=1)
        self.wavelet_gate=nn.Sequential(nn.Conv2d(bands*2,bands,1),nn.Sigmoid())
        self.consist=nn.Conv2d(dict_g+dict_m+dict_f,bands,1)
        self._init_dicts()
    def _init_dicts(self):
        s=self.srf; s_pinv=self.srfinv
        with torch.no_grad():
            for D in (self.Dg,self.Dm,self.Df):
                M=torch.randn(self.bands,D.shape[1],device=D.device)
                D0=M-(s@s_pinv)@M
                D.copy_(D0/(D0.norm(dim=0,keepdim=True)+1e-8))
    def _base(self, yH, yM):
        B,_,H_hr,W_hr=yM.shape
        base_lr=self.projector.pinv(yH,(H_hr,W_hr))
        base_srf=torch.einsum('bmhw,mc->bchw',yM,self.srfinv)
        base_r_msi=torch.einsum('bchw,cm->bmhw',base_lr,self.srf)
        base_spec=base_srf-torch.einsum('bmhw,mc->bchw',base_r_msi,self.srfinv)
        return base_lr+base_spec
    def forward(self, yH, yM):
        B,_,H_lr,W_lr=yH.shape; _,_,H_hr,W_hr=yM.shape
        base=self._base(yH,yM)
        f_hsi=self.hsi_stem(yH); f_msi=self.msi_stem(yM)
        f_hsi=self.cross1(f_hsi,f_msi)
        f_msi_pool=F.adaptive_avg_pool2d(f_msi,f_hsi.shape[-2:])
        f_hsi=self.cross2(f_hsi,f_msi_pool)
        e1=self.fuse_in(torch.cat([f_hsi,f_msi_pool],1))
        e1=self.sa1(self.enc1(e1))
        e2=self.sa2(self.enc2(self.down1(e1)))
        bn=self.bottleneck(self.down2(e2))
        d2=self.dec2(self.up2(bn)+e2)
        d1=self.dec1(self.up1(d2)+e1)
        d1_hr=F.interpolate(d1,size=(H_hr,W_hr),mode='bilinear',align_corners=False)
        obs_spec=F.interpolate(yH,size=(H_hr,W_hr),mode='bicubic',align_corners=False)
        obs_spat=torch.einsum('bmhw,mc->bchw',yM,self.srfinv)
        refined=self.unfold1(base,obs_spec,obs_spat)
        refined=self.unfold2(refined,obs_spec,obs_spat)
        f_detail=self.msi_detail(yM)
        cond=torch.cat([d1_hr,f_detail,refined],1)
        v=self.prior(self.prior_in(cond)); v=self.prior_proj(v)
        ag=F.softplus(self.code_g(v)); am=F.softplus(self.code_m(v)); af=F.softplus(self.code_f(v))
        ng=torch.einsum('ck,bkhw->bchw',self.Dg,ag)
        nm=torch.einsum('ck,bkhw->bchw',self.Dm,am)
        nf=torch.einsum('ck,bkhw->bchw',self.Df,af)
        sw=F.softmax(self.scale_w(d1_hr),-1).unsqueeze(-1).unsqueeze(-1)
        null_comp=sw[:,0:1]*ng+sw[:,1:2]*nm+sw[:,2:3]*nf
        null_comp=self.projector.project_null(null_comp,(H_hr,W_hr))
        wf=self.wavelet(null_comp)
        wf=wf*self.wavelet_gate(torch.cat([null_comp,wf],1))
        out=refined+null_comp+wf
        consist=self.consist(torch.cat([ag,am,af],1))
        return {'out':out,'base':base,'refined':refined,'null':null_comp,'wf':wf,'consist':consist}

# ---- Dataset ----
class ChikuseiDS(Dataset):
    def __init__(self, root, split='train', bands=128, scale=4, patch=64):
        self.split=split; self.bands=bands; self.scale=scale; self.patch=patch
        self.srf=chikusei_srf(bands); self.kernel=gaussian_kernel2d(9,1.2)
        # Find .mat files â€” prefer the large HSI cube, skip Ground_Truth
        mat_files=glob.glob(os.path.join(root,'**','*.mat'),recursive=True)
        hsi=[m for m in mat_files if 'Ground_Truth' not in os.path.basename(m) and 'gt' not in os.path.basename(m).lower()]
        if hsi:
            hsi.sort(key=lambda f: os.path.getsize(f), reverse=True)
            mat_path=hsi[0]
        elif mat_files:
            mat_files.sort(key=lambda f: os.path.getsize(f), reverse=True)
            mat_path=mat_files[0]
        else:
            raise FileNotFoundError(f'No .mat files found under {root}')
        print(f'[Data] Loading: {mat_path}')
        data=loadmat(mat_path)
        for key,val in data.items():
            if not key.startswith('__') and hasattr(val,'shape'):
                arr=np.array(val,dtype=np.float32)
                if arr.ndim==3 and min(arr.shape)>10:
                    if arr.shape[0]>arr.shape[-1]: arr=arr.transpose(2,0,1)
                    if arr.max()>1.0: arr=arr/arr.max()
                    self.cube=arr; break
        C,H,W=self.cube.shape; print(f'[Data] Cube: {C}b, {H}x{W}px')
        p=patch; coords=[(y,x) for y in range(0,H-p+1,p) for x in range(0,W-p+1,p)]
        random.seed(42); random.shuffle(coords)
        n=int(0.7*len(coords))
        self.patches=coords[:n] if split=='train' else coords[n:]
        print(f'[Data] {split}: {len(self.patches)} patches')
    def __len__(self): return len(self.patches)*(200 if self.split=='train' else 1)
    def _sim(self, gt):
        C,H,W=gt.shape; blurred=np.empty_like(gt)
        for c in range(C): blurred[c]=convolve(gt[c],self.kernel,mode='wrap')
        hr=H//self.scale; y0=(H-hr*self.scale)//2; x0=(W-hr*self.scale)//2
        lr=blurred[:,y0::self.scale,x0::self.scale].astype(np.float32)
        msi=np.einsum('chw,cm->mhw',gt,self.srf).astype(np.float32)
        return lr, np.clip(msi,0,1)
    def __getitem__(self, idx):
        y,x=self.patches[idx%len(self.patches)]; p=self.patch
        gt=self.cube[:,y:y+p,x:x+p].copy()
        if self.split=='train':
            if random.random()<0.5: gt=gt[:,:,::-1].copy()
            if random.random()<0.5: gt=gt[:,::-1,:].copy()
            if random.random()<0.5: gt=np.rot90(gt,random.randint(1,3),axes=(1,2)).copy()
            if random.random()<0.15: gt=(gt+np.random.randn(*gt.shape).astype(np.float32)*0.01).clip(0,1)
        lr,msi=self._sim(gt)
        return torch.from_numpy(gt), torch.from_numpy(lr), torch.from_numpy(msi)

# ---- Losses ----
def ssim_loss(pred, gt, ws=11):
    C1,C2=0.01**2,0.03**2
    coords=torch.arange(ws,dtype=pred.dtype,device=pred.device)-ws//2
    g=torch.exp(-(coords**2)/(2*1.5**2)); g=g/g.sum()
    w=(g.unsqueeze(0)*g.unsqueeze(1)).unsqueeze(0).unsqueeze(0).expand(pred.shape[1],1,-1,-1).contiguous()
    ch=pred.shape[1]
    mu1=F.conv2d(pred,w,padding=ws//2,groups=ch); mu2=F.conv2d(gt,w,padding=ws//2,groups=ch)
    s1=F.conv2d(pred*pred,w,padding=ws//2,groups=ch)-mu1*mu1
    s2=F.conv2d(gt*gt,w,padding=ws//2,groups=ch)-mu2*mu2
    s12=F.conv2d(pred*gt,w,padding=ws//2,groups=ch)-mu1*mu2
    return 1.0-((2*mu1*mu2+C1)*(2*s12+C2)/((mu1*mu1+mu2*mu2+C1)*(s1+s2+C2))).mean()

def sam_loss(pred, gt):
    p=pred.reshape(pred.shape[0],pred.shape[1],-1); g=gt.reshape(gt.shape[0],gt.shape[1],-1)
    p=p/(p.norm(dim=1,keepdim=True)+1e-8); g=g/(g.norm(dim=1,keepdim=True)+1e-8)
    return torch.acos(torch.clamp((p*g).sum(dim=1),-1+1e-7,1-1e-7)).mean()

def gradient_loss(pred, target):
    return F.l1_loss(pred[...,:,1:],target[...,:,1:])+F.l1_loss(pred[...,1:,:],target[...,1:,:])

def spectral_gradient_loss(pred, target):
    return F.l1_loss(pred[:,1:]-pred[:,:-1],target[:,1:]-target[:,:-1])

def total_loss(out, gt, yH, yM, model, w_l1=1.0, w_ssim=0.5, w_sam=0.05, w_phys=0.1, w_grad=0.1, w_sg=0.05):
    pred=out['out']
    l_l1=F.l1_loss(pred,gt); l_ssim=ssim_loss(pred,gt); l_sam=sam_loss(pred,gt)
    l_grad=gradient_loss(pred,gt); l_sg=spectral_gradient_loss(pred,gt)
    l_phys=F.mse_loss(model.projector.D(pred),yH)+F.mse_loss(torch.einsum('bchw,cm->bmhw',pred,model.srf),yM)
    return w_l1*l_l1+w_ssim*l_ssim+w_sam*l_sam+w_grad*l_grad+w_sg*l_sg+w_phys*l_phys, l_l1, l_ssim, l_sam

# ---- Metrics ----
def psnr_np(pred,gt):
    mse=np.mean((pred-gt)**2); return 100.0 if mse<1e-12 else -10*np.log10(mse)
def sam_np(pred,gt):
    p=pred.reshape(pred.shape[0],-1); g=gt.reshape(gt.shape[0],-1)
    p=p/(np.linalg.norm(p,axis=0,keepdims=True)+1e-8); g=g/(np.linalg.norm(g,axis=0,keepdims=True)+1e-8)
    return np.mean(np.arccos(np.clip((p*g).sum(0),-1,1)))*180/math.pi
def ssim_np(pred,gt):
    C1,C2=0.01**2,0.03**2
    mu1=uniform_filter(pred,3,mode='reflect'); mu2=uniform_filter(gt,3,mode='reflect')
    s12=uniform_filter(pred*gt,3,mode='reflect')-mu1*mu2
    s1=uniform_filter(pred**2,3,mode='reflect')-mu1**2; s2=uniform_filter(gt**2,3,mode='reflect')-mu2**2
    return np.mean(((2*mu1*mu2+C1)*(2*s12+C2))/((mu1**2+mu2**2+C1)*(s1+s2+C2)+1e-8))
def ergas_np(pred,gt,scale=4):
    C=pred.shape[0]; e=sum(((pred-gt)**2)[c].mean()/(gt[c].mean()**2+1e-8) for c in range(C))
    return math.sqrt(e/C)*100*scale

class EMA:
    def __init__(self,m,decay=0.999): self.d=decay; self.s={k:v.detach().clone() for k,v in m.state_dict().items()}
    def update(self,m):
        with torch.no_grad():
            for k,v in m.state_dict().items():
                if v.dtype.is_floating_point and k in self.s: self.s[k].mul_(self.d).add_(v.detach(),alpha=1-self.d)
    def apply(self,m): m.load_state_dict(self.s, strict=False)
    def restore(self,m): self.s={k:v.detach().clone() for k,v in m.state_dict().items()}

print('Library OK')
