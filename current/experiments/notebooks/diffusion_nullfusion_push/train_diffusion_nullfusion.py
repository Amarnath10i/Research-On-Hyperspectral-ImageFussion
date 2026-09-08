"""Diffusion-NullFusion â€” SOTA-beater for Chikusei x4."""
from __future__ import annotations
import argparse, glob, json, math, os, random, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.io import loadmat
from scipy.ndimage import convolve, uniform_filter
import h5py
from torch.utils.data import Dataset

def chikusei_srf(bands=128):
    wl=np.linspace(363.0,1018.0,128)
    raw=np.stack([np.exp(-((wl-620)**2)/(2*80**2)),np.exp(-((wl-540)**2)/(2*70**2)),np.exp(-((wl-460)**2)/(2*60**2))],axis=1).astype(np.float32)
    return raw/np.maximum(raw.sum(axis=0,keepdims=True),1e-8)

def gaussian_kernel2d(size=9,sigma=1.2):
    ax=np.arange(size,dtype=np.float32)-(size-1)/2; xx,yy=np.meshgrid(ax,ax)
    k=np.exp(-0.5*(xx**2+yy**2)/sigma**2); return (k/k.sum()).astype(np.float32)

class DegradationOp(nn.Module):
    def __init__(self,scale,ksize=9,sigma=1.2):
        super().__init__(); self.scale=scale
        k=gaussian_kernel2d(ksize,sigma)
        self.register_buffer('k',torch.from_numpy(k)[None,None])
    def forward(self,x):
        C=x.shape[1]
        return F.conv2d(x,self.k.repeat(C,1,1,1),padding=4,groups=C)[:,:,::self.scale,::self.scale]
    def transpose(self,y,hw):
        C=y.shape[1]; up=F.interpolate(y,size=hw,mode='bicubic',align_corners=False)
        return F.conv2d(up,self.k.repeat(C,1,1,1),padding=4,groups=C)

def scalar_cg(applyA,rhs,steps,tol=1e-10):
    z=torch.zeros_like(rhs); r=rhs-applyA(z); p=r.clone()
    rs=(r*r).flatten(1).sum(1)
    for _ in range(steps):
        ap=applyA(p); denom=(p*ap).flatten(1).sum(1)
        alpha=(rs/denom.clamp_min(tol)).reshape(-1,1,1,1)
        z=z+alpha*p; r=r-alpha*ap
        rs_new=(r*r).flatten(1).sum(1)
        beta=(rs_new/rs.clamp_min(tol)).reshape(-1,1,1,1)
        p=r+beta*p; rs=rs_new
    return z

class RangeNullProjector(nn.Module):
    def __init__(self,scale,cg_steps=40,ridge=1e-4):
        super().__init__(); self.D=DegradationOp(scale); self.scale=scale
        self.cg_steps=cg_steps; self.ridge=ridge
    def _normal_op(self,out_hw):
        def applyA(z): return self.D(self.D.transpose(z,out_hw))+self.ridge*z
        return applyA
    def pinv(self,yH,out_hw):
        return self.D.transpose(scalar_cg(self._normal_op(out_hw),yH,self.cg_steps),out_hw)
    def project_null(self,v,out_hw=None):
        if out_hw is None: out_hw=(v.shape[-2],v.shape[-1])
        return v-self.pinv(self.D(v),out_hw)

class WindowAttention(nn.Module):
    def __init__(self,dim,num_heads,window_size=8):
        super().__init__(); self.dim=dim; self.num_heads=num_heads
        self.head_dim=dim//num_heads; self.scale=self.head_dim**-0.5
        self.qkv=nn.Linear(dim,dim*3); self.proj=nn.Linear(dim,dim)
        self.window_size=window_size
    def forward(self,x):
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
    def __init__(self,dim,cond_dim):
        super().__init__(); self.norm=nn.LayerNorm(dim,elementwise_affine=False)
        self.proj=nn.Linear(cond_dim,dim*2)
    def forward(self,x,cond):
        gamma,beta=self.proj(cond).unsqueeze(1).chunk(2,dim=-1)
        return self.norm(x)*(1+gamma)+beta

class SwinBlock(nn.Module):
    def __init__(self,dim,num_heads,cond_dim,window_size=8,mlp_ratio=4.0):
        super().__init__()
        self.norm1=AdaLN(dim,cond_dim); self.attn=WindowAttention(dim,num_heads,window_size)
        self.norm2=AdaLN(dim,cond_dim)
        self.mlp=nn.Sequential(nn.Linear(dim,int(dim*mlp_ratio)),nn.GELU(),nn.Linear(int(dim*mlp_ratio),dim))
    def forward(self,x,cond):
        x=x+self.attn(self.norm1(x,cond)); x=x+self.mlp(self.norm2(x,cond)); return x

class PatchMerging(nn.Module):
    def __init__(self,dim):
        super().__init__(); self.reduction=nn.Linear(4*dim,2*dim,bias=False); self.norm=nn.LayerNorm(4*dim)
    def forward(self,x,H,W):
        B=x.shape[0]; x=x.reshape(B,H,W,-1)
        x0,x1,x2,x3=x[:,0::2,0::2],x[:,1::2,0::2],x[:,0::2,1::2],x[:,1::2,1::2]
        x=torch.cat([x0,x1,x2,x3],-1).reshape(B,-1,4*x.shape[-1])
        return self.reduction(self.norm(x)),H//2,W//2

class PatchExpanding(nn.Module):
    def __init__(self,dim):
        super().__init__(); self.linear=nn.Linear(dim,4*dim); self.norm=nn.LayerNorm(dim)
    def forward(self,x,H,W):
        x=self.linear(self.norm(x)).reshape(x.shape[0],H,W,2,2,-1).permute(0,1,3,2,4,5).reshape(x.shape[0],H*2,W*2,-1)
        return x.reshape(x.shape[0],H*2*W*2,-1),H*2,W*2

class ChannelProject(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__(); self.proj=nn.Linear(in_dim,out_dim)
    def forward(self,x): return self.proj(x)

class MultiScaleSwinUNet(nn.Module):
    def __init__(self,in_ch=128,base_dim=48,num_heads=None,cond_dim=64,window_size=8,depths=None):
        super().__init__()
        if num_heads is None: num_heads=[4,8,16]
        if depths is None: depths=[1,1,2,1,1]
        dims=[base_dim,base_dim*2,base_dim*4]
        self.input_proj=nn.Linear(in_ch,dims[0])
        self.enc1=nn.ModuleList([SwinBlock(dims[0],num_heads[0],cond_dim,window_size) for _ in range(depths[0])])
        self.merge1=PatchMerging(dims[0])
        self.enc2=nn.ModuleList([SwinBlock(dims[1],num_heads[1],cond_dim,window_size) for _ in range(depths[1])])
        self.merge2=PatchMerging(dims[1])
        self.bottleneck=nn.ModuleList([SwinBlock(dims[2],num_heads[2],cond_dim,window_size) for _ in range(depths[2])])
        self.expand2=PatchExpanding(dims[2]); self.proj_skip2=ChannelProject(dims[1],dims[2])
        self.dec2=nn.ModuleList([SwinBlock(dims[1],num_heads[1],cond_dim,window_size) for _ in range(depths[3])])
        self.expand1=PatchExpanding(dims[1]); self.proj_skip1=ChannelProject(dims[0],dims[1])
        self.dec1=nn.ModuleList([SwinBlock(dims[0],num_heads[0],cond_dim,window_size) for _ in range(depths[4])])
        self.output_proj=nn.Linear(dims[0],in_ch)
    def forward(self,x,cond):
        B,C,H,W=x.shape; x=x.reshape(B,C,H*W).permute(0,2,1)
        x=self.input_proj(x)
        for b in self.enc1: x=b(x,cond)
        skip1=x; x,H1,W1=self.merge1(x,H,W)
        for b in self.enc2: x=b(x,cond)
        skip2=x; x,H2,W2=self.merge2(x,H1,W1)
        for b in self.bottleneck: x=b(x,cond)
        x,H2,W2=self.expand2(x,H2,W2); x=x+self.proj_skip2(skip2)
        for b in self.dec2: x=b(x,cond)
        x,H1,W1=self.expand1(x,H1,W1); x=x+self.proj_skip1(skip1)
        for b in self.dec1: x=b(x,cond)
        return self.output_proj(x).reshape(B,H,W,-1).permute(0,3,1,2)

class CosineSchedule:
    def __init__(self,T=1000,s=0.008):
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
        x0_pred=((x_t-torch.sqrt(1-ab_t)*eps_pred)/torch.sqrt(ab_t)).clamp(0,1)
        return torch.sqrt(ab_p)*x0_pred+torch.sqrt(1-ab_p)*eps_pred

class SensorEmbedding(nn.Module):
    def __init__(self,srf_matrix,out_dim=64):
        super().__init__(); flat=torch.from_numpy(srf_matrix).float().flatten()
        self.register_buffer('srf_flat',flat)
        self.mlp=nn.Sequential(nn.Linear(flat.numel(),128),nn.GELU(),nn.Linear(128,out_dim))
    def forward(self): return self.mlp(self.srf_flat.unsqueeze(0)).squeeze(0)

class DiffusionNullFusion(nn.Module):
    def __init__(self,bands=128,msi=3,base_dim=48,scale=4,cond_dim=64,T=1000,num_heads=None,depths=None,window_size=8):
        super().__init__(); self.bands=bands; self.scale=scale; self.T=T
        srf=torch.from_numpy(chikusei_srf(bands)).float()
        self.register_buffer('srf',srf); self.register_buffer('srfinv',torch.linalg.pinv(srf))
        self.projector=RangeNullProjector(scale,cg_steps=8,ridge=1e-4)
        self.sensor_embed=SensorEmbedding(chikusei_srf(bands),cond_dim)
        self.schedule=CosineSchedule(T)
        self.cond_proj=nn.Sequential(nn.Conv2d(msi+bands,cond_dim,1),nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.time_mlp=nn.Sequential(nn.Linear(1,cond_dim),nn.GELU(),nn.Linear(cond_dim,cond_dim))
        self.unet=MultiScaleSwinUNet(in_ch=bands,base_dim=base_dim,num_heads=num_heads,cond_dim=cond_dim,window_size=window_size,depths=depths)
    def _conditioning(self,yH,yM,H_hr,W_hr):
        base=self.projector.pinv(yH,(H_hr,W_hr))
        obs=F.interpolate(yH,(H_hr,W_hr),mode='bicubic',align_corners=False)
        cond=self.cond_proj(torch.cat([yM,obs],1))+self.sensor_embed()
        return cond,base
    def forward(self,x_t,t,cond):
        return self.unet(x_t,cond+self.time_mlp(t.float().unsqueeze(-1)))
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
        avg_null=self.projector.project_null(torch.mean(torch.stack(samples),0),(H_hr,W_hr))
        return {'out':base+avg_null,'base':base,'null':avg_null}

class ChikuseiDS(Dataset):
    def __init__(self,root,split='train',bands=128,scale=4,patch=64):
        self.split=split; self.bands=bands; self.scale=scale; self.patch=patch
        self.srf=chikusei_srf(bands); self.kernel=gaussian_kernel2d(9,1.2)
        mat_files=glob.glob(os.path.join(root,'**','*.mat'),recursive=True)
        hsi=[m for m in mat_files if 'Ground_Truth' not in os.path.basename(m) and 'gt' not in os.path.basename(m).lower()]
        if hsi: hsi.sort(key=lambda f:os.path.getsize(f),reverse=True); mat_path=hsi[0]
        elif mat_files: mat_files.sort(key=lambda f:os.path.getsize(f),reverse=True); mat_path=mat_files[0]
        else: raise FileNotFoundError(f'No .mat files under {root}')
        print(f'[Data] Loading: {mat_path}')
        try: data=loadmat(mat_path)
        except NotImplementedError:
            print('[Data] v7.3 mat file - using h5py'); data={}
            with h5py.File(mat_path,'r') as f:
                for key in f.keys():
                    if not key.startswith('__'):
                        val=np.array(f[key]).copy()
                        if val.ndim>=2: data[key]=val
        for key,val in data.items():
            if not key.startswith('__') and hasattr(val,'shape'):
                arr=np.array(val,dtype=np.float32)
                if arr.ndim==3 and min(arr.shape)>10:
                    if arr.shape[0]>arr.shape[-1]: arr=arr.transpose(2,0,1)
                    if arr.max()>1.0: arr=arr/arr.max()
                    self.cube=arr; break
        C,H,W=self.cube.shape; print(f'[Data] Cube: {C}b, {H}x{W}px')
        p=patch; coords=[(y,x) for y in range(0,H-p+1,p) for x in range(0,W-p+1,p)]
        random.seed(42); random.shuffle(coords); n=int(0.7*len(coords))
        self.patches=coords[:n] if split=='train' else coords[n:]
        print(f'[Data] {split}: {len(self.patches)} patches')
    def __len__(self): return len(self.patches)*(200 if self.split=='train' else 1)
    def _sim(self,gt):
        C,H,W=gt.shape; blurred=np.empty_like(gt)
        for c in range(C): blurred[c]=convolve(gt[c],self.kernel,mode='wrap')
        hr=H//self.scale; y0=(H-hr*self.scale)//2; x0=(W-hr*self.scale)//2
        lr=blurred[:,y0::self.scale,x0::self.scale].astype(np.float32)
        msi=np.einsum('chw,cm->mhw',gt,self.srf).astype(np.float32)
        return lr,np.clip(msi,0,1)
    def __getitem__(self,idx):
        y,x=self.patches[idx%len(self.patches)]; p=self.patch
        gt=self.cube[:,y:y+p,x:x+p].copy()
        if self.split=='train':
            if random.random()<0.5: gt=gt[:,:,::-1].copy()
            if random.random()<0.5: gt=gt[:,::-1,:].copy()
            if random.random()<0.5: gt=np.rot90(gt,random.randint(1,3),axes=(1,2)).copy()
            if random.random()<0.15: gt=(gt+np.random.randn(*gt.shape).astype(np.float32)*0.01).clip(0,1)
        lr,msi=self._sim(gt)
        return torch.from_numpy(gt),torch.from_numpy(lr),torch.from_numpy(msi)

def diffusion_loss(model,x0,cond,schedule):
    B=x0.shape[0]; t=torch.randint(0,schedule.T,(B,),device=x0.device)
    noise=torch.randn_like(x0); x_t=schedule.add_noise(x0,noise,t)
    return F.mse_loss(model(x_t,t,cond),noise)

def physics_loss(pred,yH,yM,model):
    return F.mse_loss(model.projector.D(pred),yH)+F.mse_loss(torch.einsum('bchw,cm->bmhw',pred,model.srf),yM)

def total_loss(model,gt,yH,yM,schedule,w_noise=1.0,w_phys=0.1):
    H_hr,W_hr=yM.shape[-2],yM.shape[-1]; cond,base=model._conditioning(yH,yM,H_hr,W_hr)
    x0=model.projector.project_null(gt-base,(H_hr,W_hr))
    l_noise=diffusion_loss(model,x0,cond,schedule)
    l_phys=physics_loss(base+x0,yH,yM,model)
    return w_noise*l_noise+w_phys*l_phys,l_noise,l_phys

def psnr_np(pred,gold):
    mse=np.mean((pred-gold)**2); return 100.0 if mse<1e-12 else -10*np.log10(mse)
def sam_np(pred,gold):
    p=pred.reshape(pred.shape[0],-1); g=gold.reshape(gold.shape[0],-1)
    p=p/(np.linalg.norm(p,axis=0,keepdims=True)+1e-8); g=g/(np.linalg.norm(g,axis=0,keepdims=True)+1e-8)
    return np.mean(np.arccos(np.clip((p*g).sum(0),-1,1)))*180/math.pi
def ssim_np(pred,gold):
    C1,C2=0.01**2,0.03**2; mu1=uniform_filter(pred,3,mode='reflect'); mu2=uniform_filter(gold,3,mode='reflect')
    s12=uniform_filter(pred*gold,3,mode='reflect')-mu1*mu2
    s1=uniform_filter(pred**2,3,mode='reflect')-mu1**2; s2=uniform_filter(gold**2,3,mode='reflect')-mu2**2
    return np.mean(((2*mu1*mu2+C1)*(2*s12+C2))/((mu1**2+mu2**2+C1)*(s1+s2+C2)+1e-8))
def ergas_np(pred,gold,scale=4):
    C=pred.shape[0]; e=sum(((pred-gold)**2)[c].mean()/(gold[c].mean()**2+1e-8) for c in range(C))
    return math.sqrt(e/C)*100*scale

class EMA:
    def __init__(self,m,d=0.999): self.d=d; self.s={k:v.detach().clone() for k,v in m.state_dict().items()}
    def update(self,m):
        with torch.no_grad():
            for k,v in m.state_dict().items():
                if v.dtype.is_floating_point and k in self.s: self.s[k].mul_(self.d).add_(v.detach(),alpha=1-self.d)
    def apply(self,m): m.load_state_dict(self.s,strict=False)
    def restore(self,m): self.s={k:v.detach().clone() for k,v in m.state_dict().items()}

print('Library OK')
