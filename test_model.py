import sys, torch
sys.path.insert(0, 'current/common')
sys.path.insert(0, 'current/experiments/scripts')
from train_nullfusion_pp_v2 import NullFusionPlusV2

model = NullFusionPlusV2(bands=128, msi=3, width=32, scale=4)
n = sum(p.numel() for p in model.parameters())
print(f"Params: {n/1e6:.2f}M")

yH = torch.randn(1, 128, 16, 16, requires_grad=True)
yM = torch.randn(1, 3, 64, 64, requires_grad=True)
out = model(yH, yM)
print(f"Output: {tuple(out['out'].shape)}")
out['out'].mean().backward()
print("Forward+Backward: OK")
