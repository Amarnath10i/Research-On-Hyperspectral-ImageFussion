import sys, torch
sys.path.insert(0, 'current/experiments/scripts')
from train_nullfusion_pp_v2 import NullFusionPlusV2

for w in [24, 28, 32, 36]:
    m = NullFusionPlusV2(bands=128, msi=3, width=w, scale=4)
    n = sum(p.numel() for p in m.parameters())
    print(f'width={w:2d}: {n/1e6:.2f}M params')

print()
model = NullFusionPlusV2(bands=128, msi=3, width=32, scale=4)
yH = torch.randn(2, 128, 16, 16, requires_grad=True)
yM = torch.randn(2, 3, 64, 64, requires_grad=True)
out = model(yH, yM)
print(f"Output: {tuple(out['out'].shape)}")
print(f"Base:   {tuple(out['base'].shape)}")
print(f"Null:   {tuple(out['null'].shape)}")
print(f"Wave:   {tuple(out['wf'].shape)}")
out['out'].mean().backward()
print('Forward+Backward: OK')
has_grad = sum(1 for p in model.parameters() if p.grad is not None)
total = sum(1 for p in model.parameters())
print(f'Gradients: {has_grad}/{total} params have grads')
