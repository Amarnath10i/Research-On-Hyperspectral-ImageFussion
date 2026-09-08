import sys, torch, time
sys.path.insert(0, 'current/experiments/scripts')
from train_nullfusion_pp import NullFusionPlus as V1
from train_nullfusion_pp_v2 import NullFusionPlusV2 as V2

for name, cls in [('v1', V1), ('v2', V2)]:
    m = cls(bands=128, msi=3, width=32, scale=4)
    n = sum(p.numel() for p in m.parameters())
    yH = torch.randn(1, 128, 16, 16)
    yM = torch.randn(1, 3, 64, 64)
    t0 = time.time()
    with torch.no_grad():
        out = m(yH, yM)
    dt = time.time() - t0
    sh = tuple(out['out'].shape)
    print(f"{name}: {n/1e6:.2f}M params, {dt*1000:.0f}ms, output {sh}")
