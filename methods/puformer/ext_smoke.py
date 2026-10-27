"""Check the adapters of the published networks: build, forward, backward and output shape at our band counts.

    python ext_smoke.py smgu clsnet otpnet u2net mimosst

Every model runs in its own process (several repositories have a top-level package called `model`).
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = [("chikusei", 128, 8), ("pavia", 92, 4)]

ONE = r'''
import json, sys, time, torch
sys.path.insert(0, {here!r})
from hsifuse.external import build_external
name, bands, msi, scale = {name!r}, {bands}, {msi}, {scale}
torch.manual_seed(0)
net = build_external(name, bands=bands, msi=msi, scale=scale)
params = sum(p.numel() for p in net.parameters()) / 1e6
net.train()
yh, ym = torch.rand(2, bands, 64 // scale, 64 // scale), torch.rand(2, msi, 64, 64)
t = time.time()
x = net(yh, ym, return_all=True)[-1]
assert x.shape == (2, bands, 64, 64), x.shape
x.abs().mean().backward()
grads = sum(p.grad is not None for p in net.parameters())
train_s = time.time() - t
net.eval()
with torch.no_grad():
    t = time.time()
    y = net(torch.rand(1, bands, 128 // scale, 128 // scale), torch.rand(1, msi, 128, 128))
    assert y.shape == (1, bands, 128, 128), y.shape
    assert torch.isfinite(y).all()
print(json.dumps(dict(params_M=round(params, 3), grads=grads, n_params=len(list(net.parameters())),
                      train_step_s=round(train_s, 2), test128_s=round(time.time() - t, 2))))
'''

if __name__ == "__main__":
    names = sys.argv[1:]
    report = {}
    for name in names:
        for ds, bands, msi in CASES:
            code = ONE.format(here=HERE, name=name, bands=bands, msi=msi, scale=4)
            t = time.time()
            r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
            key = f"{name}-{ds}"
            if r.returncode == 0:
                report[key] = json.loads(r.stdout.strip().splitlines()[-1])
                print(key, "OK", report[key], flush=True)
            else:
                report[key] = dict(error=r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "?")
                print(key, "FAILED after", round(time.time() - t, 1), "s\n", r.stdout[-2000:], r.stderr[-4000:], flush=True)
    out = os.environ.get("OUT", ".")
    with open(os.path.join(out, "ext_smoke.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    sys.exit(0 if all("error" not in v for v in report.values()) else 1)
