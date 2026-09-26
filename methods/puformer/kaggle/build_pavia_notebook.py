"""Write the Kaggle notebook that trains PUFormer on Pavia Centre x4 (two runs in parallel, one per T4).

    python build_pavia_notebook.py --hours 10.4
    kaggle kernels push -p pavia

Run A trains from scratch (the comparable result); run B starts its transformer blocks from the
Chikusei v3 weights (transfer; the input / output convolutions differ and are trained from scratch).
Data: Kaggle dataset mlxlx0000/paviadata (Pavia.mat, 1096x715x102). The Chikusei v3 kernel output is
mounted for the transfer run's initial weights.
"""
import argparse
import json
import os

from build_notebook import HERE, code_blob, git_rev

SETUP = r'''# 0) setup: session clock, embedded code, helpers
import base64, glob, hashlib, io, json, os, shlex, subprocess, tarfile, threading, time
T0 = time.time()
DEADLINE = T0 + {hours} * 3600
CODE, OUT = '/kaggle/working/code', '/kaggle/working/out'
B64 = '{b64}'
raw = base64.b64decode(B64)
assert hashlib.sha256(raw).hexdigest() == '{sha}'
tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz').extractall(CODE)
os.makedirs(OUT, exist_ok=True)
print('code sha256 {sha} from git {rev}')


def run(cmd, env=None, prefix=''):
    """Run a shell command in CODE and stream its output; return the exit code."""
    p = subprocess.Popen(cmd, shell=True, cwd=CODE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env=dict(os.environ, **(env or {{}})))
    for line in p.stdout:
        print(prefix + line, end='', flush=True)
    rc = p.wait()
    print(f'{{prefix}}[exit {{rc}} after {{(time.time() - T0) / 3600:.2f}} h]', flush=True)
    return rc


subprocess.run('nvidia-smi --query-gpu=index,name,memory.total --format=csv; free -g | head -2; '
               r'find /kaggle/input -maxdepth 6 \( -name "*.mat" -o -name "best_ema.pt" \) | head', shell=True)
MAT = sorted(glob.glob('/kaggle/input/**/Pavia.mat', recursive=True))[0]
INIT = [p for p in glob.glob('/kaggle/input/**/best_ema.pt', recursive=True) if 'puformer' in p][0]
print('Pavia:', MAT, '| transfer init:', INIT)
'''

CHECK = r'''# 1) self-checks and a short CPU smoke run of the Pavia pipeline
assert run('python selfcheck.py') == 0
assert run('CUDA_VISIBLE_DEVICES= python train.py --dataset pavia --mat x --smoke --iters 6 --eval_every 3 '
           '--width 16 --stages 2 --bs 2 --amp 0 --q2n 1 --out /tmp/smoke') == 0
'''

TRAIN = r'''# 2) two runs in parallel until the deadline: A from scratch (GPU 0), B transfer from Chikusei (GPU 1)
COMMON = (f'python train.py --dataset pavia --mat {{shlex.quote(MAT)}} --pad reflect --bs 8 --warmup 1000 --w_sam 0.05 '
          f'--w_ssim 0.1 --eval_every 1000 --log_every 1000 --deadline {{DEADLINE}}')
jobs = {{'A': (f'{{COMMON}} --lr 3e-4 --out {{OUT}}/pavia_scratch', '0'),
        'B': (f'{{COMMON}} --lr 1.5e-4 --init_ckpt {{shlex.quote(INIT)}} --init_partial 1 --out {{OUT}}/pavia_transfer', '1')}}
codes = {{}}
threads = [threading.Thread(target=lambda k=k, c=c, g=g: codes.__setitem__(k, run(c, {{'CUDA_VISIBLE_DEVICES': g}}, f'[{{k}}] ')))
           for k, (c, g) in jobs.items()]
[t.start() for t in threads]
[t.join() for t in threads]
print('exit codes:', codes)
'''

SUMMARY = r'''# 3) summary against TIP'26 Table IV (Pavia Center, reported values)
tip = dict(PSNR=47.1597, SSIM=0.9972, SAM=1.4542, ERGAS=0.8262, Q2n=0.9980, CC=0.9985, SCC=0.9976, RMSE_DN=35.7086)
for run_name in ('pavia_scratch', 'pavia_transfer'):
    f = f'{OUT}/{run_name}/results.json'
    if not os.path.exists(f):
        print(run_name, 'no results'); continue
    r = json.load(open(f))
    print(run_name, {k: r[k] for k in ('iters', 'epochs', 'best_val_PSNR', 'train_hours', 'dn_scale') if k in r})
    for k in ('test', 'test_tta', 'gsa_test', 'bicubic_test'):
        print(' ', k, {m: round(v, 4) for m, v in r[k].items()})
    t = r['test_tta']
    print('  vs TIP26 best:', {m: ('WIN' if (t[m] < v if m in ('SAM', 'ERGAS', 'RMSE_DN') else t[m] > v) else 'lose') + f' {t[m]:.4f}/{v}'
                                for m, v in tip.items() if m in t}, '| SSIM_psrt', round(t['SSIM_psrt'], 4))
subprocess.run(f'ls -la {OUT}/*; du -sh {OUT}', shell=True)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=10.4)
    ap.add_argument("--slug", default="puformer-pavia-x4")
    ap.add_argument("--user", default="amarnath10chinu")
    ap.add_argument("--init_kernel", default="amarnath10chinu/puformer-chikusei-x4-v3")
    a = ap.parse_args()
    b64, sha = code_blob()
    out = os.path.join(HERE, "pavia")
    os.makedirs(out, exist_ok=True)
    srcs = [SETUP.format(hours=a.hours, b64=b64, sha=sha, rev=git_rev()), CHECK, TRAIN.format(), SUMMARY]
    nb = dict(nbformat=4, nbformat_minor=5,
              metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python"),
                            language_info=dict(name="python")),
              cells=[dict(cell_type="code", metadata={}, execution_count=None, outputs=[], id=f"c{i}", source=s)
                     for i, s in enumerate(srcs)])
    name = f"{a.slug}.ipynb"
    with open(os.path.join(out, name), "w", newline="\n") as f:
        json.dump(nb, f, indent=1)
    meta = dict(id=f"{a.user}/{a.slug}", title="PUFormer Pavia x4", code_file=name, language="python",
                kernel_type="notebook", is_private=True, enable_gpu=True, enable_internet=True,
                machine_shape="NvidiaTeslaT4", dataset_sources=["mlxlx0000/paviadata"], competition_sources=[],
                kernel_sources=[a.init_kernel])
    with open(os.path.join(out, "kernel-metadata.json"), "w", newline="\n") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote pavia/{name} (sha256 {sha[:12]}) for {meta['id']}")


if __name__ == "__main__":
    main()
