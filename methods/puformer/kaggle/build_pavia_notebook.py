"""Write the all-in-one Kaggle notebook for PUFormer on Pavia Centre x4: train, test and analyse in one session.

    python build_pavia_notebook.py --hours 10.2
    kaggle kernels push -p pavia

One model, trained from scratch on Pavia only, on both T4s (DistributedDataParallel), with checkpoints:
last.pt and best_ema.pt at every evaluation, and an EMA snapshot every --snap_epochs epochs. If
training dies, it resumes on one GPU from last.pt. Then, in the same notebook: the test report
(test.py), the robustness study (eval_gaps.py) and the paper analysis (paper_analysis.py).
Data: Kaggle dataset mlxlx0000/paviadata (Pavia.mat, 1096x715x102).
"""
import argparse
import json
import os

from build_notebook import HERE, code_blob, git_rev

SETUP = r'''# 0) setup: session clock, embedded code, helpers
import base64, glob, hashlib, io, json, os, shlex, subprocess, tarfile, time
T0 = time.time()
DEADLINE = T0 + {hours} * 3600       # training stops here
LIMIT = T0 + 11.6 * 3600             # nothing optional starts after this (Kaggle stops at 12 h)
CODE, OUT = '/kaggle/working/code', '/kaggle/working/out/pavia'
B64 = '{b64}'
raw = base64.b64decode(B64)
assert hashlib.sha256(raw).hexdigest() == '{sha}'
tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz').extractall(CODE)
os.makedirs(OUT, exist_ok=True)
print('code sha256 {sha} from git {rev}')


def run(cmd, env=None):
    """Run a shell command in CODE and stream its output; return the exit code."""
    p = subprocess.Popen(cmd, shell=True, cwd=CODE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env=dict(os.environ, **(env or {{}})))
    for line in p.stdout:
        print(line, end='', flush=True)
    rc = p.wait()
    print(f'[exit {{rc}} after {{(time.time() - T0) / 3600:.2f}} h]', flush=True)
    return rc


subprocess.run('nvidia-smi --query-gpu=index,name,memory.total --format=csv; free -g | head -2', shell=True)
MAT = shlex.quote(sorted(glob.glob('/kaggle/input/**/Pavia.mat', recursive=True))[0])
NGPU = int(subprocess.check_output('nvidia-smi -L | wc -l', shell=True))
ENV = {{'NCCL_P2P_DISABLE': '1', 'TORCH_NCCL_ASYNC_ERROR_HANDLING': '1', 'OMP_NUM_THREADS': '2'}}
DATA = f'--dataset pavia --mat {{MAT}}'
print('Pavia:', MAT, '| GPUs:', NGPU)
'''

CHECK = r'''# 1) self-checks and a short CPU smoke run of the whole Pavia pipeline
assert run('python selfcheck.py') == 0
assert run(f'CUDA_VISIBLE_DEVICES= python train.py --dataset pavia --mat x --smoke --iters 6 --eval_every 3 '
           f'--width 16 --stages 2 --bs 2 --amp 0 --out /tmp/smoke') == 0
'''

TRAIN = r'''# 2) training: one model from scratch on both GPUs, checkpoints every evaluation + EMA snapshots
TRAIN = (f'train.py {DATA} --pad reflect --bs 8 --lr 3e-4 --warmup 1000 --w_sam 0.05 --w_ssim 0.1 '
         f'--eval_every 1000 --log_every 1000 --snap_epochs 2000')
launch = f'torchrun --standalone --nproc_per_node {NGPU} ' if NGPU > 1 else 'python '
rc = run(f'{launch}{TRAIN} --deadline {DEADLINE} --out {OUT}', ENV)
if rc != 0 and os.path.exists(f'{OUT}/last.pt') and time.time() < DEADLINE - 1200:
    print('training exited with', rc, '- resuming on one GPU from last.pt until the deadline')
    rc = run(f'python {TRAIN} --sched time --deadline {DEADLINE} --out {OUT}', ENV)
print('checkpoints:', sorted(os.path.basename(p) for p in glob.glob(f'{OUT}/*.pt')))
'''

TEST = r'''# 3) testing: best-validation EMA weights, plain and self-ensemble, per image, Q2n / SCC, consistency
assert os.path.exists(f'{OUT}/best_ema.pt'), 'no checkpoint to test'
run(f'python test.py {DATA} --ckpt {OUT}/best_ema.pt --pad reflect --out {OUT}/test')
'''

ANALYSIS = r'''# 4) robustness study (blur / SRF mismatch with operator swap, noise) and 5) paper analysis
if time.time() < LIMIT - 1800:
    run(f'python eval_gaps.py {DATA} --ckpt {OUT}/best_ema.pt --pad reflect --out {OUT}')
if time.time() < LIMIT - 1200:
    run(f'python paper_analysis.py {DATA} --pred_dir {OUT}/test --out {OUT}/paper_analysis')
'''

SUMMARY = r'''# 6) summary against TIP'26 Table IV (Pavia Center, reported values)
tip = dict(PSNR=47.1597, SSIM=0.9972, SAM=1.4542, ERGAS=0.8262, Q2n=0.9980, CC=0.9985, SCC=0.9976, RMSE_DN=35.7086)
low = ('SAM', 'ERGAS', 'RMSE_DN')
for f in (f'{OUT}/results.json', f'{OUT}/test/results.json'):
    if not os.path.exists(f):
        print(f, 'missing'); continue
    r = json.load(open(f))
    print(f, {k: r[k] for k in ('iters', 'epochs', 'best_val_PSNR', 'train_hours', 'dn_scale') if k in r})
    for k in ('val', 'test', 'test_tta', 'gsa_test', 'bicubic_test', 'consistency_tta'):
        if k in r and r[k]:
            print(' ', k, {m: round(v, 4) for m, v in r[k].items()})
    t = r['test_tta']
    print('  vs TIP26 best:', {m: ('WIN' if (t[m] < v if m in low else t[m] > v) else 'lose') + f' {t[m]:.4f}/{v}'
                                for m, v in tip.items()}, '| SSIM_psrt', round(t['SSIM_psrt'], 4))
subprocess.run(f'ls -la {OUT} {OUT}/test {OUT}/paper_analysis; du -sh {OUT}', shell=True)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=10.2, help="training deadline after session start")
    ap.add_argument("--slug", default="puformer-pavia-x4")
    ap.add_argument("--user", default="amarnath10chinu")
    a = ap.parse_args()
    b64, sha = code_blob()
    out = os.path.join(HERE, "pavia")
    os.makedirs(out, exist_ok=True)
    srcs = [SETUP.format(hours=a.hours, b64=b64, sha=sha, rev=git_rev()), CHECK, TRAIN, TEST, ANALYSIS, SUMMARY]
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
                kernel_sources=[])
    with open(os.path.join(out, "kernel-metadata.json"), "w", newline="\n") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote pavia/{name} (sha256 {sha[:12]}) for {meta['id']}")


if __name__ == "__main__":
    main()
