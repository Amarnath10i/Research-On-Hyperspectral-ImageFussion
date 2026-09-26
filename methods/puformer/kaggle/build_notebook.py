"""Write the Kaggle notebook + kernel-metadata.json that fine-tunes PUFormer on Chikusei x4.

The PUFormer code (train.py, test.py, eval_gaps.py, selfcheck.py, hsifuse/) is embedded in the
notebook as a gzipped tarball, with its SHA-256 and the local git commit printed, so every run
records exactly the code it ran. Nothing has to be pushed to GitHub first.

Pipeline, all inside one 12 h Kaggle session (2 x T4):
  1. build the normalised crop cache (/tmp, not saved)
  2. score the warm-start checkpoint as it was trained (zero-padded physics) -> out/first_run
  3. speed probe: DDP on both GPUs vs one GPU (a hung or slow DDP falls back to one GPU)
  4. fine-tune until the deadline; if training dies, resume on one GPU from last.pt
  5. final test (plain + self-ensemble, Q2n, SCC, consistency), then the robustness study if time is left

    python build_notebook.py --slug puformer-chikusei-x4-v3 --hours 11.0
    # push (credentials from the environment only):  kaggle kernels push -p methods/puformer/kaggle
"""
import argparse
import base64
import hashlib
import io
import json
import os
import subprocess
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
FILES = ["train.py", "test.py", "eval_gaps.py", "selfcheck.py", "paper_analysis.py"] + \
        [f"hsifuse/{f}" for f in sorted(os.listdir(os.path.join(PKG, "hsifuse"))) if f.endswith(".py")]


def code_blob():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in FILES:
            data = open(os.path.join(PKG, f), "rb").read().replace(b"\r\n", b"\n")
            info = tarfile.TarInfo(f)
            info.size, info.mtime = len(data), 0
            tar.addfile(info, io.BytesIO(data))
    raw = buf.getvalue()
    return base64.b64encode(raw).decode(), hashlib.sha256(raw).hexdigest()


def git_rev():
    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PKG, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "."], cwd=PKG, text=True).strip()
        return rev + ("+local changes" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


SETUP = r'''# 0) setup: session clock, embedded code, environment
import base64, glob, hashlib, io, json, os, subprocess, tarfile, threading, time
T0 = time.time()
HOURS = {hours}                     # training stops this many hours after this cell starts
DEADLINE = T0 + HOURS * 3600
LIMIT = T0 + 11.6 * 3600            # nothing optional starts after this (Kaggle stops at 12 h)
CODE, OUT = '/kaggle/working/code', '/kaggle/working/out'
B64 = '{b64}'
raw = base64.b64decode(B64)
assert hashlib.sha256(raw).hexdigest() == '{sha}'
tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz').extractall(CODE)
os.makedirs(OUT, exist_ok=True)
print('code sha256 {sha} from git {rev}:', sorted(os.listdir(CODE)))


def run(cmd, timeout=None, env=None):
    """Run a shell command in CODE, stream its output, return (exit code, output). A timeout kills it."""
    p = subprocess.Popen(cmd, shell=True, cwd=CODE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env=dict(os.environ, **(env or {{}})), start_new_session=True)
    timer = threading.Timer(timeout, lambda: os.killpg(p.pid, 9)) if timeout else None
    if timer:
        timer.start()
    out = []
    for line in p.stdout:
        print(line, end='', flush=True)
        out.append(line)
    rc = p.wait()
    if timer:
        timer.cancel()
    print(f'[exit {{rc}} after {{(time.time() - T0) / 3600:.2f}} h of the session]', flush=True)
    return rc, ''.join(out)


run('nvidia-smi --query-gpu=index,name,memory.total --format=csv; free -g | head -2; nproc; '
    'python -c "import torch; print(torch.__version__, torch.cuda.device_count())"; find /kaggle/input -maxdepth 5 | head -20')
INIT = sorted(glob.glob('/kaggle/input/**/best_ema.pt', recursive=True))[0]
COMMON = '--mat /kaggle/input --cache /tmp/chikusei_crop.npy'
NGPU = int(subprocess.check_output('nvidia-smi -L | wc -l', shell=True))
ENV = {{'NCCL_P2P_DISABLE': '1', 'TORCH_NCCL_ASYNC_ERROR_HANDLING': '1', 'OMP_NUM_THREADS': '2'}}
print('warm start:', INIT, '| GPUs:', NGPU, '| deadline in', HOURS, 'h')
'''

CACHE = r'''# 1) normalised 2048x2048x128 crop -> /tmp cache (shared by every later step) + operator / metric self-checks
rc, _ = run('python -c "import time; t = time.time(); from hsifuse.data import find_mat, load_chikusei; '
            'c = load_chikusei(find_mat(\'/kaggle/input\'), \'/tmp/chikusei_crop.npy\'); '
            'print(c.shape, c.dtype, round(time.time() - t), \'s\')"')
assert rc == 0, 'data cache failed'
run('python selfcheck.py')
'''

BASELINE = r'''# 2) the warm-start checkpoint exactly as trained (zero-padded physics): full metrics incl. SSIM_psrt, Q2n, SCC
run(f'python test.py {COMMON} --ckpt {INIT} --pad zeros --out {OUT}/first_run')
'''

PROBE = r'''# 3) speed probe: DDP on all GPUs vs one GPU (identical training step). A hung / failing DDP falls back to one GPU.
import re
TRAIN = f'train.py {{COMMON}} --init_ckpt {{INIT}} {train_args}'
rate = {{}}
if NGPU > 1:
    rc, out = run(f'torchrun --standalone --nproc_per_node {{NGPU}} {{TRAIN}} --probe 150 --out /tmp/probe', timeout=600, env=ENV)
    m = re.search(r'PROBE ([0-9.]+) samples/s', out)
    rate[NGPU] = float(m.group(1)) if rc == 0 and m else 0.0
rc, out = run(f'python {{TRAIN}} --probe 150 --out /tmp/probe', timeout=600, env=ENV)
m = re.search(r'PROBE ([0-9.]+) samples/s', out)
rate[1] = float(m.group(1)) if rc == 0 and m else 0.0
USE_DDP = NGPU > 1 and rate[NGPU] > 1.25 * rate[1]
print('samples/s:', rate, '-> DDP' if USE_DDP else '-> one GPU')
'''

TRAIN = r'''# 4) fine-tune until the deadline (checkpoints every eval: last.pt, best_ema.pt). If it dies, resume on one GPU.
LOG = f'{OUT}/puformer'
launch = f'torchrun --standalone --nproc_per_node {NGPU} ' if USE_DDP else 'python '
rc, _ = run(f'{launch}{TRAIN} --deadline {DEADLINE} --out {LOG}', env=ENV)
if rc != 0 and os.path.exists(f'{LOG}/last.pt') and time.time() < DEADLINE - 1200:
    print('training exited with', rc, '- resuming on one GPU until the deadline')
    rc, _ = run(f'python {TRAIN} --sched time --deadline {DEADLINE} --out {LOG}', env=ENV)
if not os.path.exists(f'{LOG}/results.json') and os.path.exists(f'{LOG}/best_ema.pt'):
    run(f'python test.py {COMMON} --ckpt {LOG}/best_ema.pt --pad reflect --out {LOG}/final_test')
'''

GAPS = r'''# 5) robustness study (blur / SRF mismatch with operator swap, noise, consistency), only if time is left
if os.path.exists(f'{OUT}/puformer/best_ema.pt') and time.time() < LIMIT - 1800:
    run(f'python eval_gaps.py {COMMON} --ckpt {OUT}/puformer/best_ema.pt --pad reflect --out {OUT}/puformer',
        timeout=max(60, LIMIT - time.time() - 300))
'''

SUMMARY = r'''# 6) summary
for f in sorted(glob.glob(f'{OUT}/**/results.json', recursive=True)):
    r = json.load(open(f))
    print(f, {k: r[k] for k in ('iters', 'epochs', 'gpus', 'best_val_PSNR', 'train_hours') if k in r})
    for k in ('val', 'test', 'test_tta', 'consistency', 'consistency_tta'):
        if k in r:
            print(' ', k, {m: round(v, 4) for m, v in r[k].items()})
run(f'ls -la {OUT}/*; du -sh {OUT}')
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=11.0, help="training deadline after session start")
    ap.add_argument("--train_args", default="--pad reflect --bs 8 --lr 1.5e-4 --warmup 1000 --w_sam 0.05 "
                                            "--w_ssim 0.1 --eval_every 2000 --log_every 500")
    ap.add_argument("--init_dataset", default="amarnath10chinu/puformer-chikusei-x4-init")
    ap.add_argument("--gaps", type=int, default=1)
    ap.add_argument("--slug", default="puformer-chikusei-x4-v3")
    ap.add_argument("--title", default="")
    ap.add_argument("--user", default="amarnath10chinu")
    a = ap.parse_args()
    b64, sha = code_blob()
    rev = git_rev()
    srcs = [SETUP.format(hours=a.hours, b64=b64, sha=sha, rev=rev), CACHE, BASELINE,
            PROBE.format(train_args=a.train_args), TRAIN] + ([GAPS] if a.gaps else []) + [SUMMARY]
    nb = dict(nbformat=4, nbformat_minor=5,
              metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python"),
                            language_info=dict(name="python")),
              cells=[dict(cell_type="code", metadata={}, execution_count=None, outputs=[], id=f"c{i}", source=s)
                     for i, s in enumerate(srcs)])
    name = f"{a.slug}.ipynb"
    with open(os.path.join(HERE, name), "w", newline="\n") as f:
        json.dump(nb, f, indent=1)
    title = a.title or a.slug.replace("puformer", "PUFormer").replace("chikusei", "Chikusei").replace("-", " ")
    meta = dict(id=f"{a.user}/{a.slug}", title=title, code_file=name, language="python",
                kernel_type="notebook", is_private=True, enable_gpu=True, enable_internet=True,
                machine_shape="NvidiaTeslaT4",
                dataset_sources=["mingliu123/chikusei", a.init_dataset], competition_sources=[], kernel_sources=[])
    with open(os.path.join(HERE, "kernel-metadata.json"), "w", newline="\n") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {name} ({len(b64) / 1024:.0f} KB code, sha256 {sha[:12]}, git {rev}) and kernel-metadata.json "
          f"for {meta['id']}")


if __name__ == "__main__":
    main()
