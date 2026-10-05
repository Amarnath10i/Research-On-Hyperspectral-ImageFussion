"""Write the Kaggle notebook of one session of jobs.py (2 x T4, one job queue per GPU).

    python build_jobs.py --session puf-s01 --user amarnath10chinu     # -> sessions/puf-s01/
    kaggle kernels push -p sessions/puf-s01
    python build_jobs.py --smoke --user amarnath10chinu                # short end-to-end check of every path

Each job runs train.py on its own GPU (CUDA_VISIBLE_DEVICES) until its budget is used, then the final test
inside train.py (best-validation EMA weights, plain and self-ensemble), then for PUFormer jobs marked robust
the robustness study (eval_gaps.py). Large files are dropped afterwards: only best_ema.pt, results.json,
history.json, gaps.json, train.log and the float16 test prediction stay in the output.
"""
import argparse
import json
import os

from build_notebook import HERE, code_blob, git_rev
from jobs import SESSIONS, all_jobs

DATASETS = {"chikusei": "mingliu123/chikusei", "pavia": "mlxlx0000/paviadata",
            "cave": "liptee/hyperspectral-image-restoration-based-on-cave", "harvard": "nikeshreddypatlolla/harvard-hsi-2"}

SETUP = r'''# 0) setup: session clock, embedded code, data locations
import base64, glob, hashlib, io, json, os, shlex, shutil, subprocess, sys, tarfile, threading, time
T0 = time.time()
LIMIT = T0 + 11.75 * 3600            # Kaggle stops the session at 12 h
CODE, OUT = '/kaggle/working/code', '/kaggle/working/out'
raw = base64.b64decode('{b64}')
assert hashlib.sha256(raw).hexdigest() == '{sha}'
tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz').extractall(CODE)
os.makedirs(OUT, exist_ok=True)
print('code sha256 {sha} from git {rev}', flush=True)
subprocess.run('nvidia-smi --query-gpu=index,name,memory.total --format=csv; free -g | head -2; nproc', shell=True)
QUEUES = {queues}
SMOKE = {smoke}
SMOKE_ROBUST = ('bench-cave-x4-puformer', 'abl-pavia-x4-learned', 'bench-pavia-x8-puformer')


def find(pattern):
    hits = sorted(glob.glob('/kaggle/input/**/' + pattern, recursive=True), key=len)
    return hits[0] if hits else None


chik = [p for p in glob.glob('/kaggle/input/**/*.mat', recursive=True) if 'chikusei' in p.lower() and 'Ground' not in p]
DATA = dict(chikusei=sorted(chik, key=len)[0] if chik else None, pavia=find('Pavia.mat'),
            cave=os.path.dirname(os.path.dirname(os.path.dirname(find('Train/HSI/balloons.mat') or '/x/x/x/x'))),
            harvard=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(find('Data/Train/HSI/img*.mat') or '/x/x/x/x/x')))))
CACHE = dict(chikusei='/tmp/chikusei.npy', harvard='/tmp/harvard')
print('data:', DATA, flush=True)
ENV = dict(os.environ, OMP_NUM_THREADS='2', PUF_EXT='/kaggle/working/ext', PYTHONUNBUFFERED='1')
'''

PREP = r'''# 1) self-checks, data caches shared by both GPUs, the published networks' code
assert subprocess.run('python selfcheck.py', shell=True, cwd=CODE, env=ENV).returncode == 0
used = {{j['dataset'] for q in QUEUES for j in q}}
for d in used:
    if d in CACHE:   # one process builds the cache, the GPU jobs memory-map it
        t = time.time()
        rc = subprocess.run([sys.executable, '-c', f"from hsifuse.data import load_dataset; load_dataset({{d!r}}, {{DATA[d]!r}}, {{CACHE[d]!r}})"],
                            cwd=CODE, env=ENV).returncode
        print(d, 'cache built, rc', rc, f'{{time.time() - t:.0f}} s', flush=True)
models = {{j['model'] for q in QUEUES for j in q}} - {{'puformer', 'ssrnet'}}
sys.path.insert(0, CODE)
os.environ['PUF_EXT'] = ENV['PUF_EXT']
from hsifuse.external import fetch
for m in sorted(models):
    fetch(m)
'''

RUN = r'''# 2) the two GPU queues
lock = threading.Lock()


def say(*s):
    with lock:
        print(*s, flush=True)


def sh(cmd, gpu, log):
    env = dict(ENV, CUDA_VISIBLE_DEVICES=str(gpu))
    with open(log, 'a') as f:
        p = subprocess.Popen(cmd, shell=True, cwd=CODE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            f.write(line)
            if any(k in line for k in ('ep ', 'TEST', 'consistency', 'Error', 'error', 'Traceback', 'PROBE', 'nominal',
                                       'sigma', 'noise', 'srf', 'warm', 'params on', 'train ', 'test:', 'budget', 'it/s', 'external')):
                say(f'[gpu{{gpu}}]', line.rstrip()[:400])
        return p.wait()


def worker(gpu, queue):
    for i, j in enumerate(queue):
        out = f"{{OUT}}/{{j['name']}}"
        os.makedirs(out, exist_ok=True)
        log = f'{{out}}/train.log'
        left = LIMIT - time.time()
        reserve = 1800 * (len(queue) - i)                    # final test + robustness of this and later jobs
        budget = min(j['hours'] * 3600, (left - reserve) / (len(queue) - i) if i < len(queue) - 1 else left - 1800)
        if budget < 1200 and not SMOKE:
            say(f'[gpu{{gpu}}] skip {{j["name"]}}: only {{budget:.0f}} s left'); continue
        deadline = time.time() + budget
        cache = f"--cache {{CACHE[j['dataset']]}}" if j['dataset'] in CACHE else ''
        data = f"--mat {{shlex.quote(DATA[j['dataset']])}} {{cache}}"
        stop = '--iters 60 --eval_every 30 --log_every 20 --q2n 0' if SMOKE else f'--deadline {{deadline}}'
        say(f'[gpu{{gpu}}] START {{j["name"]}} budget {{budget / 3600:.2f}} h: {{j["args"]}}')
        rc = sh(f"python train.py {{data}} {{stop}} {{j['args']}} --out {{out}}", gpu, log)   # job args win
        say(f'[gpu{{gpu}}] END {{j["name"]}} rc={{rc}} after {{(time.time() - T0) / 3600:.2f}} h')
        if rc == 0 and j['robust'] and time.time() < LIMIT - 900 and (not SMOKE or j['name'] in SMOKE_ROBUST):
            sh(f"python eval_gaps.py {{data}} {{j['rargs']}} --ckpt {{out}}/best_ema.pt --out {{out}}", gpu, log)
        for f in ['last.pt', 'ema_last.pt', 'test_pred_tta.npy'] + [os.path.basename(p) for p in glob.glob(f'{{out}}/ema_ep*.pt')]:
            if os.path.exists(f'{{out}}/{{f}}'):
                os.remove(f'{{out}}/{{f}}')
        pred = f'{{out}}/test_pred.npy'
        if os.path.exists(pred):
            import numpy as np
            np.save(f'{{out}}/test_pred_f16.npy', np.load(pred).astype(np.float16)); os.remove(pred)


threads = [threading.Thread(target=worker, args=(g, q)) for g, q in enumerate(QUEUES)]
for t in threads:
    t.start()
for t in threads:
    t.join()
'''

SUMMARY = r'''# 3) summary
for q in QUEUES:
    for j in q:
        f = f"{{OUT}}/{{j['name']}}/results.json"
        if not os.path.exists(f):
            print(j['name'], 'no results'); continue
        r = json.load(open(f))
        t, tt = r['test'], r['test_tta']
        print(f"{{j['name']:34s}} it {{r['iters']:6d}} {{r['params_M']:.2f}}M val {{r['best_val_PSNR']:.3f}} "
              f"| PSNR {{t['PSNR']:.3f}}/{{tt['PSNR']:.3f}} SAM {{t['SAM']:.3f}} ERGAS {{t['ERGAS']:.3f}} "
              f"SSIM {{t['SSIM']:.4f}} RMSE {{t['RMSE_DN']:.2f}} | cons {{r['consistency']}}")
subprocess.run(f'du -sh {{OUT}}; rm -rf /kaggle/working/ext', shell=True)
'''


PSRT_PROBE = r'''# debug: PSRT output at initialisation on real CAVE patches, train vs eval mode, and inside the network
import torch
import torch.nn.functional as F
from hsifuse.data import PatchSampler, load_dataset
from hsifuse.models import build
from hsifuse.ops import Degradation, dataset_srf
tr, te, va, peak = load_dataset('cave', DATA['cave'])
deg = Degradation(dataset_srf('cave'), scale=4, pad='reflect').cuda()
lr, ms, gt = PatchSampler(tr, deg, 64, 'cuda')(8)
torch.manual_seed(0)
net = build('psrt', deg).cuda()
up = F.interpolate(lr, scale_factor=4, mode='bicubic', align_corners=False)
print('gt', gt.mean().item(), gt.std().item(), '| bicubic L1', (up - gt).abs().mean().item())
for mode in ('train', 'eval'):
    getattr(net, mode)()
    with torch.no_grad():
        out = net(lr, ms)
    print(mode, 'out mean/std/max', out.mean().item(), out.std().item(), out.abs().max().item(),
          'L1', (out - gt).abs().mean().item(), flush=True)
inner = net.net
with torch.no_grad():
    xt = torch.cat((up, ms), 1)
    f = inner.w.conv(xt)
    print('embed std/max', f.std().item(), f.abs().max().item())
    h = f
    for i, layer in enumerate(inner.w.layers):
        h = layer(64, 64, h)
        print('pyramid', i, 'std/max', h.std().item(), h.abs().max().item())
    c = inner.conv(h)
    print('final conv std/max', c.std().item(), c.abs().max().item(), flush=True)
del tr, te, va
'''


def notebook(cells):
    return dict(nbformat=4, nbformat_minor=5,
                metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python"),
                              language_info=dict(name="python")),
                cells=[dict(cell_type="code", metadata={}, execution_count=None, outputs=[], id=f"c{i}", source=s)
                       for i, s in enumerate(cells)])


def smoke_queues(jobs):
    """Every model on every dataset (x8 on Pavia), every ablation, with 40 iterations each."""
    pick = [f"bench-{d}-x4-{m}" for d in ("cave", "harvard") for m in ("puformer", "psrt", "dct", "mimformer")]
    pick += [f"bench-pavia-x8-{m}" for m in ("puformer", "psrt", "dct", "mimformer", "ssrnet")]
    pick += ["bench-chikusei-x8-puformer", "bench-cave-x8-dct", "bench-harvard-x8-mimformer"]
    pick += [f"bench-{d}-x16-{m}" for d in ("pavia", "chikusei") for m in ("puformer", "psrt", "dct", "mimformer")]
    pick += [n for n in jobs if n.startswith("abl-pavia-x4-")] + ["abl-cave-x4-learned"]
    q = [jobs[n] for n in pick]
    return [q[0::2], q[1::2]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--quick", action="store_true", help="with --smoke: only the four networks on CAVE x4 and Pavia x16")
    ap.add_argument("--debug", action="store_true", help="PSRT trajectory, memory of DCT / MIMFormer, PUFormer x8/x16")
    ap.add_argument("--user", default="amarnath10chinu")
    a = ap.parse_args()
    jobs = all_jobs()
    if a.debug:
        slug = "puf-debug"
        long = " --iters 400 --warmup 0 --eval_every 100 --log_every 50"

        def dbg(n, extra=""):
            return dict(jobs[n], name=n + "-dbg", args=jobs[n]["args"] + extra, robust=False)
        queues = [[dbg("bench-cave-x4-psrt", long), dbg("bench-cave-x4-dct"), dbg("bench-pavia-x8-puformer")],
                  [dbg("bench-pavia-x16-mimformer"), dbg("bench-pavia-x16-dct"), dbg("bench-pavia-x16-puformer"),
                   dbg("bench-chikusei-x16-psrt")]]
    elif a.smoke and a.quick:
        slug = "puf-smoke-quick"
        queues = [[jobs["bench-cave-x4-puformer"], jobs["bench-cave-x4-dct"], jobs["bench-pavia-x16-puformer"]],
                  [jobs["bench-cave-x4-psrt"], jobs["bench-cave-x4-mimformer"], jobs["abl-pavia-x4-learned"]]]
    elif a.smoke:
        slug, queues = "puf-smoke", smoke_queues(jobs)
    else:
        slug, queues = a.session, [[jobs[n] for n in q] for q in SESSIONS[a.session]]
    b64, sha = code_blob()
    out = os.path.join(HERE, "sessions", slug)
    os.makedirs(out, exist_ok=True)
    cells = [SETUP.format(b64=b64, sha=sha, rev=git_rev(), queues=repr(queues), smoke=a.smoke or a.debug),
             PREP.format()] + ([PSRT_PROBE] if a.debug else []) + [RUN.format(), SUMMARY.format()]
    name = f"{slug}.ipynb"
    with open(os.path.join(out, name), "w", newline="\n") as f:
        json.dump(notebook(cells), f, indent=1)
    used = sorted({j["dataset"] for q in queues for j in q})
    meta = dict(id=f"{a.user}/{slug}", title=slug, code_file=name, language="python", kernel_type="notebook",
                is_private=True, enable_gpu=True, enable_internet=True, machine_shape="NvidiaTeslaT4",
                dataset_sources=[DATASETS[d] for d in used], competition_sources=[], kernel_sources=[])
    with open(os.path.join(out, "kernel-metadata.json"), "w", newline="\n") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote sessions/{slug}/{name} for {meta['id']}: " + " | ".join(",".join(j["name"] for j in q) for q in queues))


if __name__ == "__main__":
    main()
