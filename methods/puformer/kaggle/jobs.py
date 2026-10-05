"""Every training job of the extended study, and how they are packed into Kaggle sessions.

A job is one train.py run on one GPU with a fixed wall-clock budget; a session is a 2 x T4 Kaggle kernel
whose two GPUs each work through a queue of jobs. Equal budgets make the comparison fair: every method,
PUFormer included, gets HOURS of training on one T4 with the same data pipeline, patch size (64; 128 at
x16), batch (8; 4 at x16), EMA and validation-only model selection; the published networks keep their official learning rate,
weight decay and clipping (hsifuse/external.py).

    benchmark  methods x ({Chikusei, Pavia Centre} x {x4, x8, x16} + {CAVE, Harvard} x {x4, x8})
    ablation   PUFormer variants on Pavia Centre x4 (+ the operator ablations on CAVE x4), 3 seeds for the
               full model and the zero-border model
"""
from __future__ import annotations

HOURS = 4.0                    # training budget of every benchmark / ablation job (one T4)
SSR_HOURS = 1.5                # SSR-NET converges long before that
SETTINGS = [(d, s) for d in ("pavia", "chikusei") for s in (4, 8, 16)] + [(d, s) for d in ("cave", "harvard") for s in (4, 8)]
METHODS = ["puformer", "psrt", "dct", "mimformer", "ssrnet"]

# PUFormer flags of each ablation variant (the full model has none)
ABLATIONS = {
    "full": "",
    "nophys": "--physics 0",                       # cascade of priors, no data step, no residual input
    "nores": "--res_input 0",                      # data step, but the prior does not see the residual
    "zeros": "--pad zeros",                        # exact kernel, zero border (mismatched with the simulation)
    "learned": "--op learned",                     # blur kernel and SRF learned with the network
    "nomem": "--memory 0",
    "k1": "--stages 1",
    "k2": "--stages 2",
    "k4": "--stages 4",
    "conv": "--mixer conv",                        # attention replaced by a convolutional mixer
    "window": "--mixer window",                    # 8x8 spatial-window attention instead of channel attention
    "l1": "--w_sam 0 --w_ssim 0",                  # L1 + deep supervision only
    "nodeep": "--w_deep 0",                        # no deep supervision
}
FULL_LOSS = "--w_sam 0.05 --w_ssim 0.1"


def job(name, dataset, scale, model, extra="", hours=HOURS, robust=False, seed=0):
    # x16 leaves a 4x4 LR patch at 64x64, so x16 trains on 128x128 patches (batch 4) for every method
    batch = "--patch 128 --bs 4" if scale == 16 else "--bs 8"
    args = f"--dataset {dataset} --scale {scale} --model {model} {batch} --seed {seed} --eval_every 1000 " \
           f"--log_every 1000 --test_log 0"
    if model == "puformer":
        args += f" --pad reflect {FULL_LOSS}"
    # eval_gaps.py takes the protocol and the model-structure flags, not the training ones
    struct = ("--pad", "--physics", "--res_input", "--memory", "--mixer", "--op", "--stages")
    toks = f"{args} {extra}".split()
    rargs = [f"--dataset {dataset} --scale {scale} --model {model}"]
    rargs += [f"{t} {v}" for t, v in zip(toks[::1], toks[1::1]) if t in struct]
    return dict(name=name, dataset=dataset, scale=scale, model=model, args=f"{args} {extra}".strip(), hours=hours,
                robust=robust, rargs=" ".join(rargs))


def all_jobs():
    jobs = {}
    for d, s in SETTINGS:
        for m in METHODS:
            n = f"bench-{d}-x{s}-{m}"
            jobs[n] = job(n, d, s, m, hours=SSR_HOURS if m == "ssrnet" else HOURS, robust=m == "puformer")
    for v, flags in ABLATIONS.items():
        if v != "full":
            n = f"abl-pavia-x4-{v}"
            jobs[n] = job(n, "pavia", 4, "puformer", flags, robust=v in ("zeros", "learned"))
    for v in ("nophys", "zeros", "learned"):
        n = f"abl-cave-x4-{v}"
        jobs[n] = job(n, "cave", 4, "puformer", ABLATIONS[v], robust=True)
    for seed in (1, 2):
        for v in ("full", "zeros"):
            n = f"seed{seed}-pavia-x4-{v}"
            jobs[n] = job(n, "pavia", 4, "puformer", ABLATIONS[v], seed=seed)
    return jobs


# sessions: slug -> (GPU 0 queue, GPU 1 queue); each queue holds at most 2 jobs of HOURS (12 h session limit)
SESSIONS = {
    # wave 1: the ablation study and the Pavia Centre x4 benchmark
    "puf-s01": (["bench-pavia-x4-puformer", "abl-pavia-x4-zeros"], ["abl-pavia-x4-nophys", "abl-pavia-x4-learned"]),
    "puf-s02": (["bench-pavia-x4-psrt", "bench-pavia-x4-dct"], ["bench-pavia-x4-mimformer", "abl-pavia-x4-nomem"]),
    "puf-s03": (["abl-pavia-x4-k1", "abl-pavia-x4-k2"], ["abl-pavia-x4-conv", "abl-pavia-x4-window"]),
    "puf-s04": (["abl-pavia-x4-nores", "abl-pavia-x4-k4"], ["abl-pavia-x4-l1", "abl-pavia-x4-nodeep"]),
    "puf-s05": (["seed1-pavia-x4-full", "seed1-pavia-x4-zeros"], ["seed2-pavia-x4-full", "seed2-pavia-x4-zeros"]),
    # wave 2: Chikusei x4 and CAVE x4
    "puf-s06": (["bench-chikusei-x4-puformer", "bench-chikusei-x4-psrt"],
                ["bench-chikusei-x4-dct", "bench-chikusei-x4-mimformer"]),
    "puf-s07": (["bench-cave-x4-puformer", "bench-cave-x4-psrt"], ["bench-cave-x4-dct", "bench-cave-x4-mimformer"]),
    "puf-s08": (["abl-cave-x4-nophys", "abl-cave-x4-zeros"],
                ["abl-cave-x4-learned", "bench-pavia-x4-ssrnet", "bench-chikusei-x4-ssrnet", "bench-cave-x4-ssrnet"]),
    # wave 3: x8
    "puf-s09": (["bench-pavia-x8-puformer", "bench-pavia-x8-psrt"], ["bench-pavia-x8-dct", "bench-pavia-x8-mimformer"]),
    "puf-s10": (["bench-chikusei-x8-puformer", "bench-chikusei-x8-psrt"],
                ["bench-chikusei-x8-dct", "bench-chikusei-x8-mimformer"]),
    "puf-s11": (["bench-cave-x8-puformer", "bench-cave-x8-psrt"], ["bench-cave-x8-dct", "bench-cave-x8-mimformer"]),
    # wave 4: Harvard, and SSR-NET on the remaining settings
    "puf-s12": (["bench-harvard-x4-puformer", "bench-harvard-x4-psrt"],
                ["bench-harvard-x4-dct", "bench-harvard-x4-mimformer"]),
    "puf-s13": (["bench-harvard-x8-puformer", "bench-harvard-x8-psrt"],
                ["bench-harvard-x8-dct", "bench-harvard-x8-mimformer"]),
    "puf-s14": (["bench-pavia-x8-ssrnet", "bench-chikusei-x8-ssrnet", "bench-cave-x8-ssrnet"],
                ["bench-harvard-x4-ssrnet", "bench-harvard-x8-ssrnet"]),
    # wave 5: x16 on Chikusei and Pavia Centre
    "puf-s15": (["bench-pavia-x16-puformer", "bench-pavia-x16-psrt"],
                ["bench-pavia-x16-dct", "bench-pavia-x16-mimformer"]),
    "puf-s16": (["bench-chikusei-x16-puformer", "bench-chikusei-x16-psrt"],
                ["bench-chikusei-x16-dct", "bench-chikusei-x16-mimformer"]),
    "puf-s17": (["bench-pavia-x16-ssrnet"], ["bench-chikusei-x16-ssrnet"]),
}
