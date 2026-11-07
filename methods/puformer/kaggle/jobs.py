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
METHODS = ["puformer", "psrt", "dct", "mimformer", "ssrnet", "dhif"]

# PUFormer flags of each ablation variant (the full model has none)
ABLATIONS = {
    "full": "",
    "nophys": "--physics 0",                       # cascade of priors, no data step, no residual input
    "nores": "--res_input 0",                      # data step, but the prior does not see the residual
    "zeros": "--pad zeros",                        # exact kernel, zero border (mismatched with the simulation)
    "learned": "--op learned",                     # blur kernel and SRF learned with the network
    "cg": "--dstep cg",                            # exact proximal data step (5 conjugate-gradient iterations)
    "mres": "--mres 1",                            # the prior also gets the back-projected HR-MSI residual
    "kron": "--dstep kron",                        # the exact proximal data step in closed form (Kronecker)
    "kl": "--dstep kronlearn",                     # ... with learned, fixed-point preserving spectral gains
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
# operator randomization of the operator-conditioned runs (train.py): blur 0.5-2 x nominal, SRF shift +-12 nm,
# noise at 30-50 dB SNR on half of the batches
RAND = "--rand_sigma 0.5,2.0 --rand_shift 12 --rand_snr 30,50 --noise_prob 0.5"
# The authors' own training recipes, read from their repositories (the pipeline otherwise gives every baseline its
# published learning rate with batch 8, AdamW(0.9, 0.99) without decay or clipping, and warm-up + cosine decay).
# Epoch-based step schedules are mapped onto the time budget as fractions of it.
#   PSRT main_x4.py / args_parser.py: AdamW(lr 1e-4, wd 1e-4), clip 10, batch 32, constant lr, L1
#   MIMFormer main.py / utils.py: Adam(lr 1e-4), batch 16, MultiStepLR(30, 60, 90; 0.5) over 101 epochs, L1
OFFICIAL = {
    "psrt": "--bs 32 --lr 1e-4 --wd 1e-4 --clip 10 --beta2 0.999 --warmup 0 --lr_shape const",
    "mimformer": "--bs 16 --lr 1e-4 --wd 0 --clip 0 --beta2 0.999 --warmup 0 --lr_shape step "
                 "--milestones 0.297,0.594,0.891 --gamma 0.5",
}
OURS_RECIPE = "--lr 3e-4 --wd 1e-4 --clip 0.1 --w_sam 0.05 --w_ssim 0.1"            # PUFormer's, for MIMFormer
BASE_RECIPE = "--lr 1e-4 --wd 0 --clip 0 --w_sam 0 --w_ssim 0 --w_deep 0"           # the baselines', for PUFormer


def job(name, dataset, scale, model, extra="", hours=HOURS, robust=False, seed=0, eval_extra=""):
    # x16 leaves a 4x4 LR patch at 64x64, so x16 trains on 128x128 patches (batch 4) for every method
    batch = "--patch 128 --bs 4" if scale == 16 else "--bs 8"
    args = f"--dataset {dataset} --scale {scale} --model {model} {batch} --seed {seed} --eval_every 1000 " \
           f"--log_every 1000 --test_log 0"
    if model == "puformer":
        args += f" --pad reflect {FULL_LOSS}"
    # eval_gaps.py takes the protocol and the model-structure flags, not the training ones
    struct = ("--pad", "--physics", "--res_input", "--memory", "--mixer", "--op", "--stages", "--omc", "--dstep", "--cg_iters", "--mres",
              "--reg", "--sa", "--sensor", "--nsp", "--ntp", "--cwd", "--unc", "--prior", "--nomsi", "--nohsi")
    toks = f"{args} {extra}".split()
    rargs = [f"--dataset {dataset} --scale {scale} --model {model}"]
    rargs += [f"{t} {v}" for t, v in zip(toks[::1], toks[1::1]) if t in struct]
    rargs += [eval_extra] if eval_extra else []
    return dict(name=name, dataset=dataset, scale=scale, model=model, args=f"{args} {extra}".strip(), hours=hours,
                robust=robust, rargs=" ".join(rargs))


def all_jobs():
    jobs = {}
    for d, s in SETTINGS:
        for m in METHODS:
            n = f"bench-{d}-x{s}-{m}"
            jobs[n] = job(n, d, s, m, hours=SSR_HOURS if m == "ssrnet" else HOURS, robust=m == "puformer")
    for v, flags in ABLATIONS.items():
        if v not in ("full", "cg", "mres", "kron", "kl"):
            n = f"abl-pavia-x4-{v}"
            jobs[n] = job(n, "pavia", 4, "puformer", flags, robust=v in ("zeros", "learned"))
    # architecture extensions of the data step and the prior input (x4, both datasets, with the robustness run)
    for d in ("chikusei", "pavia"):
        for v in ("cg", "mres", "kron", "kl"):
            n = f"abl-{d}-x4-{v}"
            jobs[n] = job(n, d, 4, "puformer", ABLATIONS[v], robust=True)
    # the operator ablation (no data step, zero border, learned operator) on the other datasets and at x8
    for d, s in (("cave", 4), ("harvard", 4), ("chikusei", 8), ("pavia", 8)):
        for v in ("nophys", "zeros", "learned"):
            n = f"abl-{d}-x{s}-{v}"
            jobs[n] = job(n, d, s, "puformer", ABLATIONS[v], robust=True)
    # Chikusei x4: every ablation row of Pavia Centre, and three seeds for the full model and the zero border
    for v in ABLATIONS:
        if v not in ("full", "cg", "mres", "kron", "kl"):
            n = f"abl-chikusei-x4-{v}"
            jobs[n] = job(n, "chikusei", 4, "puformer", ABLATIONS[v], robust=v in ("zeros", "learned"))
    for seed in (1, 2):
        for v in ("full", "zeros"):
            n = f"seed{seed}-chikusei-x4-{v}"
            jobs[n] = job(n, "chikusei", 4, "puformer", ABLATIONS[v], seed=seed)
    for seed in (1, 2):
        for v in ("full", "zeros"):
            n = f"seed{seed}-pavia-x4-{v}"
            jobs[n] = job(n, "pavia", 4, "puformer", ABLATIONS[v], seed=seed)
    # the strongest baseline with the same three seeds, so the x4 comparison has a run-to-run spread on both sides
    for seed in (1, 2):
        for d in ("chikusei", "pavia"):
            n = f"seed{seed}-{d}-x4-mimformer"
            jobs[n] = job(n, d, 4, "mimformer", seed=seed)
    # operator-conditioned training: every batch has its own blur, SRF shift and noise; "puformer" gets that
    # operator in its data steps, "nocond" the same data with the nominal operator (augmentation only), "zeros" the
    # zero-border operator, "mimformer" the strongest baseline with the same randomized data
    for d in ("chikusei", "pavia"):
        for v, m, extra in (("puformer", "puformer", ""), ("nocond", "puformer", "--cond 0"),
                            ("zeros", "puformer", "--pad zeros"), ("mimformer", "mimformer", "")):
            n = f"cond-{d}-x4-{v}"
            jobs[n] = job(n, d, 4, m, f"{RAND} {extra}".strip(), robust=True, eval_extra="--wide 1")
        # + observation-model conditioning: noise-adaptive data steps and a prior modulated by the operator / noise
        n = f"cond-{d}-x4-omc"
        jobs[n] = job(n, d, 4, "puformer", f"{RAND} --omc 1", robust=True, eval_extra="--wide 1")
        # + the closed-form exact data step: the full method
        n = f"cond-{d}-x4-kron"
        jobs[n] = job(n, d, 4, "puformer", f"{RAND} --omc 1 --dstep kron", robust=True, eval_extra="--wide 1")
        n = f"cond-{d}-x4-kl"
        jobs[n] = job(n, d, 4, "puformer", f"{RAND} --omc 1 --dstep kronlearn", robust=True, eval_extra="--wide 1")
    # real data (NEON spectrometer + RGB camera, x4): every method, the operator ablation, and an operator-
    # conditioned PUFormer (blur and noise only: the measured RGB cannot be re-simulated with another response)
    for m in METHODS:
        n = f"bench-neon-x4-{m}"
        jobs[n] = job(n, "neon", 4, m, hours=SSR_HOURS if m == "ssrnet" else HOURS, robust=m == "puformer")
    for v in ("nophys", "zeros", "learned"):
        n = f"abl-neon-x4-{v}"
        jobs[n] = job(n, "neon", 4, "puformer", ABLATIONS[v], robust=True)
    jobs["cond-neon-x4-puformer"] = job("cond-neon-x4-puformer", "neon", 4, "puformer",
                                        "--rand_sigma 0.5,2.0 --rand_snr 30,50 --noise_prob 0.5", robust=True,
                                        eval_extra="--wide 1")
    # where does the margin over the retrained networks come from? (Chikusei and Pavia Centre x4)
    #   off-*  PSRT and MIMFormer with their authors' own recipe (batch, optimiser, schedule), not the pipeline's
    #   rec-*  the recipes swapped: MIMFormer with PUFormer's, PUFormer with the baselines'
    #   unet   PUFormer's prior alone: one U-Net on the bicubic image and the HR-MSI, no data step, no stages
    for d in ("chikusei", "pavia"):
        for m in ("psrt", "mimformer"):
            n = f"off-{d}-x4-{m}"
            jobs[n] = job(n, d, 4, m, OFFICIAL[m])
        jobs[f"rec-{d}-x4-mimformer"] = job(f"rec-{d}-x4-mimformer", d, 4, "mimformer", OURS_RECIPE)
        jobs[f"rec-{d}-x4-puformer"] = job(f"rec-{d}-x4-puformer", d, 4, "puformer", BASE_RECIPE)
        jobs[f"abl-{d}-x4-unet"] = job(f"abl-{d}-x4-unet", d, 4, "puformer", "--stages 1 --physics 0")
    # a longer budget for PUFormer and the strongest baseline: does the gap survive 8 h of training?
    for m in ("puformer", "mimformer"):
        for d in ("pavia", "chikusei"):
            n = f"long-{d}-x4-{m}"
            jobs[n] = job(n, d, 4, m, hours=8.0)
    # further published networks of the comparison in Chen et al. (TIP 2026), x4 on all four datasets, and x8 on
    # Chikusei and Pavia Centre for those whose code supports it (SMGU-Net is written for x4 only)
    for d in ("chikusei", "pavia", "cave", "harvard"):
        for m in NEW_METHODS:
            n = f"bench-{d}-x4-{m}"
            jobs[n] = job(n, d, 4, m)
    for d in ("chikusei", "pavia"):
        for m in NEW_METHODS:
            if m != "smgu":
                n = f"bench-{d}-x8-{m}"
                jobs[n] = job(n, d, 8, m)
    # the wider operator family: axis-aligned anisotropic blur and sub-pixel offsets of the LR-HSI grid
    # (misregistration) on top of RAND; PUFormer gets the batch's operator and the 7-dim observation code, MIMFormer
    # the same randomized data
    for d in ("chikusei", "pavia"):
        n = f"cond-{d}-x4-reg"
        jobs[n] = job(n, d, 4, "puformer", f"{RAND} {WIDE} --omc 1 --reg 1", robust=True, eval_extra="--wide 1")
        n = f"cond-{d}-x4-mimreg"
        jobs[n] = job(n, d, 4, "mimformer", f"{RAND} {WIDE}", robust=True, eval_extra="--wide 1")
    # one network for any multispectral sensor: a random response per batch (3-10 bands); the network sees the HR-MSI
    # only through R^+ Y_M (and PUFormer through its data steps). MIMFormer gets the same input and data (control).
    # References: networks trained for one sensor only (Sentinel-2, 10 bands; RGB camera, 3 bands).
    for d in ("chikusei", "pavia"):
        jobs[f"sa-{d}-x4-puformer"] = job(f"sa-{d}-x4-puformer", d, 4, "puformer", f"{RAND} --rand_srf 1 --sa 1 --omc 1")
        jobs[f"sa-{d}-x4-mimformer"] = job(f"sa-{d}-x4-mimformer", d, 4, "mimformer", f"{RAND} --rand_srf 1 --sa 1")
        # one network for every scale factor: x4 and x8 in training, x16 unseen
        jobs[f"scl-{d}-x4-puformer"] = job(f"scl-{d}-x4-puformer", d, 4, "puformer", "--rand_scale 4,8")
    for sensor in ("sentinel2", "rgb"):
        for m in ("puformer", "mimformer"):
            n = f"spec-chikusei-x4-{m}-{sensor}"
            jobs[n] = job(n, "chikusei", 4, m, f"--sensor {sensor}")
    # the redesigned stage: closed-form data step with learned spectral gains, null-space-gated prior, and an
    # interface in the hyperspectral band space only (R^+ Y_M and both back-projected residuals)
    for d in ("chikusei", "pavia"):
        jobs[f"arch-{d}-x4-new"] = job(f"arch-{d}-x4-new", d, 4, "puformer", NEWARCH, robust=True)
        jobs[f"arch-{d}-x4-nogate"] = job(f"arch-{d}-x4-nogate", d, 4, "puformer", NEWARCH.replace(" --nsp 1", ""),
                                          robust=True)
        jobs[f"cond-{d}-x4-new"] = job(f"cond-{d}-x4-new", d, 4, "puformer", f"{RAND} --omc 1 {NEWARCH}", robust=True,
                                       eval_extra="--wide 1")
        jobs[f"cond-{d}-x4-regnew"] = job(f"cond-{d}-x4-regnew", d, 4, "puformer",
                                          f"{RAND} {WIDE} --omc 1 --reg 1 {NEWARCH}", robust=True, eval_extra="--wide 1")
        jobs[f"sa-{d}-x4-new"] = job(f"sa-{d}-x4-new", d, 4, "puformer", f"{RAND} --rand_srf 1 --omc 1 {NEWARCH}")
        # the null-space transfer prior on top of the redesigned stage
        jobs[f"arch-{d}-x4-ntp"] = job(f"arch-{d}-x4-ntp", d, 4, "puformer", f"{NEWARCH} --ntp 1", robust=True)
        # corrupted observations (local changes in the HR-MSI, LR-HSI stripes): with and without the confidence-
        # weighted data step, and MIMFormer on the same data
        cor = f"{RAND} --rand_corrupt 0.5"
        jobs[f"cond-{d}-x4-cwd"] = job(f"cond-{d}-x4-cwd", d, 4, "puformer", f"{cor} --omc 1 {NEWARCH} --cwd 1",
                                       robust=True, eval_extra="--wide 1")
        jobs[f"cond-{d}-x4-cor"] = job(f"cond-{d}-x4-cor", d, 4, "puformer", f"{cor} --omc 1 {NEWARCH}", robust=True,
                                       eval_extra="--wide 1")
        jobs[f"cond-{d}-x4-mimcor"] = job(f"cond-{d}-x4-mimcor", d, 4, "mimformer", cor, robust=True,
                                          eval_extra="--wide 1")
        # observability-aware uncertainty (the head is trained on the detached null-space error)
        jobs[f"arch-{d}-x4-unc"] = job(f"arch-{d}-x4-unc", d, 4, "puformer", f"{NEWARCH} --unc 1", robust=True)
    # the eigen-operator prior (no U-Net) in the redesigned stage, nominal and over the operator family
    for d in ("chikusei", "pavia"):
        jobs[f"arch-{d}-x4-eon"] = job(f"arch-{d}-x4-eon", d, 4, "puformer", f"{NEWARCH} --prior eon", robust=True)
        jobs[f"cond-{d}-x4-eon"] = job(f"cond-{d}-x4-eon", d, 4, "puformer", f"{RAND} --omc 1 {NEWARCH} --prior eon",
                                       robust=True, eval_extra="--wide 1")
    # why the HR-MSI: the same network with the LR-HSI only (single-image super-resolution) or the HR-MSI only
    # (spectral super-resolution), same budget
    for d in ("chikusei", "pavia"):
        jobs[f"abl-{d}-x4-nomsi"] = job(f"abl-{d}-x4-nomsi", d, 4, "puformer", "--nomsi 1")
        jobs[f"abl-{d}-x4-nohsi"] = job(f"abl-{d}-x4-nohsi", d, 4, "puformer", "--nohsi 1")
    # the adopted model (the redesign without the gate, 2026-10-31) in the robustness, recipe and uncertainty runs
    ng = VARIANTS["nogate"]
    for d in ("chikusei", "pavia"):
        cor = f"{RAND} --rand_corrupt 0.5"
        for n, flags, kw in ((f"cond-{d}-x4-ng", f"{RAND} --omc 1 {ng}", dict(robust=True, eval_extra="--wide 1")),
                             (f"cond-{d}-x4-regng", f"{RAND} {WIDE} --omc 1 --reg 1 {ng}",
                              dict(robust=True, eval_extra="--wide 1")),
                             (f"sa-{d}-x4-ng", f"{RAND} --rand_srf 1 --omc 1 {ng}", {}),
                             (f"cond-{d}-x4-cwdng", f"{cor} --omc 1 {ng} --cwd 1", dict(robust=True, eval_extra="--wide 1")),
                             (f"cond-{d}-x4-corng", f"{cor} --omc 1 {ng}", dict(robust=True, eval_extra="--wide 1")),
                             (f"rec-{d}-x4-ng", f"{BASE_RECIPE} {ng}", {}),
                             (f"arch-{d}-x4-uncng", f"{ng} --unc 1", dict(robust=True))):
            jobs[n] = job(n, d, 4, "puformer", flags, **kw)
        jobs[f"long-{d}-x4-ng"] = job(f"long-{d}-x4-ng", d, 4, "puformer", ng, hours=8.0)
        jobs[f"scl-{d}-x4-ng"] = job(f"scl-{d}-x4-ng", d, 4, "puformer", f"--rand_scale 4,8 {ng}")
    for sensor in ("sentinel2", "rgb"):
        jobs[f"spec-chikusei-x4-ng-{sensor}"] = job(f"spec-chikusei-x4-ng-{sensor}", "chikusei", 4, "puformer",
                                                   f"--sensor {sensor} {ng}")
    jobs["nb-nogate-neon-x4"] = job("nb-nogate-neon-x4", "neon", 4, "puformer", ng, robust=True)
    jobs["cond-neon-x4-ng"] = job("cond-neon-x4-ng", "neon", 4, "puformer",
                                  f"--rand_sigma 0.5,2.0 --rand_snr 30,50 --noise_prob 0.5 {ng}", robust=True,
                                  eval_extra="--wide 1")
    # ExactFuse ablation: each run changes one part of the adopted model (robust runs give the operator swap)
    for d in ("chikusei", "pavia"):
        for v, (flags, robust) in NG_ABLATIONS.items():
            jobs[f"abl-{d}-x4-ng-{v}"] = job(f"abl-{d}-x4-ng-{v}", d, 4, "puformer", flags, robust=robust)
    # the redesigned model on every other benchmark setting and with two more x4 seeds; only the variant that the
    # architecture runs (arch-*) select is queued, so it can replace PUFormer in every table
    for v, flags in VARIANTS.items():
        for d, s in (("cave", 4), ("harvard", 4), ("cave", 8), ("harvard", 8), ("chikusei", 8), ("pavia", 8),
                     ("chikusei", 16), ("pavia", 16)):
            jobs[f"nb-{v}-{d}-x{s}"] = job(f"nb-{v}-{d}-x{s}", d, s, "puformer", flags, robust=True)
        for d in ("chikusei", "pavia"):
            for seed in (1, 2):
                jobs[f"nb-{v}-{d}-x4-s{seed}"] = job(f"nb-{v}-{d}-x4-s{seed}", d, 4, "puformer", flags, seed=seed)
    # CaFormer (PR 2026), an unfolding transformer with learned operators, x4 on all four datasets
    for d in ("chikusei", "pavia", "cave", "harvard"):
        jobs[f"bench-{d}-x4-caformer"] = job(f"bench-{d}-x4-caformer", d, 4, "caformer")
        # registration-aware baseline: trained normally (benchmark) and on misregistered HR-MSIs (registration)
        jobs[f"bench-{d}-x4-m2dtn"] = job(f"bench-{d}-x4-m2dtn", d, 4, "m2dtn")
        jobs[f"reg-{d}-x4-m2dtn"] = job(f"reg-{d}-x4-m2dtn", d, 4, "m2dtn", "--rand_msi_shift 1.0")
        # ... and with the reference shifted with the HR-MSI, the frame M2DTN reconstructs in (it warps the LR-HSI)
        jobs[f"regf-{d}-x4-m2dtn"] = job(f"regf-{d}-x4-m2dtn", d, 4, "m2dtn", "--rand_msi_shift 1.0 --msi_shift_target 1")
    return jobs


NEWARCH = "--dstep kronlearn --nsp 1 --mres 1 --sa 1"
NG = "--dstep kronlearn --mres 1 --sa 1"                                      # ExactFuse (the adopted model)
NG_ABLATIONS = {                       # variant -> (flags, robustness run); every row changes one part of ExactFuse
    "grad": ("--dstep grad --mres 1 --sa 1", True),          # gradient data step instead of the closed form
    "kron": ("--dstep kron --mres 1 --sa 1", True),          # closed form without the learned spectral gains
    "nomres": ("--dstep kronlearn --sa 1", True),            # no HR-MSI residual input
    "nosa": ("--dstep kronlearn --mres 1", True),            # the HR-MSI itself instead of R^+ Y_M
    "nophys": ("--physics 0 --sa 1", False),                 # no data steps (and no residual inputs)
    "zeros": (f"{NG} --pad zeros", True),                    # zero border instead of the simulation's
    "learned": (f"{NG} --op learned", False),                # learned blur and response (gradient step)
    "nomsi": (f"{NG} --nomsi 1", False),                     # LR-HSI only
    "nohsi": (f"{NG} --nohsi 1", False),                     # HR-MSI only
    "conv": (f"{NG} --mixer conv", True),                    # convolutional mixer instead of attention
    "k1": (f"{NG} --stages 1", True),                        # one stage
    "nomem": (f"{NG} --memory 0", True),                     # no cross-stage memory
}
VARIANTS = {"new": NEWARCH, "nogate": NEWARCH.replace(" --nsp 1", ""), "ntp": f"{NEWARCH} --ntp 1",
            "eon": f"{NEWARCH} --prior eon"}                              # candidates of the redesign


WIDE = "--rand_aniso 1 --rand_offset 1.0"


NEW_METHODS = ["smgu", "clsnet", "otpnet", "u2net", "mimosst"]


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
    # --- one job per GPU (~4.5 h sessions, so two fit the remaining weekly quota of an account) ---
    # Chikusei x4 ablation first (main dataset)
    "puf-c01": (["abl-chikusei-x4-zeros"], ["abl-chikusei-x4-learned"]),
    "puf-c02": (["abl-chikusei-x4-nophys"], ["abl-chikusei-x4-nomem"]),
    "puf-c03": (["abl-chikusei-x4-k1"], ["abl-chikusei-x4-conv"]),
    # DHIF-Net (model-guided unfolding with learned operators), then the longer-budget check
    "puf-d01": (["bench-pavia-x4-dhif"], ["bench-chikusei-x4-dhif"]),
    "puf-d02": (["bench-cave-x4-dhif"], ["bench-chikusei-x8-dhif"]),
    "puf-d03": (["bench-pavia-x8-dhif"], ["bench-cave-x8-dhif"]),
    "puf-d04": (["bench-harvard-x4-dhif"], ["bench-harvard-x8-dhif"]),
    "puf-d05": (["bench-pavia-x16-dhif"], ["bench-chikusei-x16-dhif"]),
    "puf-L01": (["long-pavia-x4-puformer"], ["long-pavia-x4-mimformer"]),
    # --- x4 only (Chikusei and Pavia Centre): SSR-NET, the rest of the Chikusei ablation, Chikusei seeds ---
    "puf-x01": (["bench-pavia-x4-ssrnet"], ["bench-chikusei-x4-ssrnet"]),
    "puf-x02": (["abl-chikusei-x4-nores"], ["abl-chikusei-x4-window"]),
    "puf-x03": (["abl-chikusei-x4-k2"], ["abl-chikusei-x4-k4"]),
    "puf-x04": (["abl-chikusei-x4-l1"], ["abl-chikusei-x4-nodeep"]),
    "puf-x05": (["seed1-chikusei-x4-full"], ["seed1-chikusei-x4-zeros"]),
    "puf-x06": (["seed2-chikusei-x4-full"], ["seed2-chikusei-x4-zeros"]),
    "puf-t01": (["bench-cave-x4-puformer"], ["bench-cave-x4-mimformer"]),
    "puf-t02": (["bench-cave-x4-psrt"], ["bench-cave-x4-dct"]),
    "puf-t03": (["bench-pavia-x8-puformer"], ["bench-pavia-x8-mimformer"]),
    "puf-t04": (["bench-pavia-x8-psrt"], ["bench-pavia-x8-dct"]),
    "puf-t05": (["bench-chikusei-x8-puformer"], ["bench-chikusei-x8-mimformer"]),
    "puf-t06": (["bench-chikusei-x8-psrt"], ["bench-chikusei-x8-dct"]),
    "puf-t07": (["bench-cave-x8-puformer"], ["bench-cave-x8-mimformer"]),
    "puf-t08": (["bench-cave-x8-psrt"], ["bench-cave-x8-dct"]),
    "puf-t09": (["bench-harvard-x4-puformer"], ["bench-harvard-x4-mimformer"]),
    "puf-t10": (["bench-harvard-x4-psrt"], ["bench-harvard-x4-dct"]),
    "puf-t11": (["nb-nogate-harvard-x8"], ["bench-harvard-x8-mimformer"]),
    "puf-t12": (["bench-harvard-x8-psrt"], ["bench-harvard-x8-dct"]),
    "puf-t13": (["nb-nogate-pavia-x16"], ["bench-pavia-x16-mimformer"]),
    "puf-t14": (["bench-pavia-x16-psrt"], ["bench-pavia-x16-dct"]),
    "puf-t15": (["bench-chikusei-x16-puformer"], ["bench-chikusei-x16-mimformer"]),
    "puf-t16": (["bench-chikusei-x16-psrt"], ["bench-chikusei-x16-dct"]),
    "puf-t17": (["abl-cave-x4-nophys"], ["abl-cave-x4-zeros"]),
    "puf-t18": (["abl-cave-x4-learned"], ["bench-pavia-x4-ssrnet", "bench-chikusei-x4-ssrnet"]),
    "puf-t19": (["bench-cave-x4-ssrnet", "bench-cave-x8-ssrnet"], ["bench-harvard-x4-ssrnet", "bench-harvard-x8-ssrnet"]),
    "puf-t20": (["bench-pavia-x8-ssrnet", "bench-pavia-x16-ssrnet"],
                ["bench-chikusei-x8-ssrnet", "bench-chikusei-x16-ssrnet"]),
    # wave 5: x16 on Chikusei and Pavia Centre
    "puf-s15": (["bench-pavia-x16-puformer", "bench-pavia-x16-psrt"],
                ["bench-pavia-x16-dct", "bench-pavia-x16-mimformer"]),
    "puf-s16": (["bench-chikusei-x16-puformer", "bench-chikusei-x16-psrt"],
                ["bench-chikusei-x16-dct", "bench-chikusei-x16-mimformer"]),
    "puf-s17": (["bench-pavia-x16-ssrnet"], ["bench-chikusei-x16-ssrnet"]),
    # --- the rest of the benchmark (with puf-t09..t14, t16, t19, t20 and d05 above) and the operator ablation
    #     beyond Chikusei / Pavia Centre x4, one job per GPU ---
    "puf-a01": (["abl-cave-x4-zeros"], ["abl-cave-x4-learned"]),
    "puf-a02": (["abl-harvard-x4-zeros"], ["abl-harvard-x4-learned"]),
    "puf-a03": (["abl-cave-x4-nophys"], ["abl-harvard-x4-nophys"]),
    "puf-a04": (["abl-chikusei-x8-zeros"], ["abl-chikusei-x8-learned"]),
    "puf-a05": (["abl-pavia-x8-zeros"], ["abl-pavia-x8-learned"]),
    "puf-a06": (["abl-chikusei-x8-nophys"], ["abl-pavia-x8-nophys"]),
    "puf-d06": (["bench-harvard-x4-dhif"], ["bench-cave-x4-dhif"]),
    "puf-d07": (["bench-harvard-x8-dhif"], ["bench-cave-x8-dhif"]),
    "puf-d08": (["bench-chikusei-x8-dhif"], ["bench-pavia-x8-dhif"]),
    # operator-conditioned training (Chikusei and Pavia Centre x4)
    "puf-r01": (["cond-chikusei-x4-puformer"], ["cond-chikusei-x4-nocond"]),
    "puf-r02": (["cond-pavia-x4-puformer"], ["cond-pavia-x4-nocond"]),
    "puf-r03": (["cond-chikusei-x4-zeros"], ["cond-chikusei-x4-mimformer"]),
    "puf-r04": (["cond-pavia-x4-zeros"], ["cond-pavia-x4-mimformer"]),
    "puf-r05": (["cond-chikusei-x4-omc"], ["cond-pavia-x4-omc"]),
    # architecture extensions: exact proximal data step, back-projected MSI residual as a prior input
    "puf-v01": (["abl-chikusei-x4-cg"], ["abl-pavia-x4-cg"]),
    "puf-v02": (["abl-chikusei-x4-mres"], ["abl-pavia-x4-mres"]),
    # the closed-form exact data step: alone, and with randomized training and observation-model conditioning
    "puf-k01": (["abl-chikusei-x4-kron"], ["abl-pavia-x4-kron"]),
    "puf-k02": (["cond-chikusei-x4-kron"], ["cond-pavia-x4-kron"]),
    "puf-k03": (["abl-chikusei-x4-kl"], ["abl-pavia-x4-kl"]),
    "puf-k04": (["cond-chikusei-x4-kl"], ["cond-pavia-x4-kl"]),
    # real data (NEON); the account must have run kaggle/neon/prep (puf-neon-data) and fit (puf-neon-fit)
    "puf-n01": (["bench-neon-x4-puformer"], ["bench-neon-x4-mimformer"]),
    "puf-n02": (["bench-neon-x4-psrt"], ["bench-neon-x4-dct"]),
    "puf-n03": (["bench-neon-x4-dhif"], ["abl-neon-x4-nophys"]),
    "puf-n04": (["abl-neon-x4-zeros"], ["abl-neon-x4-learned"]),
    "puf-n05": (["bench-neon-x4-ssrnet"], ["cond-neon-x4-puformer"]),
    # where the margin comes from: official recipes, swapped recipes, the prior alone
    "puf-g01": (["off-chikusei-x4-mimformer"], ["off-pavia-x4-mimformer"]),
    "puf-g02": (["off-chikusei-x4-psrt"], ["off-pavia-x4-psrt"]),
    "puf-g03": (["rec-chikusei-x4-mimformer"], ["rec-pavia-x4-mimformer"]),
    "puf-g04": (["rec-chikusei-x4-puformer"], ["rec-pavia-x4-puformer"]),
    "puf-g05": (["abl-chikusei-x4-unet"], ["abl-pavia-x4-unet"]),
    # MIMFormer seeds 1 and 2 at x4
    "puf-m01": (["seed1-chikusei-x4-mimformer"], ["seed2-chikusei-x4-mimformer"]),
    "puf-m02": (["seed1-pavia-x4-mimformer"], ["seed2-pavia-x4-mimformer"]),
    # further published networks (SMGU-Net, CLSNet, OTPNet, U2Net, MIMO-SST), x4
    **{f"puf-b{i:02d}": ([f"bench-chikusei-x4-{m}"], [f"bench-pavia-x4-{m}"]) for i, m in enumerate(NEW_METHODS, 1)},
    **{f"puf-b{i + 5:02d}": ([f"bench-cave-x4-{m}"], [f"bench-harvard-x4-{m}"]) for i, m in enumerate(NEW_METHODS, 1)},
    **{f"puf-b{i + 10:02d}": ([f"bench-chikusei-x8-{m}"], [f"bench-pavia-x8-{m}"])
       for i, m in enumerate([m for m in NEW_METHODS if m != "smgu"], 1)},
    # the wider operator family (anisotropic blur + misregistration): PUFormer, and MIMFormer on the same data
    "puf-r06": (["cond-chikusei-x4-reg"], ["cond-pavia-x4-reg"]),
    "puf-r07": (["cond-chikusei-x4-mimreg"], ["cond-pavia-x4-mimreg"]),
    # one network for any MSI sensor / any scale, and the sensor-specific references
    "puf-s20": (["sa-chikusei-x4-puformer"], ["sa-pavia-x4-puformer"]),
    "puf-s21": (["sa-chikusei-x4-mimformer"], ["sa-pavia-x4-mimformer"]),
    "puf-s22": (["spec-chikusei-x4-puformer-sentinel2"], ["spec-chikusei-x4-mimformer-sentinel2"]),
    "puf-s23": (["spec-chikusei-x4-ng-rgb"], ["spec-chikusei-x4-mimformer-rgb"]),
    "puf-s24": (["scl-chikusei-x4-ng"], ["scl-pavia-x4-ng"]),
    # the redesigned stage (null-space-gated prior + band-space interface + learned spectral data step)
    "puf-a10": (["arch-chikusei-x4-new"], ["arch-pavia-x4-new"]),
    "puf-a11": (["arch-chikusei-x4-nogate"], ["arch-pavia-x4-nogate"]),
    "puf-a12": (["cond-chikusei-x4-new"], ["cond-pavia-x4-new"]),
    "puf-a13": (["cond-chikusei-x4-regnew"], ["cond-pavia-x4-regnew"]),
    "puf-a14": (["sa-chikusei-x4-new"], ["sa-pavia-x4-new"]),
    # null-space transfer prior; confidence-weighted data step on corrupted observations, and its controls
    "puf-a15": (["arch-chikusei-x4-ntp"], ["arch-pavia-x4-ntp"]),
    "puf-a16": (["cond-chikusei-x4-cwd"], ["cond-pavia-x4-cwd"]),
    "puf-a17": (["cond-chikusei-x4-cor"], ["cond-pavia-x4-cor"]),
    "puf-a18": (["cond-chikusei-x4-mimcor"], ["cond-pavia-x4-mimcor"]),
    # does the gap survive 8 h of training on Chikusei as well? (one 8 h job per GPU)
    "puf-L02": (["long-chikusei-x4-ng"], ["long-chikusei-x4-mimformer"]),
    "puf-L03": (["long-pavia-x4-ng"], ["spec-chikusei-x4-ng-sentinel2"]),
    "puf-n27": (["nb-nogate-neon-x4"], ["cond-neon-x4-ng"]),          # needs the NEON data of amarnath10chinu
    "puf-a19": (["arch-chikusei-x4-unc"], ["arch-pavia-x4-unc"]),
    "puf-a20": (["arch-chikusei-x4-eon"], ["arch-pavia-x4-eon"]),
    "puf-a21": (["cond-chikusei-x4-eon"], ["cond-pavia-x4-eon"]),
    "puf-a22": (["abl-chikusei-x4-nomsi"], ["abl-pavia-x4-nomsi"]),
    "puf-a23": (["abl-chikusei-x4-nohsi"], ["abl-pavia-x4-nohsi"]),
    "puf-b15": (["bench-chikusei-x4-caformer"], ["bench-pavia-x4-caformer"]),
    "puf-b16": (["bench-cave-x4-caformer"], ["bench-harvard-x4-caformer"]),
    "puf-m2a": (["bench-chikusei-x4-m2dtn"], ["bench-pavia-x4-m2dtn"]),
    "puf-m2b": (["reg-chikusei-x4-m2dtn"], ["reg-pavia-x4-m2dtn"]),
    "puf-m2c": (["regf-chikusei-x4-m2dtn"], ["regf-pavia-x4-m2dtn"]),
}
# the adopted model in the robustness, recipe and uncertainty experiments
for _i, _j in enumerate(("cond-{d}-x4-ng", "cond-{d}-x4-regng", "sa-{d}-x4-ng", "cond-{d}-x4-cwdng", "cond-{d}-x4-corng",
                         "rec-{d}-x4-ng", "arch-{d}-x4-uncng")):
    SESSIONS[f"puf-n{20 + _i}"] = ([_j.format(d="chikusei")], [_j.format(d="pavia")])
# ExactFuse ablation, one session per variant (both datasets): puf-na-<variant>
for _v in NG_ABLATIONS:
    SESSIONS[f"puf-na-{_v}"] = ([f"abl-chikusei-x4-ng-{_v}"], [f"abl-pavia-x4-ng-{_v}"])
# the selected variant of the redesign everywhere else: puf-nb-<variant>-1..5
for _v in VARIANTS:
    SESSIONS[f"puf-nb-{_v}-1"] = ([f"nb-{_v}-cave-x4"], [f"nb-{_v}-harvard-x4"])
    SESSIONS[f"puf-nb-{_v}-2"] = ([f"nb-{_v}-cave-x8"], [f"nb-{_v}-chikusei-x8"])
    SESSIONS[f"puf-nb-{_v}-3"] = ([f"nb-{_v}-pavia-x8"], [f"nb-{_v}-chikusei-x16"])
    SESSIONS[f"puf-nb-{_v}-4"] = ([f"nb-{_v}-chikusei-x4-s1"], [f"nb-{_v}-chikusei-x4-s2"])
    SESSIONS[f"puf-nb-{_v}-5"] = ([f"nb-{_v}-pavia-x4-s1"], [f"nb-{_v}-pavia-x4-s2"])
