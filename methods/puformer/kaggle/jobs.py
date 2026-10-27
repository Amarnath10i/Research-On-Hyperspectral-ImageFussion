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
    struct = ("--pad", "--physics", "--res_input", "--memory", "--mixer", "--op", "--stages", "--omc", "--dstep", "--cg_iters", "--mres")
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
        n = f"long-pavia-x4-{m}"
        jobs[n] = job(n, "pavia", 4, m, hours=8.0)
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
    "puf-t11": (["bench-harvard-x8-puformer"], ["bench-harvard-x8-mimformer"]),
    "puf-t12": (["bench-harvard-x8-psrt"], ["bench-harvard-x8-dct"]),
    "puf-t13": (["bench-pavia-x16-puformer"], ["bench-pavia-x16-mimformer"]),
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
}
