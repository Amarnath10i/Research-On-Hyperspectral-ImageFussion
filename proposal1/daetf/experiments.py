"""Experiment harness: ablations, efficiency profiling, statistics and tables.

What separates a Q1 submission from a demo is not the architecture, it is the
evidence around it. This module produces:

  * per-scene results, not just means, so comparisons can be paired
  * a paired Wilcoxon signed-rank test and Cohen's d against every baseline
  * bootstrap 95% confidence intervals on each mean
  * a component ablation with matched control arms
  * multi-seed repeats, reported as mean +/- std
  * cost accounting: parameters, GFLOPs, latency, peak GPU memory
  * cross-domain transfer with and without test-time adaptation
  * scale-factor generalisation
  * Markdown and LaTeX tables ready to paste into a manuscript

Dependencies are numpy/torch only; the statistics are implemented directly so
the module runs on a bare Kaggle image without scipy.stats.
"""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .engine import (evaluate_dataset, evaluate_with_tta, load_checkpoint,
                     set_seed, tiled_inference, train)
from .model import DAETFNet

METRICS = ("psnr", "ssim", "sam", "ergas")
HIGHER_IS_BETTER = {"psnr": True, "ssim": True, "sam": False, "ergas": False}


# ---------------------------------------------------------------- statistics
def bootstrap_ci(values: Sequence[float], n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean."""
    v = np.asarray(values, dtype=np.float64)
    if v.size < 2:
        return (float(v.mean()) if v.size else float("nan"),) * 2
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), \
        float(np.percentile(means, 100 * (1 - alpha / 2)))


def _normal_sf(z: float) -> float:
    """Upper-tail standard normal probability via the error function."""
    return 0.5 * math_erfc(z / (2 ** 0.5))


def math_erfc(x: float) -> float:
    import math
    return math.erfc(x)


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Paired Wilcoxon signed-rank test with a normal approximation.

    Paired over scenes: every method is scored on the same scenes, so pairing is
    the correct design and is far more sensitive than an unpaired test on the
    10-20 scenes these datasets provide.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    d = a - b
    d = d[d != 0]
    n = d.size
    if n < 1:
        return {"n": 0, "W": float("nan"), "z": float("nan"), "p": float("nan")}
    order = np.argsort(np.abs(d))
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1)
    # average ranks within ties of |d|
    absd = np.abs(d)[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and absd[j + 1] == absd[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = np.mean(np.arange(i + 1, j + 2))
        i = j + 1
    w_pos = ranks[d > 0].sum()
    w_neg = ranks[d < 0].sum()
    w = min(w_pos, w_neg)
    mu = n * (n + 1) / 4.0
    sigma = (n * (n + 1) * (2 * n + 1) / 24.0) ** 0.5
    z = (w - mu) / sigma if sigma > 0 else 0.0
    p = 2 * _normal_sf(abs(z))
    return {"n": float(n), "W": float(w), "z": float(z), "p": float(min(p, 1.0))}


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Paired Cohen's d (mean difference over the sd of the differences)."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("inf") if d.mean() else 0.0


def compare_methods(ours: List[Dict], theirs: List[Dict],
                    name_a: str = "ours", name_b: str = "baseline") -> Dict:
    """Paired comparison over the scenes both methods were scored on."""
    by_b = {r["scene"]: r for r in theirs}
    shared = [r for r in ours if r["scene"] in by_b]
    out = {"name_a": name_a, "name_b": name_b, "n_scenes": len(shared)}
    for m in METRICS:
        a = [r[m] for r in shared]
        b = [by_b[r["scene"]][m] for r in shared]
        test = wilcoxon_signed_rank(a, b)
        delta = float(np.mean(a) - np.mean(b))
        improved = delta > 0 if HIGHER_IS_BETTER[m] else delta < 0
        out[m] = {"mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
                  "delta": delta, "improved": bool(improved),
                  "p": test["p"], "z": test["z"], "d": cohens_d(a, b),
                  "ci_a": bootstrap_ci(a), "ci_b": bootstrap_ci(b)}
    return out


def summarise_rows(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Mean, std and bootstrap CI for each metric across scenes."""
    out = {}
    for m in METRICS:
        v = [r[m] for r in rows]
        lo, hi = bootstrap_ci(v)
        out[m] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                  "ci_lo": lo, "ci_hi": hi, "n": len(v)}
    return out


# ------------------------------------------------------------------ efficiency
def count_flops(model: nn.Module, lr_shape: Tuple[int, ...],
                msi_shape: Tuple[int, ...], device: str = "cpu") -> float:
    """Multiply-accumulate count for conv and linear layers, in GFLOPs.

    Implemented with forward hooks rather than an external dependency, so the
    number can be reported from a bare Kaggle image. Counts 2 FLOPs per MAC.
    """
    total = [0]

    def conv_hook(m, inp, out):
        out_elems = out.numel()
        k = m.weight.shape[2] * m.weight.shape[3]
        total[0] += 2 * out_elems * (m.in_channels // m.groups) * k

    def deconv_hook(m, inp, out):
        k = m.weight.shape[2] * m.weight.shape[3]
        total[0] += 2 * inp[0].numel() * (m.out_channels // m.groups) * k

    def lin_hook(m, inp, out):
        total[0] += 2 * out.numel() * m.in_features

    handles = []
    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            handles.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.ConvTranspose2d):
            handles.append(mod.register_forward_hook(deconv_hook))
        elif isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(lin_hook))

    model.eval()
    with torch.no_grad():
        model(torch.zeros(*lr_shape, device=device), torch.zeros(*msi_shape, device=device))
    for h in handles:
        h.remove()
    return total[0] / 1e9


@torch.no_grad()
def profile_model(model: DAETFNet, cfg: Config, device: str = "cuda",
                  hr: int = 512, warmup: int = 3, runs: int = 10) -> Dict[str, float]:
    """Parameters, GFLOPs, latency and peak memory for one full scene."""
    lr_shape = (1, cfg.bands, hr // cfg.scale, hr // cfg.scale)
    msi_shape = (1, cfg.msi_bands, hr, hr)
    gflops = count_flops(copy.deepcopy(model).to("cpu"), lr_shape, msi_shape, "cpu")

    model = model.to(device).eval()
    lr = torch.zeros(*lr_shape, device=device)
    msi = torch.zeros(*msi_shape, device=device)
    for _ in range(warmup):
        tiled_inference(model, lr, msi, cfg.scale)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(runs):
        tiled_inference(model, lr, msi, cfg.scale)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / runs
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if device == "cuda" else float("nan")
    return {"params_M": model.n_params() / 1e6, "gflops": gflops,
            "latency_s": dt, "peak_mem_MB": peak, "hr": hr}


# ------------------------------------------------------------------- ablations
ABLATIONS: List[Tuple[str, Dict]] = [
    ("full model", {}),
    ("w/o equivariant EFE", {"use_equivariant": False}),
    ("w/o Tucker TSSE", {"use_tsse": False}),
    ("w/o region-aware MoE", {"use_moe": False}),
    ("w/o wavelet FDRM", {"use_fdrm": False}),
    ("w/o back-projection", {"use_backprojection": False}),
    ("w/o degradation code", {"use_degradation_code": False}),
    ("w/o physics losses", {"use_physics": False}),
]


def run_ablation(base_cfg: Config, device: str = "cuda", iters: Optional[int] = None,
                 variants: Optional[List[Tuple[str, Dict]]] = None,
                 log_fn=print) -> List[Dict]:
    """Train each variant from scratch and evaluate in-domain and cross-domain.

    Every variant is trained with an identical budget, seed and data order, so
    differences are attributable to the component rather than to the schedule.
    """
    variants = variants or ABLATIONS
    results = []
    for name, overrides in variants:
        cfg = replace(base_cfg, **overrides)
        if iters:
            cfg.iters = iters
        cfg.out_dir = os.path.join(base_cfg.out_dir, "ablation",
                                   name.replace("/", "").replace(" ", "_"))
        log_fn(f"\n=== ablation: {name} ===")
        model, _ = train(cfg, device=device, log_fn=log_fn)
        row: Dict = {"variant": name, "params_M": model.n_params() / 1e6}
        src, src_rows = evaluate_dataset(model, cfg.source_root, cfg, "Test", device,
                                         verbose=False, return_rows=True)
        row["source"] = src
        row["source_rows"] = src_rows
        if cfg.target_root:
            tgt, tgt_rows = evaluate_dataset(model, cfg.target_root, cfg, "Test", device,
                                             verbose=False, return_rows=True)
            row["target"] = tgt
            row["target_rows"] = tgt_rows
        results.append(row)
        log_fn(f"  {name}: in-domain PSNR {src['psnr']:.3f} SAM {src['sam']:.3f}"
               + (f" | cross-domain PSNR {row['target']['psnr']:.3f} "
                  f"SAM {row['target']['sam']:.3f}" if cfg.target_root else ""))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return results


def run_multiseed(base_cfg: Config, seeds: Sequence[int] = (0, 1, 2),
                  device: str = "cuda", log_fn=print) -> Dict:
    """Repeat the full training run across seeds and report mean +/- std.

    A single run is not evidence; reviewers ask for variance.
    """
    runs = []
    for s in seeds:
        cfg = replace(base_cfg, seed=int(s))
        cfg.out_dir = os.path.join(base_cfg.out_dir, f"seed{s}")
        log_fn(f"\n=== seed {s} ===")
        model, _ = train(cfg, device=device, log_fn=log_fn)
        entry = {"seed": int(s)}
        entry["source"] = evaluate_dataset(model, cfg.source_root, cfg, "Test",
                                           device, verbose=False)
        if cfg.target_root:
            entry["target"] = evaluate_dataset(model, cfg.target_root, cfg, "Test",
                                               device, verbose=False)
        runs.append(entry)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    agg = {}
    for domain in ("source", "target"):
        if not all(domain in r for r in runs):
            continue
        agg[domain] = {m: {"mean": float(np.mean([r[domain][m] for r in runs])),
                           "std": float(np.std([r[domain][m] for r in runs], ddof=1))
                           if len(runs) > 1 else 0.0}
                       for m in METRICS}
    return {"runs": runs, "aggregate": agg}


def run_scale_generalisation(cfg: Config, ckpt: str, scales: Sequence[int] = (4, 8),
                             device: str = "cuda", log_fn=print) -> List[Dict]:
    """Evaluate a single trained model at scale factors it was not trained on.

    The v1 benchmark compared methods that were each run at a different scale
    factor (4, 8, 16 and 32), which is not a comparison at all. Here the factor
    is an explicit, reported axis.
    """
    out = []
    for s in scales:
        model, mcfg, _ = load_checkpoint(ckpt, device)
        mcfg.scale = s
        # the learned upsampler is tied to its training factor; rebuilding at a
        # new factor is only valid for the fixed-degradation evaluation path
        if s != cfg.scale:
            log_fn(f"  note: model was trained at x{cfg.scale}; evaluating at x{s} "
                   f"measures degradation-generalisation, not retrained performance")
        try:
            m = evaluate_dataset(model, mcfg.source_root, mcfg, "Test", device,
                                 verbose=False)
            out.append({"scale": s, **m})
        except Exception as exc:                       # shape mismatch at other factors
            log_fn(f"  x{s} not evaluable for this checkpoint: {exc}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return out


# ----------------------------------------------------------------------- tables
def markdown_table(headers: Sequence[str], rows: Sequence[Sequence]) -> str:
    head = "| " + " | ".join(str(h) for h in headers) + " |"
    sep = "|" + "|".join("---" for _ in headers) + "|"
    body = "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
    return "\n".join([head, sep, body])


def latex_table(headers: Sequence[str], rows: Sequence[Sequence],
                caption: str = "", label: str = "") -> str:
    cols = "l" + "c" * (len(headers) - 1)
    lines = [r"\begin{table}[t]", r"\centering",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             " & ".join(str(h) for h in headers) + r" \\", r"\midrule"]
    lines += [" & ".join(str(c) for c in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def comparison_table(entries: Dict[str, Dict[str, float]], fmt: str = "markdown",
                     caption: str = "", label: str = "") -> str:
    """entries: {method name -> {psnr, ssim, sam, ergas}} rendered with the
    best value in each column marked."""
    headers = ["Method", "PSNR (dB) up", "SSIM up", "SAM (deg) down", "ERGAS down"]
    best = {m: (max if HIGHER_IS_BETTER[m] else min)(
        e[m] for e in entries.values() if m in e) for m in METRICS}
    rows = []
    for name, e in entries.items():
        cells = [name]
        for m, prec in zip(METRICS, (3, 4, 3, 3)):
            val = e.get(m)
            if val is None:
                cells.append("-")
                continue
            txt = f"{val:.{prec}f}"
            if abs(val - best[m]) < 1e-9:
                txt = f"**{txt}**" if fmt == "markdown" else rf"\textbf{{{txt}}}"
            cells.append(txt)
        rows.append(cells)
    return (markdown_table(headers, rows) if fmt == "markdown"
            else latex_table(headers, rows, caption, label))


def ablation_table(results: List[Dict], fmt: str = "markdown") -> str:
    has_target = any("target" in r for r in results)
    headers = ["Variant", "Params (M)", "PSNR", "SAM", "ERGAS"]
    if has_target:
        headers += ["PSNR (cross)", "SAM (cross)", "ERGAS (cross)"]
    rows = []
    for r in results:
        cells = [r["variant"], f"{r['params_M']:.2f}",
                 f"{r['source']['psnr']:.3f}", f"{r['source']['sam']:.3f}",
                 f"{r['source']['ergas']:.3f}"]
        if has_target:
            t = r.get("target", {})
            cells += [f"{t.get('psnr', float('nan')):.3f}",
                      f"{t.get('sam', float('nan')):.3f}",
                      f"{t.get('ergas', float('nan')):.3f}"]
        rows.append(cells)
    return (markdown_table(headers, rows) if fmt == "markdown"
            else latex_table(headers, rows, "Component ablation.", "tab:ablation"))


def significance_table(comparisons: List[Dict], metric: str = "psnr",
                       fmt: str = "markdown") -> str:
    headers = ["Baseline", f"ours {metric}", f"baseline {metric}", "delta",
               "Wilcoxon p", "Cohen d", "n"]
    rows = []
    for c in comparisons:
        e = c[metric]
        star = "***" if e["p"] < 0.001 else "**" if e["p"] < 0.01 else \
               "*" if e["p"] < 0.05 else "n.s."
        rows.append([c["name_b"], f"{e['mean_a']:.3f}", f"{e['mean_b']:.3f}",
                     f"{e['delta']:+.3f}", f"{e['p']:.4f} {star}",
                     f"{e['d']:.2f}", int(c["n_scenes"])])
    return (markdown_table(headers, rows) if fmt == "markdown"
            else latex_table(headers, rows, f"Paired significance on {metric}.",
                             f"tab:sig_{metric}"))


# --------------------------------------------------------------------- reports
def environment_report() -> Dict[str, str]:
    """Everything a reviewer needs to reproduce the numbers."""
    import platform
    import sys
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "torch": torch.__version__, "numpy": np.__version__,
            "cuda_available": str(torch.cuda.is_available())}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda or "unknown"
        info["gpu_mem_GB"] = f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f}"
    return info


def save_results(path: str, payload: Dict) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, default=default)
    return path


def write_report(path: str, title: str, sections: List[Tuple[str, str]]) -> str:
    """Assemble a Markdown report from (heading, body) pairs."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    parts = [f"# {title}", ""]
    for heading, body in sections:
        parts += [f"## {heading}", "", body, ""]
    text = "\n".join(parts)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text
