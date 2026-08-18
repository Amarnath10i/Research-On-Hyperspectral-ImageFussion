"""Push a notebook to Kaggle, watch it, repair common failures, re-run.

    python tools/kaggle_autorun.py --notebook proposal1/notebooks/DAETF_Net_Kaggle_P100.ipynb

The loop is: push -> poll until the kernel stops -> if it failed, pull the log,
match the error against a repair rule, apply the repair, push again. It stops
on success, when no rule matches, or after --max-attempts.

Credentials are read from, in order:
  1. KAGGLE_USERNAME / KAGGLE_KEY environment variables
  2. ~/.kaggle/kaggle.json           (the standard location)
  3. KAGGLE_CONFIG_DIR/kaggle.json

Nothing is ever written into this file or the repository. Treat any key you
have pasted into a chat, ticket or terminal transcript as compromised and
rotate it at https://www.kaggle.com/settings once you are done.

Every attempt is archived under --state-dir: the notebook as pushed, the log,
and the diff of what was repaired, so the run is auditable afterwards.

IMPORTANT - accelerator selection. kernel-metadata carries `enable_gpu`, and
Kaggle resolves that to whatever its current default GPU is. The `accelerator`
field is accepted by the API and then ignored, so a push cannot request a
specific GPU - and worse, it RESETS a choice made in the web UI back to the
default. Since torch >= 2.6 ships no sm_60 kernels, a push can silently turn a
working T4 setup into a broken P100 one.

  To run on a T4: choose it in the UI (Settings -> Accelerator -> "GPU T4 x2"),
  start the run from the UI, and watch it from here with --watch-only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------- credentials
def load_credentials(verbose: bool = True) -> Tuple[str, str]:
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if user and key:
        if verbose:
            print(f"[auth] using environment variables (user: {user})")
        return user, key

    candidates = [
        os.path.join(os.environ.get("KAGGLE_CONFIG_DIR", ""), "kaggle.json"),
        os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                creds = json.load(f)
            if "username" in creds and "key" in creds:
                os.environ["KAGGLE_USERNAME"] = creds["username"]
                os.environ["KAGGLE_KEY"] = creds["key"]
                if verbose:
                    print(f"[auth] using {path} (user: {creds['username']})")
                return creds["username"], creds["key"]

    raise SystemExit(
        "No Kaggle credentials found.\n"
        "  Either set KAGGLE_USERNAME and KAGGLE_KEY, or save your token to\n"
        f"  {os.path.join(os.path.expanduser('~'), '.kaggle', 'kaggle.json')}\n"
        "  Download it from https://www.kaggle.com/settings -> API -> Create New Token"
    )


def kaggle_api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("[setup] installing the kaggle package ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"],
                       check=True)
        from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


# ------------------------------------------------------------- repair rules
@dataclass
class Repair:
    """One known failure mode and how to fix the run that hit it.

    `apply(nb, match, ctx)` may edit the notebook, mutate `ctx` (which carries
    push settings such as the accelerator), or both. It returns True when it
    actually changed something - returning False stops the loop rather than
    re-pushing an identical run.
    """
    name: str
    pattern: str
    apply: Callable[[dict, "re.Match", dict], bool]
    note: str = ""


# Escalation order, oldest architecture first. On an "unsupported GPU" failure
# we move to the NEXT entry, i.e. toward newer hardware that current PyTorch
# still ships kernels for. Pascal (P100, sm_60) was dropped by torch >= 2.6,
# so P100 escalates to T4 (sm_75); a T4 failure has nowhere better to go and
# the loop stops rather than churning.
#
# MEASURED CAVEAT: pushing "accelerator": "nvidiaTeslaT4" in kernel-metadata
# is accepted by the API but Kaggle assigned a P100 regardless. The accelerator
# appears to be a per-kernel UI setting that the push API does not override, so
# this escalation is best-effort. If it does not take, set it once by hand:
#   open the kernel -> Settings -> Accelerator -> "GPU T4 x2" -> Save,
# after which subsequent pushes reuse that setting.
ACCELERATOR_FALLBACK = ["nvidiaTeslaP100", "nvidiaTeslaT4"]
ACCELERATOR_CHOICES = ["nvidiaTeslaT4", "nvidiaTeslaP100", "none", ""]


def _iter_code_cells(nb: dict):
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            yield cell


def _cell_text(cell: dict) -> str:
    return "".join(cell.get("source", []))


def _set_cell(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def _is_magic_cell(cell: dict) -> bool:
    """True when the cell opens with a cell magic such as %%writefile.

    A cell magic is only recognised on the very first line, so prepending
    anything to such a cell silently turns it into an unknown *line* magic and
    the cell fails with 'Line magic function `%%writefile` not found'. Repairs
    must insert a separate cell instead of editing these.
    """
    src = cell.get("source") or [""]
    return src[0].lstrip().startswith("%%")


def _new_code_cell(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


def _insert_cell_before(nb: dict, target: dict, text: str) -> bool:
    """Insert a new code cell immediately before `target`."""
    cells = nb.setdefault("cells", [])
    for idx, cell in enumerate(cells):
        if cell is target:
            cells.insert(idx, _new_code_cell(text))
            return True
    return False


def _prepend_install(nb: dict, package: str) -> bool:
    """Add a pip install for a missing module as the first code cell."""
    line = f"!pip install -q {package}\n"
    if line in "".join(_cell_text(c) for c in _iter_code_cells(nb)):
        return False
    for cell in _iter_code_cells(nb):
        if _is_magic_cell(cell):
            return _insert_cell_before(nb, cell, line)
        _set_cell(cell, line + _cell_text(cell))
        return True
    return False


def _fix_missing_module(nb: dict, m: "re.Match", ctx: dict) -> bool:
    mod = m.group(1)
    alias = {"cv2": "opencv-python-headless", "skimage": "scikit-image",
             "PIL": "pillow", "sklearn": "scikit-learn", "yaml": "pyyaml"}
    return _prepend_install(nb, alias.get(mod, mod))


def _switch_accelerator(nb: dict, _m: "re.Match", ctx: dict) -> bool:
    """The GPU is not supported by the installed torch build.

    torch >= 2.6 dropped Pascal, so a P100 raises 'no kernel image is available'
    on every CUDA op. No notebook change can fix that - the run needs different
    hardware, so move to the next accelerator in the fallback order.
    """
    current = ctx.get("accelerator") or ACCELERATOR_FALLBACK[0]
    try:
        nxt = ACCELERATOR_FALLBACK[ACCELERATOR_FALLBACK.index(current) + 1]
    except (ValueError, IndexError):
        print(f"  no newer accelerator to escalate to from {current!r}; "
              f"this needs different hardware, not a code change")
        return False
    ctx["accelerator"] = nxt
    print(f"  escalating accelerator: {current} -> {nxt}")
    print("  NOTE: Kaggle has been observed to ignore this field and assign "
          "hardware of its own choosing.\n"
          "        If the next attempt reports the same GPU, set it by hand: "
          "open the kernel ->\n"
          "        Settings -> Accelerator -> 'GPU T4 x2' -> Save, then re-run.")
    return True


def _sub_in_plain_cells(nb: dict, fn: Callable[[str], str]) -> bool:
    """Apply a text transform to ordinary code cells only.

    Cell-magic cells hold the embedded package source; rewriting them would
    edit library defaults as a side effect of tuning a run parameter, so the
    repairs deliberately touch only the notebook's own configuration.
    """
    changed = False
    for cell in _iter_code_cells(nb):
        if _is_magic_cell(cell):
            continue
        text = _cell_text(cell)
        new = fn(text)
        if new != text:
            _set_cell(cell, new)
            changed = True
    return changed


def _halve_batch(nb: dict, _m: "re.Match", ctx: dict) -> bool:
    """CUDA OOM: halve the batch size and shrink the inference tile."""
    def fn(text: str) -> str:
        text = re.sub(r"\bbatch=(\d+)",
                      lambda x: f"batch={max(1, int(x.group(1)) // 2)}", text)
        return re.sub(r"\btile_hr=(\d+)",
                      lambda x: f"tile_hr={max(64, int(x.group(1)) // 2)}", text)
    return _sub_in_plain_cells(nb, fn)


def _disable_amp(nb: dict, _m: "re.Match", ctx: dict) -> bool:
    """fp16 overflow/NaN: fall back to fp32."""
    return _sub_in_plain_cells(nb, lambda t: t.replace("amp=True", "amp=False"))


def _drop_workers(nb: dict, _m: "re.Match", ctx: dict) -> bool:
    """DataLoader worker crash / shared-memory exhaustion: run in-process."""
    return _sub_in_plain_cells(nb, lambda t: re.sub(r"\bworkers=\d+", "workers=0", t))


def _relax_dataset_discovery(nb: dict, _m: "re.Match", ctx: dict) -> bool:
    """Dataset not found: print what IS attached so the next run can be fixed,
    and let discovery search every attached input."""
    probe = (
        "# injected by kaggle_autorun: show how the inputs are actually mounted\n"
        "import os\n"
        "print('attached inputs:')\n"
        "for base, dirs, files in os.walk('/kaggle/input'):\n"
        "    depth = base.rstrip('/').count('/') - 2\n"
        "    if depth > 3:\n"
        "        dirs[:] = []\n"
        "        continue\n"
        "    print('  ' * depth, base, sorted(dirs)[:6], len(files), 'files')\n"
    )
    for cell in _iter_code_cells(nb):
        if "attached inputs:" in _cell_text(cell):
            return False
    # target the cell that resolves the config, but never a cell-magic cell
    for cell in _iter_code_cells(nb):
        if _is_magic_cell(cell):
            continue
        if "cfg.resolve(" in _cell_text(cell):
            return _insert_cell_before(nb, cell, probe)
    return False


def _shrink_run(nb: dict, _m: "re.Match", ctx: dict) -> bool:
    """Kernel exceeded its time limit: cut the iteration budget in half (floor 500)."""
    return _sub_in_plain_cells(
        nb, lambda t: re.sub(r"\biters=(\d+)",
                             lambda x: f"iters={max(500, int(x.group(1)) // 2)}", t))


REPAIRS: List[Repair] = [
    Repair("missing-module", r"ModuleNotFoundError: No module named '([\w\.]+)'",
           _fix_missing_module, "pip install the missing package"),
    Repair("gpu-unsupported",
           r"(no kernel image is available for execution|"
           r"CUDA capability sm_\d+ is not compatible|"
           r"AcceleratorError: CUDA error: no kernel image)",
           _switch_accelerator, "switch to a supported accelerator"),
    Repair("cuda-oom", r"(CUDA out of memory|CUDA error: out of memory)",
           _halve_batch, "halve batch size and inference tile"),
    Repair("amp-nan", r"(Attempting to unscale FP16 gradients|found (inf|nan)|"
                      r"Gradient overflow)", _disable_amp, "disable fp16 AMP"),
    Repair("dataloader-workers", r"(DataLoader worker \(pid \d+\) is killed|"
                                 r"unable to open shared memory|"
                                 r"RuntimeError: DataLoader worker)",
           _drop_workers, "set num_workers to 0"),
    Repair("dataset-missing", r"(FileNotFoundError: no dataset matching|"
                              r"no usable split under|no matched pairs under)",
           _relax_dataset_discovery, "probe the attached inputs"),
    Repair("timeout", r"(exceeded the maximum|KernelRunTimeExceeded|"
                      r"Your notebook tried to allocate more)",
           _shrink_run, "halve the training budget"),
]


def diagnose(log: str) -> Optional[Tuple[Repair, "re.Match"]]:
    for rule in REPAIRS:
        m = re.search(rule.pattern, log, re.I)
        if m:
            return rule, m
    return None


# ---------------------------------------------------------------- kaggle ops
def slug_for(username: str, notebook_path: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit if "/" in explicit else f"{username}/{explicit}"
    stem = os.path.splitext(os.path.basename(notebook_path))[0]
    clean = re.sub(r"[^a-z0-9-]+", "-", stem.lower()).strip("-")[:50]
    return f"{username}/{clean}"


def push(api, notebook_path: str, slug: str, work_dir: str,
         datasets: List[str], models: List[str], gpu: bool = True,
         private: bool = True, internet: bool = True,
         accelerator: Optional[str] = None) -> None:
    """Push one notebook from an isolated directory, so the Kaggle client
    cannot pick up any other file."""
    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir, exist_ok=True)
    fname = os.path.basename(notebook_path)
    shutil.copy2(notebook_path, os.path.join(work_dir, fname))
    meta = {
        "id": slug,
        "title": slug.split("/")[-1],
        "code_file": fname,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": bool(private),
        "enable_gpu": bool(gpu),
        "enable_internet": bool(internet),
        "dataset_sources": datasets,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": models,
    }
    if accelerator:
        # only emitted when asked for: older Kaggle clients reject the field
        meta["accelerator"] = accelerator
    with open(os.path.join(work_dir, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    try:
        api.kernels_push(work_dir)
    except TypeError:
        if not accelerator:
            raise
        print("  (this kaggle client rejects the accelerator field; retrying "
              "without it - set the accelerator in the notebook UI instead)")
        meta.pop("accelerator", None)
        with open(os.path.join(work_dir, "kernel-metadata.json"), "w",
                  encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        api.kernels_push(work_dir)


def status_of(api, slug: str) -> str:
    try:
        resp = api.kernels_status(slug)
    except Exception as exc:                      # transient API hiccup
        return f"unknown ({exc})"
    raw = getattr(resp, "status", None)
    if raw is None and isinstance(resp, dict):
        raw = resp.get("status")
    return str(raw or resp).lower().replace("kernelworkerstatus.", "")


def fetch_log(api, slug: str, dest: str) -> str:
    """Download the kernel output and return every log/text file concatenated."""
    os.makedirs(dest, exist_ok=True)
    try:
        api.kernels_output(slug, path=dest)
    except Exception as exc:
        return f"<could not download output: {exc}>"
    chunks = []
    for root, _, files in os.walk(dest):
        for fn in files:
            if fn.endswith((".log", ".txt", ".json")):
                p = os.path.join(root, fn)
                try:
                    with open(p, "r", encoding="utf-8", errors="replace") as f:
                        chunks.append(f"----- {fn} -----\n{f.read()}")
                except OSError:
                    pass
    return "\n".join(chunks) or "<no log files in the kernel output>"


def wait_for_finish(api, slug: str, poll: int, timeout_min: int,
                    log_fn=print) -> str:
    deadline = time.time() + timeout_min * 60
    last = None
    while time.time() < deadline:
        st = status_of(api, slug)
        if st != last:
            log_fn(f"    status: {st}")
            last = st
        if any(k in st for k in ("complete", "error", "cancel", "fail")):
            return st
        time.sleep(poll)
    return "timeout"


# --------------------------------------------------------------------- main
def run(args) -> int:
    username, _ = load_credentials()
    api = kaggle_api()

    nb_path = os.path.abspath(args.notebook)
    if not os.path.exists(nb_path):
        raise SystemExit(f"notebook not found: {nb_path}")
    slug = slug_for(username, nb_path, args.slug)
    state = os.path.abspath(args.state_dir)
    os.makedirs(state, exist_ok=True)
    print(f"[run] notebook {nb_path}")
    print(f"[run] kernel   https://www.kaggle.com/{slug}")
    print(f"[run] state    {state}")

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # settings a repair may change between attempts
    ctx: Dict[str, object] = {"accelerator": args.accelerator}
    if args.accelerator:
        print(f"[run] accelerator {args.accelerator}")

    history = []
    for attempt in range(1, args.max_attempts + 1):
        print(f"\n=== attempt {attempt}/{args.max_attempts} ===")
        att_dir = os.path.join(state, f"attempt{attempt:02d}")
        os.makedirs(att_dir, exist_ok=True)
        pushed = os.path.join(att_dir, os.path.basename(nb_path))
        with open(pushed, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)

        if args.watch_only:
            print("  watch-only: NOT pushing (a push resets the accelerator "
                  "to Kaggle's default, which is currently a P100)")
        else:
            print("  pushing ...")
            push(api, pushed, slug, os.path.join(state, "_push"),
                 args.dataset or [], args.model or [], gpu=not args.no_gpu,
                 private=not args.public, internet=not args.no_internet,
                 accelerator=ctx.get("accelerator"))

        print("  waiting ...")
        st = wait_for_finish(api, slug, args.poll, args.timeout,
                             log_fn=lambda s: print("  " + s))
        entry = {"attempt": attempt, "status": st,
                 "accelerator": ctx.get("accelerator")}
        print(f"  finished: {st}")

        log = fetch_log(api, slug, os.path.join(att_dir, "output"))
        with open(os.path.join(att_dir, "kernel.log"), "w", encoding="utf-8") as f:
            f.write(log)

        if "complete" in st:
            entry["result"] = "success"
            history.append(entry)
            print(f"\n[done] kernel completed. Output saved to {att_dir}/output")
            # keep the working notebook in sync with whatever finally worked
            if args.write_back:
                shutil.copy2(pushed, nb_path)
                print(f"[done] repaired notebook written back to {nb_path}")
            break

        found = diagnose(log)
        if not found:
            entry["result"] = "unrecognised-failure"
            history.append(entry)
            tail = "\n".join(log.strip().splitlines()[-40:])
            print("\n[stop] failure did not match any repair rule. Log tail:\n")
            print(tail)
            break

        rule, match = found
        print(f"  diagnosed: {rule.name} -> {rule.note}")
        if not rule.apply(nb, match, ctx):
            entry["result"] = f"repair-{rule.name}-noop"
            history.append(entry)
            print("[stop] the repair changed nothing; stopping to avoid a loop.")
            break
        entry["result"] = f"repaired-{rule.name}"
        entry["matched"] = match.group(0)[:200]
        history.append(entry)
        if args.watch_only:
            fixed = os.path.join(att_dir, "repaired-" + os.path.basename(nb_path))
            with open(fixed, "w", encoding="utf-8") as f:
                json.dump(nb, f, indent=1, ensure_ascii=False)
            print(f"  watch-only: repair written to {fixed}")
            print("  upload it yourself and re-run - pushing from here would "
                  "reset the accelerator")
            break
        print("  repaired; retrying")
    else:
        print(f"\n[stop] gave up after {args.max_attempts} attempts")

    with open(os.path.join(state, "history.json"), "w", encoding="utf-8") as f:
        json.dump({"slug": slug, "notebook": nb_path, "attempts": history},
                  f, indent=1)
    print(f"\n[run] history -> {os.path.join(state, 'history.json')}")
    return 0 if history and history[-1].get("result") == "success" else 1


def main() -> int:
    p = argparse.ArgumentParser(
        description="Push a notebook to Kaggle, repair failures, re-run.")
    p.add_argument("--notebook", required=True, help="path to the .ipynb")
    p.add_argument("--slug", help="kernel slug (default: derived from the filename)")
    p.add_argument("--dataset", action="append",
                   help="dataset source, e.g. owner/cave-dataset-2 (repeatable)")
    p.add_argument("--model", action="append", help="model source (repeatable)")
    p.add_argument("--max-attempts", type=int, default=5)
    p.add_argument("--poll", type=int, default=60, help="seconds between polls")
    p.add_argument("--timeout", type=int, default=540, help="minutes to wait per run")
    p.add_argument("--state-dir", default="kaggle_runs")
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--accelerator", default="nvidiaTeslaT4",
                   choices=ACCELERATOR_CHOICES,
                   help="Kaggle accelerator. Defaults to a T4: torch >= 2.6 "
                        "ships no kernels for the P100 (sm_60), so a P100 run "
                        "fails on every CUDA op regardless of the code.")
    p.add_argument("--public", action="store_true")
    p.add_argument("--no-internet", action="store_true")
    p.add_argument("--watch-only", action="store_true",
                   help="Do not push; only poll an existing kernel run, fetch "
                        "its log and diagnose. Use this when the accelerator "
                        "was chosen in the Kaggle UI: a push sends "
                        "enable_gpu=true, which Kaggle resolves to its default "
                        "GPU and silently overrides that choice.")
    p.add_argument("--write-back", action="store_true",
                   help="overwrite the source notebook with the repaired version")
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
