"""Robust checkpoint manager for Kaggle (and local) training.

Kaggle kernels have a hard 12-hour time limit.  When the limit is hit,
the kernel is killed with SIGTERM.  This module:

1. Saves checkpoints periodically (not just at log_every — that can miss
   the save window entirely).
2. Registers a SIGTERM handler that force-saves before the kernel dies.
3. On resume, prints a clear indicator showing what step we're at and
   how much training was completed.

Usage in the training loop:

    from common.hsifusion.checkpoint import CheckpointManager

    ckpt = CheckpointManager(cfg.out_dir, cfg.name, device=device)
    ckpt.maybe_resume(model, opt, scaler)   # restores state if checkpoint exists

    for step in range(start_step, cfg.iters + 1):
        ...
        ckpt.save(model, opt, scaler, step, best, history,
                  every=2000, signal_save=True)

    ckpt.save_final(model, cfg, srf, total_params)
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn


class CheckpointManager:
    """Manages training checkpoints with Kaggle-aware auto-save."""

    def __init__(self, out_dir: str, name: str, device: str = "cuda"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.device = device

        self.ckpt_path = self.out_dir / f"{name}_checkpoint.pth"
        self.best_path = self.out_dir / f"{name}_best.pth"
        self.final_path = self.out_dir / f"{name}_final.pth"
        self.history_path = self.out_dir / "history.json"

        # Kaggle time tracking
        self._t0 = time.time()
        self._kaggle_limit = 12 * 3600 - 300  # 11h55m (save 5min before kill)
        self._signal_saved = False
        self._signal_model = None
        self._signal_opt = None
        self._signal_scaler = None
        self._signal_get_step = None

        # register SIGTERM handler for Kaggle
        self._register_signal_handler()

    def _register_signal_handler(self):
        """Register a SIGTERM handler that force-saves before Kaggle kills us."""
        original_sigterm = signal.getsignal(signal.SIGTERM)

        def _handler(signum, frame):
            if self._signal_saved:
                return
            print(f"\n[KAGGLE] Received signal {signum} — force-saving checkpoint...")
            try:
                if (self._signal_model is not None and
                        self._signal_opt is not None and
                        self._signal_get_step is not None):
                    step = self._signal_get_step()
                    self.save(
                        self._signal_model, self._signal_opt,
                        self._signal_scaler, step,
                        best=-1e9, history={},
                        force=True
                    )
                    print(f"[KAGGLE] Checkpoint saved at step {step}")
                    self._signal_saved = True
            except Exception as e:
                print(f"[KAGGLE] Failed to save: {e}")
            # call original handler
            if callable(original_sigterm):
                original_sigterm(signum, frame)

        # only register on Linux (Kaggle); Windows doesn't support SIGTERM
        if sys.platform != "win32":
            try:
                signal.signal(signal.SIGTERM, _handler)
            except (OSError, ValueError):
                pass  # can't set signal in some environments

    def _time_remaining(self) -> float:
        """Seconds remaining before Kaggle time limit (estimate)."""
        elapsed = time.time() - self._t0
        return max(0.0, self._kaggle_limit - elapsed)

    def maybe_resume(self, model: nn.Module, opt, scaler=None,
                     log_fn=print) -> int:
        """Load checkpoint if it exists. Returns the start step."""
        if not self.ckpt_path.exists():
            log_fn(f"[ckpt] No checkpoint found at {self.ckpt_path} — training from scratch")
            return 1

        log_fn(f"[ckpt] Resuming from {self.ckpt_path}")
        ckpt = torch.load(self.ckpt_path, map_location=self.device)

        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        if scaler is not None and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])

        step = ckpt.get("step", 0)
        best = ckpt.get("best", -1e9)
        history = ckpt.get("history", {})

        total_iters = ckpt.get("cfg", {}).get("iters", "?")
        pct = step / max(total_iters, 1) * 100 if isinstance(total_iters, int) else 0
        log_fn(f"[ckpt] Resumed at step {step}/{total_iters} ({pct:.1f}% complete)")

        return step + 1, best, history

    def save(self, model, opt, scaler, step, best, history,
             force=False, every=2000, signal_save=True):
        """Save checkpoint periodically or on signal.

        Args:
            every: save every N steps (in addition to log_every saves).
            signal_save: register model/opt for emergency SIGTERM save.
            force: save regardless of step/time conditions.
        """
        # register for signal handler
        if signal_save:
            self._signal_model = model
            self._signal_opt = opt
            self._signal_scaler = scaler
            self._signal_get_step = lambda: step

        # periodic save
        if not force and step % every != 0:
            return

        # Kaggle time check: save more frequently if time is running out
        remaining = self._time_remaining()
        if remaining < 1800 and not force:  # <30min left
            # save every 500 steps in the danger zone
            if step % 500 != 0:
                return

        save_dict = {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "step": step,
            "best": best,
            "history": history,
            "wall_time": time.time() - self._t0,
        }
        if scaler is not None:
            save_dict["scaler"] = scaler.state_dict()

        # atomic save: write to temp then rename
        tmp_path = self.ckpt_path.with_suffix(".tmp")
        torch.save(save_dict, tmp_path)
        tmp_path.rename(self.ckpt_path)

    def save_best(self, model, cfg, srf, val_metrics, log_fn=print):
        """Save best model checkpoint."""
        torch.save({
            "model": model.state_dict(),
            "cfg": cfg.to_dict(),
            "srf": srf,
            "val": val_metrics,
        }, self.best_path)
        log_fn(f"[ckpt] New best model saved: PSNR={val_metrics['psnr']:.3f}")

    def save_final(self, model, cfg, srf, total_params, log_fn=print):
        """Save final model and history."""
        torch.save({
            "model": model.state_dict(),
            "cfg": cfg.to_dict(),
            "srf": srf,
            "params": total_params,
        }, self.final_path)

        # save history (already in checkpoint, but also standalone)
        ckpt_path = self.ckpt_path
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device)
            history = ckpt.get("history", {})
            with open(self.history_path, "w") as f:
                json.dump(history, f, indent=1, default=str)

        log_fn(f"[ckpt] Final model saved to {self.final_path}")

    def status(self) -> Dict:
        """Return current checkpoint status."""
        info = {"ckpt_exists": self.ckpt_path.exists()}
        if self.ckpt_path.exists():
            ckpt = torch.load(self.ckpt_path, map_location=self.device)
            info["step"] = ckpt.get("step", 0)
            info["best"] = ckpt.get("best", -1e9)
            info["wall_time"] = ckpt.get("wall_time", 0)
        info["time_remaining"] = self._time_remaining()
        return info

    def print_status(self, log_fn=print):
        """Print a human-readable status line."""
        s = self.status()
        if s["ckpt_exists"]:
            log_fn(f"[ckpt] Step {s['step']}, best={s['best']:.3f}, "
                   f"wall={s.get('wall_time', 0)/3600:.1f}h, "
                   f"remaining={s['time_remaining']/3600:.1f}h")
        else:
            log_fn(f"[ckpt] No checkpoint. Time remaining: "
                   f"{s['time_remaining']/3600:.1f}h")