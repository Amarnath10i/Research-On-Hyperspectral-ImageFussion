"""Inference and the final test report, shared by train.py and test.py."""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from .metrics import evaluate
from .ops import Degradation


@torch.no_grad()
def predict(model, lr, ms, bs=1):
    return torch.cat([model(lr[i:i + bs], ms[i:i + bs]) for i in range(0, lr.shape[0], bs)])


@torch.no_grad()
def predict_tta(model, lr, ms, bs=1):
    """Identity + transpose. Transposing keeps the x4 decimation phase, so D stays exact."""
    t = lambda z: z.transpose(-1, -2)
    return 0.5 * (predict(model, lr, ms, bs) + t(predict(model, t(lr), t(ms), bs)))


def _psnr(a, b):
    return float(10 * torch.log10(1.0 / torch.mean((a.double() - b.double()) ** 2)))


@torch.no_grad()
def consistency(pred, lr, ms, deg: Degradation):
    """How well the fused image reproduces its own inputs (same operators as the simulation)."""
    pred = pred.clamp(0, 1)
    h = np.mean([_psnr(deg.D(pred[i:i + 1], pad_mode="reflect"), lr[i:i + 1]) for i in range(len(pred))])
    m = np.mean([_psnr(deg.R(pred[i:i + 1]), ms[i:i + 1]) for i in range(len(pred))])
    return dict(cons_H_dB=float(h), cons_M_dB=float(m))


@torch.no_grad()
def final_test(model, test, deg: Degradation, out_dir: str | None = None, full: bool = True):
    """Plain and self-ensemble (identity + transpose) test metrics, per image, plus consistency."""
    lr, ms, gt = test
    if lr.is_cuda:
        torch.cuda.synchronize()
    t1 = time.time()
    pred = predict(model, lr, ms)
    if lr.is_cuda:
        torch.cuda.synchronize()
    infer_s = (time.time() - t1) / lr.shape[0]
    pred_tta = predict_tta(model, lr, ms)
    res = dict(test=evaluate(gt, pred, full), test_tta=evaluate(gt, pred_tta, full),
               test_per_image=[evaluate(gt[i:i + 1], pred[i:i + 1], full) for i in range(len(gt))],
               test_per_image_tta=[evaluate(gt[i:i + 1], pred_tta[i:i + 1], full) for i in range(len(gt))],
               consistency=consistency(pred, lr, ms, deg), consistency_tta=consistency(pred_tta, lr, ms, deg),
               infer_s_per_256=infer_s)
    if out_dir:
        np.save(os.path.join(out_dir, "test_pred.npy"), pred.clamp(0, 1).float().cpu().numpy())
        np.save(os.path.join(out_dir, "test_pred_tta.npy"), pred_tta.clamp(0, 1).float().cpu().numpy())
    return res
