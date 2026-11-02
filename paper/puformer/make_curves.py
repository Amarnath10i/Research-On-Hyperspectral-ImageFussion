"""Validation PSNR against training time of every network on Chikusei and Pavia Centre x4 (equal budget), from the
history.json of each run: writes results/q1/curves.json, small enough to embed in the paper build.

    python make_curves.py
"""
import json
import os

import make_q1_tables as t

OUT = os.path.join(t.RES, "curves.json")


def main():
    res = {}
    for d in ("chikusei", "pavia"):
        for m, lab, _ in t.ALL:
            if m in ("bicubic", "gsa"):
                continue
            job = f"bench-{d}-x4-{m}"
            for pat, rep in t.ADOPT:                    # ExactFuse: the adopted run
                if t.re.fullmatch(pat, job):
                    job = t.re.sub(pat, rep, job)
            p = os.path.join(t.RES, job, "history.json")
            if not os.path.exists(p):
                continue
            h = json.load(open(p))
            pts = [(round(x["hours"], 3), round(x["val_PSNR"], 3)) for x in h if "val_PSNR" in x and "hours" in x]
            if pts:
                res.setdefault(d, {})[lab] = pts
    with open(OUT, "w") as f:
        json.dump(res, f, separators=(",", ":"))
    print({d: {k: len(v) for k, v in r.items()} for d, r in res.items()})


if __name__ == "__main__":
    main()
