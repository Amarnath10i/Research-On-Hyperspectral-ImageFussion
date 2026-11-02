"""How different is the architecture from prior work? Two measures, as used by recent novelty-assessment tools.

    python novelty_check.py                     # facet matrix: tables/novelty_facets.md and tables/cvpr_props.tex
    python novelty_check.py --semantic          # + SPECTER-2 similarity of our abstract to the prior abstracts
                                                #   (needs internet: Semantic Scholar or OpenAlex, HF model)

1. Facet matrix. Each method is described by architectural facets (1 = has it, 0.5 = partly, 0 = no), filled from
   the papers. Reported: the facets of ours that no prior method has (all prior work, and fusion methods only), and
   the weighted Jaccard overlap of our facet set with each method (1 - max overlap = distance to the nearest work).
2. Semantic similarity (optional). Cosine similarity of SPECTER-2 embeddings of our abstract and each prior
   paper's abstract (title + abstract, as SPECTER-2 is trained); the highest values are the closest papers in content.
   Papers of one field all score high, so the values are calibrated by the similarity of each prior paper to its
   nearest other prior paper: ours is as distinct as an average prior paper when the two maxima are equal.
Values are judgements from the papers' descriptions; "partly" marks a weaker or different form of the facet.
"""
import argparse
import json
import os
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
FACETS = [
    ("unfold", "unfolded data step + learned prior"),
    ("exact", "exact (non-learned) observation operators"),
    ("border", "border rule and adjoint of the simulation"),
    ("closed", "closed-form exact data step"),
    ("eigfilt", "learned filters in the operator's eigen domain"),
    ("estimate", "operators estimated from the observation pair"),
    ("family", "trained over a family of observation models"),
    ("cond", "operator / noise conditioning"),
    ("sensor", "one network for any MSI response (no retraining)"),
    ("scale", "one network for several scale factors"),
    ("misreg", "misregistration in the operator"),
    ("null", "prior confined to the null space (range-null)"),
    ("unc", "uncertainty output"),
    ("robust", "down-weights corrupted / changed inputs"),
    ("global", "global mixing in the operator's eigenbasis"),
    ("guarantee", "fixed-point / non-expansiveness guarantees"),
]
# fusion = HSI-MSI or pansharpening method; others are related ideas from other inverse problems
PRIOR = {
    "MHF-Net (TPAMI'22)": (True, dict(unfold=1)),
    "MoG-DCN (TIP'21)": (True, dict(unfold=1)),
    "DHIF-Net (TCI'22)": (True, dict(unfold=1)),
    "SMGU-Net (PR'25)": (True, dict(unfold=1)),
    "PSTUN (InfFus'25)": (True, dict(unfold=1, scale=1)),
    "CaFormer (PR'26)": (True, dict(unfold=1, exact=0.5)),
    "OTPNet (AAAI'25)": (True, dict(unfold=1, guarantee=0.5)),
    "IR&ArF (TIP'25)": (True, dict(unfold=1, scale=1, misreg=1)),
    "PGU-Net (arXiv'26)": (True, dict(unfold=1, estimate=1, sensor=0.5)),
    "HyCoNet (TGRS'20)": (True, dict(estimate=0.5, exact=0.5)),
    "SSA (ICML'26)": (True, dict(sensor=0.5, scale=1)),
    "MIMFormer (JSTARS'24)": (True, dict()),
    "FUSE (TIP'15)": (True, dict(exact=1, closed=0.5)),
    "HySure (TGRS'15)": (True, dict(exact=1, estimate=1)),
    "Robust fusion CD (TGRS'18)": (True, dict(exact=1, robust=1)),
    "USRNet (CVPR'20, SR)": (False, dict(unfold=1, exact=1, closed=0.5, family=1, cond=1, scale=1)),
    "SC-Net (arXiv'26)": (False, dict(exact=1, eigfilt=1, global_=0)),
    "DDNM (ICLR'23)": (False, dict(exact=1, null=1, unfold=0.5)),
    "Deep decomposition (ECCV'20)": (False, dict(exact=1, null=1)),
}
# ExactFuse as adopted (2026-10-31): no null-space gate and no eigen-operator prior (both tested, not used); the
# uncertainty head and the confidence-weighted data step are optional heads reported separately (0.5)
OURS = {f: 1.0 for f, _ in FACETS} | {"null": 0.0, "global": 0.0, "unc": 0.5, "robust": 0.5}
# the paper's property table: the closest methods and the facets a reader can check at a glance
TABLE_METHODS = ["DHIF-Net (TCI'22)", "SMGU-Net (PR'25)", "CaFormer (PR'26)", "IR&ArF (TIP'25)",
                 "USRNet (CVPR'20, SR)", "SSA (ICML'26)", "SC-Net (arXiv'26)"]
TABLE_FACETS = ["exact", "border", "closed", "estimate", "sensor", "misreg", "unc", "robust"]


def facet(m, f):
    return float(PRIOR[m][1].get(f, 0.0))


def jaccard(a, b):
    """Weighted Jaccard overlap of two facet vectors."""
    num = sum(min(a.get(f, 0), b.get(f, 0)) for f, _ in FACETS)
    den = sum(max(a.get(f, 0), b.get(f, 0)) for f, _ in FACETS)
    return num / den if den else 0.0


def facet_report():
    lines = ["# Architecture novelty: facet matrix", ""]
    names = list(PRIOR)
    lines.append("| Facet | " + " | ".join(names) + " | ExactFuse |")
    lines.append("|---|" + "---|" * (len(names) + 1))
    mark = {1.0: "✓", 0.5: "~", 0.0: ""}
    for f, lab in FACETS:
        lines.append(f"| {lab} | " + " | ".join(mark[facet(m, f)] for m in names) + f" | {mark[OURS[f]]} |")
    res = {}
    for scope, keep in (("all prior work", lambda m: True), ("fusion methods only", lambda m: PRIOR[m][0])):
        ms = [m for m in names if keep(m)]
        # only facets ExactFuse has can be new (an optional head counts as "partly")
        unique = [lab for f, lab in FACETS if OURS[f] == 1.0 and max(facet(m, f) for m in ms) == 0.0]
        partly = [lab for f, lab in FACETS if OURS[f] > 0 and lab not in unique and
                  max(facet(m, f) for m in ms) < OURS[f]]
        overlap = sorted(((jaccard(OURS, PRIOR[m][1]), m) for m in ms), reverse=True)
        res[scope] = dict(unique=unique, partly=partly, nearest=overlap[:3],
                          unique_share=len(unique) / len(FACETS),
                          new_or_partly_share=(len(unique) + 0.5 * len(partly)) / len(FACETS))
        lines += ["", f"## Compared with {scope} ({len(ms)} methods)", "",
                  f"- facets no method has: {len(unique)} of {len(FACETS)} ({100 * len(unique) / len(FACETS):.0f}%): "
                  + "; ".join(unique),
                  f"- facets only partly present elsewhere: {len(partly)}: " + "; ".join(partly),
                  "- closest methods by facet overlap (weighted Jaccard): "
                  + "; ".join(f"{m} {100 * j:.0f}%" for j, m in overlap[:3]),
                  f"- distance to the nearest method: {100 * (1 - overlap[0][0]):.0f}%"]
    os.makedirs(os.path.join(HERE, "tables"), exist_ok=True)
    with open(os.path.join(HERE, "tables", "novelty_facets.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[-14:]))
    return res


def props_table():
    """Compact property table for the paper (\\cmark / \\xmark / partly)."""
    lab = dict(FACETS)
    short = dict(exact="Exact op.", border="True border", closed="Closed form", estimate="Op. estimated",
                 sensor="Any sensor", misreg="Misreg.", null="Null-space prior", unc="Uncertainty",
                 robust="Corrupt. inputs")
    sym = {1.0: "\\checkmark", 0.5: "$\\circ$", 0.0: "--"}
    body = []
    for m in TABLE_METHODS:
        body.append(m.replace("&", "\\&") + " & " + " & ".join(sym[facet(m, f)] for f in TABLE_FACETS) + " \\\\")
    body.append("\\midrule")
    body.append("ExactFuse & " + " & ".join(sym[OURS[f]] for f in TABLE_FACETS) + " \\\\")
    lines = ["% generated by novelty_check.py - do not edit", "\\begin{table}[t]", "\\centering",
             "\\caption{Properties of the closest methods (\\checkmark: yes, $\\circ$: partly or in another form, "
             "--: no; for ExactFuse $\\circ$ marks an optional head reported separately). SR: single-image super-resolution.}", "\\label{tab:props}", "\\setlength{\\tabcolsep}{2pt}",
             "\\resizebox{\\columnwidth}{!}{%", "\\begin{tabular}{@{}l" + "c" * len(TABLE_FACETS) + "@{}}",
             "\\toprule", "Method & " + " & ".join(short[f] for f in TABLE_FACETS) + " \\\\", "\\midrule", *body,
             "\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    with open(os.path.join(HERE, "tables", "cvpr_props.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


S2 = "https://api.semanticscholar.org/graph/v1/paper/search?query={q}&limit=1&fields=title,abstract,year"
OA = "https://api.openalex.org/works?search={q}&per-page=1&select=title,publication_year,abstract_inverted_index"
TITLES = [
    "Multispectral and Hyperspectral Image Fusion by MS/HS Fusion Net",
    "Model-Guided Deep Hyperspectral Image Super-Resolution",
    "Deep Hyperspectral Image Fusion Network With Iterative Spatio-Spectral Regularization",
    "Spatial-spectral unfolding network with mutual guidance for multispectral and hyperspectral image fusion",
    "Cross-domain-aware deep unfolding transformer for hyperspectral image super-resolution",
    "IR&ArF: Toward Deep Interpretable Arbitrary Resolution Fusion of Unregistered Hyperspectral and Multispectral Images",
    "Hyperspectral Image Fusion with Spectral-Band and Fusion-Scale Agnosticism",
    "Deep Unfolding Network for Image Super-Resolution",
    "Interpretable Operator Learning for Inverse Problems via Adaptive Spectral Filtering",
    "Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model",
    "Deep Decomposition Learning for Inverse Imaging Problems",
    "MIMFormer: Multiscale Inception Mixer Transformer for Hyperspectral and Multispectral Image Fusion",
    "Hyperspectral Super-Resolution by Coupled Spectral Unmixing",
    "A Two-Stage Conditional Diffusion Model With Differential Attention for Hyperspectral and Multispectral Image Fusion",
    "Robust Fusion of Multiband Images With Different Spatial and Spectral Resolutions for Change Detection",
    "A Convex Formulation for Hyperspectral Image Superresolution via Subspace-Based Regularization",
]


def semantic(ours_text):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoModel.from_pretrained("allenai/specter2_base").eval()

    def emb(texts):
        with torch.no_grad():
            b = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
            return torch.nn.functional.normalize(model(**b).last_hidden_state[:, 0], dim=1)

    def get(url):
        req = urllib.request.Request(url, headers={"User-Agent": "novelty-check"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)

    def fetch(t):
        """-> (title, year, abstract) from Semantic Scholar (rate-limited without a key), else OpenAlex."""
        for wait in (0, 5):
            try:
                time.sleep(wait)
                d = get(S2.format(q=urllib.parse.quote(t)))["data"][0]
                if d.get("abstract"):
                    return d["title"], d.get("year"), d["abstract"]
                break
            except Exception:
                pass
        d = get(OA.format(q=urllib.parse.quote(t)))["results"][0]
        inv = d.get("abstract_inverted_index") or {}
        words = sorted((i, w) for w, pos in inv.items() for i in pos)
        return d["title"], d.get("publication_year"), " ".join(w for _, w in words)

    papers = []
    for t in TITLES:
        try:
            title, year, abstract = fetch(t)
            if abstract:
                papers.append((title, year, title + tok.sep_token + abstract))
            else:
                print("no abstract:", t[:60], flush=True)
        except Exception as err:
            print("no abstract:", t[:60], repr(err)[:80], flush=True)
        time.sleep(1.2)
    e = emb([p[2] for p in papers] + [ours_text])
    sims = (e[:-1] @ e[-1]).tolist()
    pp = e[:-1] @ e[:-1].T
    pp.fill_diagonal_(-1)
    nearest_prior = pp.max(1).values.tolist()
    ranked = sorted(zip(sims, [p[0] for p in papers], [p[1] for p in papers], nearest_prior), reverse=True)
    for s, t, y, n in ranked:
        print(f"{s:.3f}  {t} ({y})  [its nearest other prior paper: {n:.3f}]", flush=True)
    calib = dict(ours_max=round(max(sims), 4), prior_nearest_mean=round(sum(nearest_prior) / len(nearest_prior), 4),
                 prior_nearest_min=round(min(nearest_prior), 4), prior_nearest_max=round(max(nearest_prior), 4),
                 papers=len(papers))
    print("calibration:", calib, flush=True)
    return dict(ranked=[dict(similarity=round(s, 4), title=t, year=y, nearest_prior=round(n, 4))
                        for s, t, y, n in ranked], calibration=calib)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semantic", action="store_true")
    ap.add_argument("--abstract", default=os.path.join(HERE, "cvpr_main.tex"))
    ap.add_argument("--out", default=os.path.join(HERE, "tables"))
    a = ap.parse_args()
    res = facet_report()
    props_table()
    if a.semantic:
        import re
        tex = open(a.abstract, encoding="utf-8").read()
        title = re.search(r"\\title\{([^}]*)\}", tex).group(1)
        abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1)
        abstract = re.sub(r"\\qone\{[^}]*\}", "X", abstract)
        abstract = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", abstract).replace("~", " ")
        res["semantic"] = semantic(title + " [SEP] " + abstract)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "novelty.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
