# PUFormer paper

Two versions are built from the same results and generated tables:

* **Journal version** (IEEEtran): [main.pdf](main.pdf) and [supplementary.pdf](supplementary.pdf). Full problem
  formulation and analysis (identifiability of the operators, closed-form data step and fusion null space, cost of
  an inexact operator), the module-by-module ablation, the equal-budget benchmark, other datasets and scale
  factors, the comparison with reported results, and the operator-family experiments.
* **Conference version** (8 pages, anonymous, two columns): [cvpr_main.pdf](cvpr_main.pdf) and
  [cvpr_supp.pdf](cvpr_supp.pdf). *Unfolded Hyperspectral and Multispectral Image Fusion with Exact Observation
  Operators*: the same theory in short form, Tables 1–5 (benchmark, other datasets and scales, where the margin
  comes from, ablation, one network for a family of observation models) and the limitations. The supplement holds
  the proofs, data and baseline details, the full metric tables, the architecture property table, the comparison
  with the 17 reported methods, NEON and future directions. Its LLM-usage statement is a placeholder that the
  authors must complete.

Both cover:

* an equal-budget benchmark: PUFormer and the published networks (SSR-NET, PSRT, DCT, MIMFormer, DHIF-Net,
  MIMO-SST, OTPNet, SMGU-Net, CLSNet, U2Net; CaFormer pending) retrained from their authors' code under one
  pipeline, on Chikusei and Pavia Centre ×4, CAVE and Harvard ×4, and ×8 / ×16
* the split of every method's error into the observed subspace and the fusion null space
* the margin decomposition (authors' recipes, PUFormer without data steps) and the ablation
* training over a family of blurs, spectral responses and noise levels, with the operators estimated from the pair,
  and the failure modes outside the family (rotated/defocus blur, misregistration)

## Files

| File | Contents |
|---|---|
| `main.tex`, `supplementary.tex`, `preamble.tex` | Journal manuscript, its supplement (numbered S-I, S1, ...) and the shared preamble |
| `cvpr_main.tex`, `cvpr_supp.tex`, `cvpr_preamble.tex`, `cvpr.sty`, `ieeenat_fullname.bst` | Conference version and author kit |
| `cvpr_refs.py` | Writes the labels of `cvpr_supp` into a file that `cvpr_main` reads (`\mref`), since Tectonic has no `xr` |
| `fig_*.tex` | TikZ figures: overview, architecture, prior, block, attention, data step |
| `references.bib` | References; every entry with a DOI was taken from its Crossref record |
| `make_q1_tables.py` | Benchmark, ablation, operator-family, margin and null-space tables, and the values quoted in the text as `\qone{key}` (`tables/q1_numbers.tex`) |
| `make_tables.py`, `make_numbers.py` | Comparison with reported results (`\pav{...}`) and the Chikusei Q2n numbers (`numbers.tex`) |
| `make_figures.py`, `make_q1_figures.py` | Figures in `figures/` |
| `novelty_check.py` | Architecture facet matrix against 19 prior methods (`tables/novelty_facets.md`, `tables/cvpr_props.tex`); with `--semantic`, SPECTER-2 similarity of the abstracts, calibrated by the similarity between prior papers |
| `integrity_check.py`, `build_integrity_kernel.py` | Similarity check against the cited works and AI-text screening of the compiled PDFs |
| `build_paper_kernel.py` | Writes the Kaggle notebook that runs every generator, compiles the four documents and renders page previews |

All numbers come from the committed result files: `results/q1/<job>/` (one folder per job of
`methods/puformer/kaggle/jobs.py`: `results.json`, `history.json`, `gaps.json`, `cond_eval.json`, `train.log`),
`results/q1/msi-value-*/`, `results/q1/novelty/`, and the long-trained models in `results/puformer_chikusei_x4_v3/`
and `results/puformer_pavia_x4/`. Missing results print as `--` in the tables and `??` in the text, and table rows
of runs that do not exist yet are left out, so the paper builds at any stage.

## Build (on Kaggle)

```bash
cd paper/puformer
python build_paper_kernel.py              # writes kaggle_paper/ (sources and result files embedded)
kaggle kernels push -p kaggle_paper
kaggle kernels output amarnath10chinu/puformer-paper-build -p out   # PDFs, figures/, tables/, pages/
```

The notebook runs the generators, downloads Tectonic, compiles `main.tex`, `supplementary.tex`, `cvpr_main.tex` and
`cvpr_supp.tex`, and renders every page as a PNG with its text (`pages/`) for checking unresolved references and
the page limit. The journal preamble loads TeX Gyre Termes because XeTeX has no Type-1 Times for IEEEtran.
