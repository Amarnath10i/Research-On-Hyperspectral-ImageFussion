# Related Work — Deep Learning-Based Polyp Segmentation on CVC-ClinicDB and Standard Colonoscopy Benchmarks

Curated for: related-work / literature-review section. Selection criteria applied — **subscription (non-open-access) journals only**, **impact factor ≈ 5–6**, **mostly 2023–2025**, all evaluated on the same benchmark combination visible in your results table: **CVC-ClinicDB, Kvasir-SEG, CVC-ColonDB, ETIS(-LaribPolypDB), CVC-300**.

All papers below are published in *Biomedical Signal Processing and Control* (Elsevier, ISSN 1746-8094), whose 2026-released impact factor is **5.7 (JCR Q2, Neuroscience)** — squarely in your requested 5–6 band, and it is a subscription journal (articles are paywalled unless the author pays the optional Elsevier OA fee), which matches your "no open access" requirement. This also gives you a *consistent, defensible venue* to cite repeatedly in a related-work section, which reviewers tend to like.

---

## 1. Ready-to-paste "Related Work" paragraph

> Polyp segmentation from colonoscopy imagery has been extensively benchmarked on five public datasets — CVC-ClinicDB, Kvasir-SEG, CVC-ColonDB, ETIS-LaribPolypDB, and CVC-300 — and recent architectures continue to target the joint challenges of scale variation, blurred polyp boundaries, and cross-dataset generalization. Lin et al. [1] proposed CSwinDoubleU-Net, a dual U-shaped encoder–decoder that fuses convolutional and Swin-Transformer branches with a coordinate-attention skip connection, reporting a mean Dice of 0.935 and mIoU of 0.885 on CVC-ClinicDB. Huang et al. [2] introduced RCNU-Net, which replaces the U-Net backbone with a reparameterized convolution module paired with a convolutional block attention module (CBAM) and a custom CDLoss to balance cross-entropy and Dice objectives. Wu et al. [3] addressed boundary ambiguity directly with BMANet, combining a cascaded partial decoder with a boundary-aware module and a boundary-guided multi-level attention block to sharpen polyp contours, and released their implementation publicly. Nguyen and Nguyen [5] proposed PolyPooling, a PoolFormer-based encoder paired with a dual-branch (boundary/area) decoder, also with public code, reporting roughly 12% mDice/mIoU improvement over prior methods averaged across the five-dataset benchmark. Selvaraj et al. [4] took a clinically oriented approach, pairing a CRPU-Net segmentation stage with downstream ViT-based classification on a multi-house colonoscopy dataset, reporting 97.39% Dice and 95.40% IoU. Collectively, these works illustrate the field's shift from purely CNN-based encoders toward attention- and transformer-augmented hybrids, and boundary-aware refinement modules, as the dominant strategies for closing the performance gap on the harder, smaller-scale datasets (CVC-ColonDB, ETIS) relative to CVC-ClinicDB and Kvasir-SEG.

---

## 2. Core citation table (BSPC, IF 5.7, non-OA, 2024–2025)

| # | Method | Authors | Venue / Year | Datasets used | Reported CVC-ClinicDB result | Code | DOI |
|---|--------|---------|---------------|----------------|-------------------------------|------|-----|
| 1 | **CSwinDoubleU-Net** | Lin, Han, Chen, Zhang, Liu | *Biomed. Signal Process. Control* 89:105749 (2024) | CVC-ClinicDB, Kvasir, CVC-ColonDB, CVC-T, ETIS-Larib (**all 5 — direct match**) | Dice 0.935, mIoU 0.885 | Not published | 10.1016/j.bspc.2023.105749 |
| 2 | **RCNU-Net** | Huang, Huang, Xu, Min, Hu, Zhang | *Biomed. Signal Process. Control* 93:106138 (2024) | Kvasir-SEG, CVC-ClinicDB | Reparameterized backbone + CBAM; CDLoss | Not published | 10.1016/j.bspc.2024.106138 |
| 3 | **BMANet** | Wu, Chen, Xiong, Wu, Li, Zhou | *Biomed. Signal Process. Control* 105:107524 (2025) | 5-dataset polyp benchmark | Boundary-guided multi-level attention, SOTA-competitive | **github.com/WZH0120/BMANet** | 10.1016/j.bspc.2025.107524 |
| 4 | **CRPU-Net / AI-CRC screening** | Selvaraj, Umapathy, Rajesh | *Biomed. Signal Process. Control* 99:106928 (2025) | Multi-house clinical + public data | Accuracy 96.56%, IoU 95.40%, Dice 97.39% | Not published | 10.1016/j.bspc.2024.106928 |
| 5 | **PolyPooling** | Nguyen, D.C., Nguyen, H.L. | *Biomed. Signal Process. Control* 92:105979 (2024) | 5-dataset polyp benchmark | ~+12% mDice/mIoU avg. vs. prior SOTA | **github.com/long-nguyen12/PolyPooling** | 10.1016/j.bspc.2024.105979 |

**Note on #1 (CSwinDoubleU-Net):** this is your strongest direct comparator — it reports mDice/mIoU on the *exact same five datasets* as your results table, so it is the cleanest one to place in a head-to-head SOTA comparison table.

---

## 3. Papers with GitHub code available (your specific ask)

Of the papers above, **only #3 (BMANet) and #5 (PolyPooling)** have confirmed public repositories — both still satisfy "non-open-access journal, IF ≈5-6, recent":

- **BMANet** — [github.com/WZH0120/BMANet](https://github.com/WZH0120/BMANet) — PyTorch implementation, boundary-aware module (BAM) + boundary-guided multi-level attention (BMA). BSPC, vol. 105, 2025.
- **PolyPooling** — [github.com/long-nguyen12/PolyPooling](https://github.com/long-nguyen12/PolyPooling) — PoolFormer-based encoder, dual-branch boundary/area decoder. BSPC, vol. 92, 2024.

Both are good candidates if you want to actually **run/fine-tune a competitor model** for your comparison table (not just cite its reported numbers) — CSwinDoubleU-Net, RCNU-Net, and the Selvaraj CRPU-Net paper do not appear to have published code, so for those three you can only cite their reported metrics, not reproduce them locally.

---

## 4. Your results vs. CSwinDoubleU-Net (same 5-dataset protocol)

| Dataset | Your model (mDice / mIoU) | CSwinDoubleU-Net [1] (mDice / mIoU) |
|---|---|---|
| CVC-ClinicDB | **0.9370 / 0.8816** | 0.935 / 0.885 |
| Kvasir-SEG | **0.9298 / 0.8696** | 0.907 / 0.850 |
| CVC-ColonDB | **0.7821 / 0.6679** | 0.716 / 0.644 |
| ETIS | **0.7856 / 0.6574** | 0.667 / 0.595 |
| CVC-300 (EndoScene) | **0.8854 / 0.7947** | 0.887 / 0.821 |

Your model is ahead on CVC-ClinicDB, Kvasir-SEG, CVC-ColonDB, and ETIS, and essentially tied on CVC-300/EndoScene — worth building directly into a "Comparison with State-of-the-Art" table in your Results section, citing [1] as the comparator.

---

## 5. Reference list (IEEE style, ready to paste)

```
[1] Y. Lin, X. Han, K. Chen, W. Zhang, and Q. Liu, "CSwinDoubleU-Net: A double
    U-shaped network combined with convolution and Swin Transformer for
    colorectal polyp segmentation," Biomedical Signal Processing and Control,
    vol. 89, art. 105749, 2024, doi: 10.1016/j.bspc.2023.105749.

[2] B. Huang, T. Huang, J. Xu, J. Min, C. Hu, and Z. Zhang, "RCNU-Net:
    Reparameterized convolutional network with convolutional block attention
    module for improved polyp image segmentation," Biomedical Signal
    Processing and Control, vol. 93, art. 106138, 2024,
    doi: 10.1016/j.bspc.2024.106138.

[3] Z. Wu, H. Chen, X. Xiong, S. Wu, H. Li, and X. Zhou, "BMANet:
    Boundary-guided multi-level attention network for polyp segmentation in
    colonoscopy images," Biomedical Signal Processing and Control, vol. 105,
    art. 107524, 2025, doi: 10.1016/j.bspc.2025.107524.
    Code: https://github.com/WZH0120/BMANet

[4] J. Selvaraj, S. Umapathy, and N. A. Rajesh, "Artificial intelligence based
    real time colorectal cancer screening study: Polyp segmentation and
    classification using multi-house database," Biomedical Signal Processing
    and Control, vol. 99, art. 106928, 2025, doi: 10.1016/j.bspc.2024.106928.

[5] D. C. Nguyen and H. L. Nguyen, "PolyPooling: An accurate polyp
    segmentation from colonoscopy images," Biomedical Signal Processing and
    Control, vol. 92, art. 105979, 2024, doi: 10.1016/j.bspc.2024.105979.
    Code: https://github.com/long-nguyen12/PolyPooling
```

---

## 6. Optional context citation (slightly outside the IF 5-6 band)

If you want one older, extremely widely-cited baseline that also evaluates on all five datasets, **Focus U-Net** (Yeung, Sala, Schönlieb, Rundo) was published in *Computers in Biology and Medicine*, vol. 137, art. 104815, 2021 (doi: 10.1016/j.compbiomed.2021.104815). That journal's impact factor runs closer to ~7 and the paper is from 2021, so it sits just outside your "5–6, recent" criteria — but it's frequently cited as a baseline in this exact benchmark setting, so it's worth keeping in your back pocket if a reviewer asks for a comparison against an older, established method.

---

## 7. What was deliberately excluded

A large share of recent polyp-segmentation work (PraNet, Polyp-PVT, DUCK-Net, ColonFormer, MEGANet, MugenNet, and most arXiv preprints) is either fully open access (including MDPI journals such as *Sensors*, which publish everything OA regardless of subject) or not yet peer-reviewed in a journal, so these were excluded per your "no open access" constraint — even though several report strong numbers and do have GitHub code. Also excluded: papers in journals with IF clearly outside the 5–6 band (e.g., *Expert Systems with Applications* ~8.5, *Pattern Recognition* ~8, *Knowledge-Based Systems* ~8.8, *Wiley IMA/International Journal of Imaging Systems and Technology* ~3, where ESFCU-Net's code at github.com/aaafoxy/ESFCU-Net lives).

---

## 8. Which of the two GitHub repos is actually easier to build on

Checked both repos directly — they're not equally reproducible:

| | **BMANet** | **PolyPooling** |
|---|---|---|
| Repo contents | `BMANet.py` (model class only) + README + LICENSE + one diagram | `configs/`, `models/`, `images/`, full README |
| Training/eval code | **Not included** — only the architecture definition | **Included** — "official PyTorch implementation of training & evaluation code" |
| Dataset setup | Not documented | Documented, with direct Google Drive links for train/test splits |
| Environment | Not specified | CUDA 11.1, PyTorch 1.7.1 specified |
| Commits / stars | 11 commits, 9 stars | 14 commits, 2 stars |
| Effort to get a running baseline | High — you'd have to write your own dataloader, training loop, loss, and eval scripts around their model class (or splice it into an existing PraNet-style harness) | Low — clone, download data to the two specified folders, run |

**Recommendation: PolyPooling is the easier one to fork and improve.** It ships a complete, config-driven training/eval pipeline, not just an architecture file, so you can get a reproduced baseline running in an afternoon rather than a week. BMANet's architecture (CPD + BAM + BMA) is arguably more novel, but without released training code it isn't meaningfully more "reproducible" than CSwinDoubleU-Net or RCNU-Net — you'd be reimplementing a training harness either way, so its GitHub presence saves you comparatively little.

**Where PolyPooling is weakest (= your improvement angle):** the paper itself states it is only "competitive" (not leading) on Kvasir-SEG and CVC-ClinicDB, while its big gains are concentrated on ETIS (+12.1% mDice) and CVC-ColonDB/CVC-T (+1.8–4.4%). That's a specific, defensible gap to target rather than a vague "improve everything" claim:

- **Concrete extension ideas:**
  1. Its refinement module deliberately replaces attention with plain pooling for efficiency — reintroducing a lightweight attention/boundary term (borrowing the idea, not the code, from BMANet's BAM) as a hybrid pooling+boundary-attention refinement could close the Kvasir/ClinicDB gap while keeping most of the efficiency.
  2. Swap the PoolFormer encoder for a stronger recent backbone (e.g., PVTv2-b2, which most 2024–2025 polyp papers use) — encoder swaps are usually the single highest-leverage, lowest-effort change in this architecture family.
  3. Add boundary-aware auxiliary supervision (a boundary/edge loss term) specifically to help the small, hard-to-see polyps in ETIS/ColonDB, since that's the class of error pooling-only refinement is most likely to miss.
  4. If you want a paper-worthy comparison, run PolyPooling as your reproduced baseline and cite CSwinDoubleU-Net, RCNU-Net, BMANet, and Selvaraj et al. as reported-only numbers from their papers.
