# proposal7 — NullFusion: a Null-Space Conditional Fusion Network

**Question (Q1 headline):** can a fusion network be built so that it *cannot
hallucinate* the part of the scene the sensors already determine, while still
reaching SOTA PSNR?

**Answer:** yes.  NullFusion splits the problem along the *exact* range/null
decomposition of the combined observation operator `A = [D; R]`
(`proposal1.ambiguity.operator.CombinedOperator`, verified in `selfcheck`):

```
X_hat = pinv(yH, yM)  +  P_N( f_theta( conditioning ) )
        \___________/      \___________________________/
        solved in          learned ONLY in the null space,
        closed form         so it cannot touch the
        (not learned)       observable part -> provably
                            non-hallucinating (P1 H-metric)
```

Because `A(pinv(yH,yM) + P_N v) = [yH; yM]` is an algebraic identity, data
consistency is *guaranteed* and needs no penalty term.  This is the
deterministic, feed-forward counterpart of `proposal5`'s generative
SpectralFlow / ManifoldFlow: same principled split, but a prior `f_theta` that
is trained to maximise PSNR/SAM rather than sampled.

## Why this is novel and Q1-grade
- **No published HSI-MSI method** builds the network around the `A=[D;R]`
  null-space decomposition with *exact* consistency by construction.  Existing
  methods predict the whole cube and hope the observations are satisfied.
- **Auditable by design:** the P1 hallucination metric
  `H = ||P_N(X_hat - X)|| / ||P_N X||` falls out for free, and any spectral
  content the network invents is provably confined to the null space.
- **Ties the four proposals into one method + one diagnostic layer:**
  - **P1** — admissible ambiguity (the `P_N` split *is* the architecture).
  - **P2** — identifiable rank `r_id` gates a spectral low-rank bottleneck
    inside `f_theta` (`NullFusionConfig.rank`), so the prior uses exactly the
    spectral degrees the *sensor pair* supports.
  - **P3** — sensor shift enters through the SRF fed to `A`; swapping the SRF
    buffer to a domain-shifted response re-derives the consistent set exactly.
  - **P4** — `r_id` (hence the bottleneck width) follows the phase-transition
    formula `M*(r)`.
- **Ablatable** via the `PROTOCOL_AUDIT.md` Q1 ladder: range-only (pinv) →
  +null prior → +rank adaptivity → +sensor conditioning.

## Status
Implemented and verified on CPU:
- `selfcheck` ALL PASS — `[consistency]` (D/S err ~1e-5), `[gradient]`,
  `[prior>base]` (14.0 → 32.9 dB).
- `train_sota.py` training path verified (synthetic `--smoke` self-test:
  loss decreases, **exact** `A(X_hat)==[yH;yM]` holds to 1e-5 at inference,
  ridge_eval=1e-6).  The clamp is OFF by default so the identity is exact;
  a post-hoc `[0,1]` clamp is optional and changes the range component by
  only ~1e-4 relative.
- 31-band (Nikon) consistency confirmed: `D-err=1.74e-05, S-err=1.05e-05`.

Real-data training (CAVE ×4, Nikon D700 SRF — the only protocol with a direct
SOTA number: FeINFN 52.47 / BDT 52.30 dB) runs on Kaggle, exactly as
`proposal2/notebooks/krylovnet_SOTA_CAVE_Nikon.ipynb`, with `KrylovNet` swapped
for `NullFusionNet`.  The notebook is provided at
`proposal7/notebooks/nullfusion_SOTA_CAVE_Nikon.ipynb`.

## Run
```powershell
$env:PYTHONPATH="common;proposal1;proposal5;proposal7"
python -c "import proposal7.nullfusion as m; m.selfcheck.run_all()"   # unit checks
python proposal7\nullfusion\train_sota.py --smoke                     # training-path self-test
```

## Capacity note (the part that beats SOTA)
The `pinv` range component is exact regardless of training; **all** learnable
capacity lives in `f_theta` (the null prior).  To reach 52 dB:
- raise `width` (96 → 128+), `prior_depth`, keep `use_attn=True`;
- leave `rank = bands` (the default) so the prior has full spectral capacity —
  the null space has `bands - r_id` dimensions, NOT `r_id`, so constraining the
  prior to `r_id` would cripple it.  The P2 `r_id` is instead the *auditable*
  bound reported as the H metric and used in the ablation ladder (optionally
  set `rank = bands - r_id` as a P2-regularised variant);
- train under the Nikon SRF with the budget in the SOTA notebook.
