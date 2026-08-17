# proposal6 — ConsistentFlow (legacy, superseded)

Prior Q1 architecture, selfchecked.  Consistency-regularised normalising flow
for HSI–MSI fusion.

**Status:** superseded by the restructured Q1 program.  The null-space
consistency mechanism it explored now lives in `proposal5/spectralflow`
(`RangeNullProjector`) and the P1 paper's ambiguity scaffold in
`proposal1/ambiguity`; the program's present-work papers are P2 → P1 → P3 → P4
(see each proposal's README and the root README).

```powershell
$env:PYTHONPATH="common;proposal6"; python -c "import consistentflow; consistentflow._selfcheck.run_all()"
```