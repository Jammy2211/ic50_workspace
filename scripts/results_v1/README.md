# Archived results — pre-rerun snapshot (`_v1`)

**These are provenance, not evidence. Do not cite them, and do not compare new
runs against them as a baseline.**

Archived 2026-09-09 as part of graphical-ep campaign phase 4 (IC50), immediately
before rerunning everything on a current PyAutoFit.

## Why they cannot be interpreted

Every run archived here was produced by a PyAutoFit that predates the entire
phase-1/phase-2 EP fix wave of this campaign. Phase 1's benchmark verdict was
that **every failing cell was autofit's EP column** — so these numbers largely
measure bugs that have since been fixed, not the IC50 model.

Fixes landed after these runs and absent from them:

| PR | What it fixed |
|----|---------------|
| #1558 | Prior id 0 compared equal to the `FactorValue` sentinel, corrupting every multi-variable factor gradient |
| #1560 | Truncation limits dropped by `from_natural_parameters`/`__pow__`; `TransformedMessage.from_mode` skipped the Jacobian for scalar variables |
| #1562 | Laplace "covariance" was mean-field precision + a non-accumulating random diagonal secant (no factor curvature); failed line searches still overwrote the message |
| #1572 | Coupled-transform covariance in `from_mode` |
| #1573 | Deterministic variables kept cavity covariance on the fd-Hessian path |
| #1574 | A fully reverted projection was counted as updated |
| #1576 | Per-(factor, variable) stale tracking; `reverted_variables` added to `ep_history.csv` |
| #1578 | `errors_at_sigma(as_instance=True)` crashed on a prior-valued global model |
| #1580 | EP stale-mask fixed-point fix |

## A second reason for caution

The 3σ recovery tables here report 15/15, 3/3, 15/15 "within 3σ" — but the
recovered errors are wide enough that the test is close to vacuous. In
`ep_sim_summary.txt`, `dataset_0 log_ic50` has truth −0.4118 and σ = 1.573, i.e.
a 3σ acceptance window of roughly ±4.7 on a parameter whose plausible range is a
few units. A pass here does not mean the fit was good. These runs used
`nlive = 50`, `max_steps = 3`.

## What is archived where

| Archive path | Was | Contents |
|---|---|---|
| `output_v1/` | `output/` | 7 raw AutoFit run trees, 31 MB (gitignored, local only) |
| `scripts/results_v1/` | `scripts/results/` | committed summary `.json` / `.txt` / `.png` |
| `dataset/**/ep_results_v1/`, `dataset/**/graphical_results_v1/` | `…/ep_results/`, `…/graphical_results/` | per-dataset fit plots |
