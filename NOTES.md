# IC50 Workspace — Resume Notes

Last update: 2026-05-07. Repo: https://github.com/Jammy2211/ic50_workspace
on `main`.

This document is a snapshot you can read cold to pick the project back up
without having to scroll back through past conversations. Designed for
"I just sat down on Monday, what's the state and what do I do next?".

---

## Big picture

The workspace is the focused single-domain spin-out of `concr/`'s cancer
side, aimed at scientific publication. The Hill IC50 dose-response model
is fit four different ways across this repo, and they all produce
log_ic50 values in the **same units** (`ln(µM)`) under the **same
sign convention** (canonical decreasing dose-response: high concentration
→ low intensity).

All fitting goes through `scripts/util.py` which is the single source of
truth for the Hill curve, the Gaussian log-likelihood, and the plotting.

JAX support is on from the start (Hill curve and log-likelihood both
take an `xp=` kwarg defaulting to numpy; the AutoFit Analysis classes
JIT the hot path internally).

---

## What works end-to-end today

```
dataset/real/raw/GDSC2_public_raw_data_27Oct23.csv  (~2 GB symlink, gitignored)
dataset/real/rnaseq_all_20250117.csv                 (~5 GB, gitignored)
dataset/real/data_rnaseq_svd_df.npz                  (~170 KB, in git)
                                |
                                v
                    scripts/preprocess_real.py
                                |
                                v
dataset/real/drug_<id>/dataset_<i>/  (5 datasets per drug, both 1003 and 1073)
                                |
              ─────┬────────────┴────────────┬─────
                   v                          v
        scripts/ep_real.py            scripts/least_squares.py
                   |                          |
                   v                          v
   scripts/results/ep_real_<id>_summary  scripts/results/least_squares_fits.csv
                   |                          |
                   └────────────┬─────────────┘
                                v
                    scripts/compare_ls_ep.py
                                |
                                v
   scripts/results/compare_ls_ep_<id>.{txt,png}

dataset/ic50_sim/                <── scripts/simulator.py
                   |
                   v
      scripts/ep_sim.py           <── recovers truth (33/33 within 3σ)
                   |
                   v
   scripts/results/ep_sim_summary.{txt,json}
```

Plus two tutorial scripts that do not fit anything, just illustrate the
math, and have notebook versions:

- `scripts/likelihood_function.py` → `notebooks/likelihood_function.ipynb`
- `scripts/simulator.py` → `notebooks/simulator.ipynb`

(Both notebooks are auto-executed via `PyAutoBuild` and have plots
embedded.)

---

## Run cheat-sheet

From the workspace root (`z_projects/ic50_workspace/`):

```bash
# regenerate sim data (5 datasets at ln(µM) x ∈ [-3.45, 3.45])
python3 scripts/simulator.py

# preprocess 5 real datasets each for drugs {1003, 1073}
# (~30 s — chunk-streams the 2 GB GDSC2 CSV)
python3 scripts/preprocess_real.py
python3 scripts/preprocess_real.py --drug_id 1003   # single drug

# fast iteration mode (sub-minute, no recovery assertion)
PYAUTO_TEST_MODE=1 python3 scripts/ep_sim.py
PYAUTO_TEST_MODE=1 python3 scripts/ep_real.py

# proper EP runs (~1.5–2 min each, dynesty nlive=50, max_steps=3)
python3 scripts/ep_sim.py
python3 scripts/ep_real.py            # both drugs sequentially
python3 scripts/ep_real.py --drug_id 1003

# least-squares MAP Hill fit on the same 5 datasets per drug
python3 scripts/least_squares.py

# LS vs EP comparison (writes scripts/results/compare_ls_ep_<id>.{txt,png})
python3 scripts/compare_ls_ep.py

# rebuild + execute the notebooks
PYTHONPATH=../../PyAutoBuild/autobuild python3 ../../PyAutoBuild/autobuild/generate.py ic50
cd notebooks && for nb in simulator likelihood_function; do
  jupyter nbconvert --to notebook --execute --inplace "$nb.ipynb"
done && cd ..
```

---

## Sign / unit conventions (don't break these)

- Hill curve: `y = base / (1 + exp(n * (x - log_ic50)))` with `n = exp(n_log)`.
  Monotonically **decreasing** in `x`. y → base at low dose, y → 0 at high
  dose.
- `x` = `ln(CONC / µM)`. Real data: from `np.log(CONC_µM)` per-dataset.
  Sim: `np.linspace(-3.45, 3.45, 7)` (= 7-dose 3.16x dilution series
  centred on 1 µM).
- `log_ic50` is therefore in **`ln(µM)`** everywhere — sim, real-data EP,
  least-squares, comparison plots. Don't convert anywhere.
- `n_log` is the parameter; `n = exp(n_log)` is the actual Hill slope.
  Storing in log space keeps `n` positive without a hard constraint.

---

## Current results (as of 2026-05-07)

### Simulator recovery (`scripts/results/ep_sim_summary.txt`)

```
Pass/fail at 3σ:
  hill_coef:    15/15 within 3σ
  coef_mean:    3/3 within 3σ
  coef_matrix:  15/15 within 3σ
Wall time: 89 s
```

`coef_mean_true = [0, 0, 35000]` recovered as
`[-0.02±0.96, 0.12±0.48, 36664±5113]`. Mean σ-distance 0.20.

### Drug 1003 (Camptothecin) — the cleaner case

LS↔EP RMS Δ = 2.31 ln µM. All 5 LS values inside the EP 1σ error bars.
EP-LS deltas all positive (+1.0 to +4.2) — driven by the wide
log_ic50 prior `GaussianPrior(0, 5)` pulling the posterior toward 0
when LS values cluster around -2.5. Easy to fix later by recentring
the prior; intentionally left wide so the same priors work for both
drugs.

See `scripts/results/compare_ls_ep_1003.{txt,png}`.

### Drug 1073 (5-Fluorouracil) — degenerate case

LS↔EP RMS Δ = 15.0 ln µM. Most curves don't show full decline so IC50
isn't constrained — both methods push log_ic50 outside the data range.
This is a property of the data, not a bug in either fitter.

See `scripts/results/compare_ls_ep_1073.{txt,png}`.

---

## Caveats baked into the current code (read before changing anything)

- **`logic50_sigma` in `least_squares_fits.csv` is a variance** (the
  `hess_inv[0, 0]` element from `scipy.optimize.minimize`), not a
  standard error. Take `sqrt` if you want a frequentist sigma. This
  matches concr's convention; renaming would break upstream consumers.
- **All 5 drug-1003 datasets share `cosmic_id=683667`** — they're
  replicates across drugsets of one cell line, not independent cell
  lines. The "5-dataset" comparison is replicate-style. To compare
  across cell lines, bump `N_DATASETS` in `preprocess_real.py` or
  pick non-contiguous tuples.
- **Sim x-range and drug-1003 x-range don't overlap.** Sim is `[-3.45,
  +3.45]`; drug 1003 is `[-9.21, -2.30]` (sub-µM range). Fine for the
  sim's role as a sanity check but mention it if you ever want to fit
  drug-1003 data with priors centred on the sim's truth.
- **`force_x1_cpu=True`** is passed to `af.DynestyStatic` everywhere.
  Concr does the same — keeps dynesty single-CPU per-likelihood,
  avoiding pool-fork issues with JAX. Don't drop without testing.

---

## Where to resume — open follow-ups

In rough priority order. Tackle these one at a time.

### 1. Random-forest pipeline (`scripts/random_forest.py`)

Status: present but not runnable on this workspace yet.

What's missing:
- Its hard-coded input path `/Users/ctsl28/PD/...` → should read
  `scripts/results/least_squares_fits.csv` (already produced by
  `least_squares.py`, includes `latent_logRNAseq_svd5/svd20`).
- Output dir defaults — repoint to `scripts/results/random_forest/`.
- The script's feature is `latent_logRNAseq` (per-gene log2-TPM
  vector). Our LS CSV currently saves `latent_logRNAseq_svd5` and
  `latent_logRNAseq_svd20` (SVD-compressed). Decide which:
  - SVD: matches what EP uses; small, quick.
  - Per-gene: matches RF script's original assumption; needs adding
    a column to `least_squares.py` output.

End goal: an `RF predicted log_ic50` column in the comparison alongside
`LS logic50` and `EP log_ic50`, all three in `ln(µM)`.

### 2. Better real-data sample for EP

Currently 5 replicates of one cell line. To make EP's global
`coef_matrix @ latent + coef_mean` actually do something interesting,
we need ≥10 different cosmic_ids per drug. Bump `N_DATASETS` in
`preprocess_real.py` (and accept the longer EP runtime).

### 3. Tighter priors per drug

Drug 1003's IC50 is in `[-9, -2]` ln µM; drug 1073 in `[-3, +3]`.
Single shared prior `N(0, 5)` is wasteful for both. Add per-drug
`COEF_MEAN_PRIOR` table and pass into `ep_real.py` based on
`--drug_id`. Likely to remove the +2 ln µM systematic on drug 1003.

### 4. Remaining tutorial scripts

- `scripts/one_by_one.py` — single-dataset fit using AutoFit. Easy to
  adapt from `ep_real.py` (drop the global factor).
- `scripts/graphical.py` — full joint dynesty fit over the factor
  graph (no EP). Use as a check on EP convergence.
- `scripts/hierarchical.py` — explicit hyperprior on `coef_mean`.

### 5. Notebookify the fitting scripts

`ep_sim.py`, `ep_real.py`, `least_squares.py`, `compare_ls_ep.py` are
not in the PyAutoBuild notebook flow yet (long-running scripts; the
generated notebook would take minutes to execute). Decide whether to
split each into a "tutorial-explainer" notebook + a "production" `.py`,
or just leave as `.py`-only.

### 6. Migrate concr's production fitting code

Concr's `scripts/cancer/ep.py` etc. use the **inverted** sign convention
`n*(log_ic50 - x)`. They were left untouched in concr. Now that this
workspace has its own working EP fits with the canonical convention,
concr's cancer-side scripts can be migrated here (or just retired).
This is bigger than it sounds — concr has graphical/hierarchical
variants too.

### 7. Real-data scale check

CONC values for drug 1003 in GDSC2 are 10⁻⁴ to 10⁻¹ µM. The intensity
noise σ=9099 was calibrated for drug-1073 plates (where intensities are
~25–60 k). For drug 1003 the intensities are similar (~17–34 k) so
9099 is plausible, but worth measuring per-drug if the comparison
pivots on absolute likelihood values.

---

## Where things live

| Thing | Path |
|---|---|
| All Hill math + plotting | `scripts/util.py` |
| Sim data | `dataset/ic50_sim/dataset_<i>/` + `_sample/` ground truth |
| Real data | `dataset/real/drug_<id>/dataset_<i>/` (5 each for 1003, 1073) |
| Raw GDSC2 (~2 GB, gitignored, symlink to concr) | `dataset/real/raw/GDSC2_public_raw_data_27Oct23.csv` |
| Raw RNAseq (~5 GB, gitignored) | `dataset/real/rnaseq_all_20250117.csv` |
| Cached SVD (in git) | `dataset/real/data_rnaseq_svd_df.npz` |
| Fit outputs (text + JSON) | `scripts/results/` |
| AutoFit run dirs (~21 MB, gitignored) | `output/ep_<sim\|real>/ep/<hash>/...` |
| Per-iteration Visualizer plots | `output/.../dataset_<i>/optimization_<n>/image/hill_curve_{data,fit}.png` |
| Notebooks (auto-generated from scripts) | `notebooks/` |

---

## Things that would surprise you on a fresh clone

1. The 2 GB GDSC2 CSV is a symlink to `../../../concr/cancer_legacy/dataset/`.
   On a different machine without concr, the symlink will be broken. Either
   download GDSC2 directly into `dataset/real/raw/` or repoint the symlink.
2. The 5 GB `rnaseq_all_20250117.csv` is gitignored. If you need to
   recompute the SVD, fetch it from
   `https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources/Home.html`.
3. The cached SVD `.npz` (committed) is what both `preprocess_real.py`
   and `least_squares.py` actually consume — they don't need the 5 GB
   raw RNAseq unless you regenerate the SVD.
4. `output/` is gitignored. The fresh clone won't have AutoFit output
   until you re-run an EP fit. The summary files in `scripts/results/`
   (committed) are the canonical record of past runs.

---

## Recent commit history (most recent first)

```
b0e364e  Final sanity-check pass: drop fragile .pkl symlink, fix stale docstring
bf04ba6  Switch workspace to ln(µM) x-axis; LS↔EP comparison framework
7544bb3  Add ep_sim.py + ep_real.py — working AutoFit EP fits
75d0b35  Rebuild simulator and likelihood_function notebooks
da8dd3d  Add scripts/likelihood_function.py tutorial
65ee150  Rebuild simulator notebook with fresh execution
4cde480  Render __Contents__ as a Markdown bullet list
0f2316f  Bootstrap ic50_workspace with simulator and shared util
```

---

## Quick gut-check on resume

```bash
cd z_projects/ic50_workspace
git pull
PYAUTO_TEST_MODE=1 python3 scripts/ep_sim.py
```

If that runs in <1 min and writes `scripts/results/ep_sim_summary.{txt,json}`,
the JAX/AutoFit/dynesty toolchain is healthy. If it crashes, the most
likely culprit is the GDSC2 raw CSV symlink (preprocess), or the .npz file
(SVD) — neither of which `ep_sim.py` actually touches, so a crash here is
almost certainly an env issue.
