"""
Least-squares (MAP) Hill-curve fitting for real GDSC2 IC50 data.

For each of the same first-5 (cosmic_id, drugset_id) tuples per drug
that `ep_real.py` and `preprocess_real.py` use, run a JAX-jitted MAP
fit and write a row to `scripts/results/least_squares_fits.csv`.

The saved `logic50` is in **ln(µM)**, matching `ep_real.py`'s
`log_ic50`, so the two outputs are directly comparable per dataset.

Usage:
    python3 scripts/least_squares.py            # both drugs
    python3 scripts/least_squares.py --drug_id 1003

Inputs:
    dataset/real/raw/GDSC2_public_raw_data_27Oct23.csv  (~2 GB, chunked)
    dataset/real/data_rnaseq_svd_df.pkl                  (cached 20-dim SVD)
    dataset/real/screened_compounds_rel_8.5.csv

Output:
    scripts/results/least_squares_fits.csv
"""

import argparse
import os
import warnings
from pathlib import Path

os.environ["USE_JAX"] = "1"

import numpy as np
import pandas as pd
import jax
from jax import numpy as jnp
from scipy import optimize
from sklearn import isotonic

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore")

DRUGS = [1003, 1073]
N_DATASETS = 5
NOISE_SIGMA = 9099.568421768981
GDSC_CHUNK_ROWS = 1_000_000

eps = np.finfo(np.float64).eps


# ---------------------------------------------------------------------------
# Hill fitter (JAX MAP via scipy.optimize.minimize)
# ---------------------------------------------------------------------------


@jax.jit
def hill(x, k=0, n=0, base=1):
    n = jnp.exp(n)
    max_val = jnp.log(jnp.finfo(jnp.dtype(x)).max) - 1
    nkx = n * (k - x)
    overflow = nkx > max_val
    underflow = nkx < -max_val
    pred = underflow | overflow
    x0 = jnp.where(pred, 0.0, x)
    k0 = jnp.where(pred, 0.0, k)
    n0 = jnp.where(pred, 0.0, n)
    nkx0 = jnp.where(pred, 0.0, n0 * (k0 - x0))
    return base * jnp.where(
        underflow, 1, jnp.where(overflow, 0, 1 / (1 + jnp.exp(nkx0)))
    )


@jax.jit
def hill_predict(x, k=0, n_log=0, base=1):
    """Prediction function — n_log is already linear here (no exp)."""
    n = n_log
    max_val = jnp.log(jnp.finfo(jnp.dtype(x)).max) - 1
    nkx = n * (k - x)
    overflow = nkx > max_val
    underflow = nkx < -max_val
    pred = underflow | overflow
    x0 = jnp.where(pred, 0.0, x)
    k0 = jnp.where(pred, 0.0, k)
    n0 = jnp.where(pred, 0.0, n)
    nkx0 = jnp.where(pred, 0.0, n0 * (k0 - x0))
    return base * jnp.where(
        underflow, 1, jnp.where(overflow, 0, 1 / (1 + jnp.exp(nkx0)))
    )


@jax.jit
def hill_like(p, x, y, like_prec):
    z = (hill(x, *p) - y) * like_prec
    return z.dot(z) * 0.5


@jax.jit
def norm_prior(x, prior_x, prior_prec):
    z = (x - prior_x) * prior_prec
    return z.dot(z) * 0.5


@jax.jit
def hill_post(p, x, y, like_prec, prior_p, prior_prec):
    return hill_like(p, x, y, like_prec) + norm_prior(p, prior_p, prior_prec)


hill_jac = jax.grad(hill_post)


def fit_isotonic(y, x):
    iso = isotonic.IsotonicRegression().fit(np.asarray(x), np.asarray(y))
    iso_fit = pd.Series(iso.y_thresholds_, iso.X_thresholds_)
    iso_grad = pd.Series(
        np.gradient(iso.y_thresholds_, iso.X_thresholds_), iso.X_thresholds_
    )
    return pd.DataFrame(
        {"intensity": iso_fit, "gradient": iso_grad}
    )


def initial_ic50_fit(iso_data):
    concs = iso_data.index.values
    max_conc = iso_data.loc[concs.max()]
    grad = iso_data.gradient.min()
    base = max_conc.intensity + max_conc.gradient
    ic50 = np.interp(base / 2, iso_data.intensity.values[::-1], concs[::-1])
    n = grad * 4 / (base + eps)
    return pd.Series([ic50 * n, n, base], ["c0", "c1", "base"])


def fit_ic50(x, y, like_std=NOISE_SIGMA, prior_std=10.0):
    """Fit a Hill curve to (x, y) where x = log(CONC_µM) (decreasing y in x).

    Internally we use the script's existing parameterisation
    (`x_fit = -x`, `nkx = n*(k - x_fit)`, increasing in x_fit, decreasing
    in x), then convert at output. The returned `result` has `x[0] = k`,
    where `k = -log_ic50` in our convention. The caller takes `-result.x[0]`
    to get `log_ic50` in `ln(µM)`.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    x_fit = -x

    # Initial guess via isotonic regression on the (x_fit, y) ordering.
    try:
        iso_data = fit_isotonic(np.asarray(y), np.asarray(x_fit))
        p_iso = initial_ic50_fit(iso_data)
        k0 = p_iso.c0 / p_iso.c1 if p_iso.c1 else float(jnp.max(x_fit))
        p0 = jnp.array([k0, jnp.log(np.maximum(p_iso.c1, 0.1)), p_iso.base])
        if not np.all(np.isfinite(np.asarray(p0))):
            raise ValueError("non-finite isotonic init")
    except Exception:
        p0 = jnp.array([float(jnp.mean(x_fit)), 0.5, float(jnp.max(y))])

    args = (x_fit, y, 1.0 / like_std, p0, 1.0 / prior_std)
    return optimize.minimize(hill_post, p0, jac=hill_jac, args=args)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_drug_subset(real_dir, drug_ids):
    raw_path = real_dir / "raw" / "GDSC2_public_raw_data_27Oct23.csv"
    print(f"Streaming GDSC2 raw CSV ({raw_path})...", flush=True)
    keep_cols = ["DRUG_ID", "COSMIC_ID", "DRUGSET_ID", "CONC", "INTENSITY"]
    pieces = []
    rows_total = 0
    for chunk in pd.read_csv(raw_path, usecols=keep_cols, chunksize=GDSC_CHUNK_ROWS):
        rows_total += len(chunk)
        chunk = chunk[chunk["DRUG_ID"].isin(drug_ids)]
        if len(chunk) == 0:
            continue
        chunk = chunk.dropna(subset=["CONC", "INTENSITY"])
        chunk = chunk[(chunk["CONC"] > 0) & np.isfinite(chunk["CONC"])]
        pieces.append(chunk)
    print(f"  scanned {rows_total} rows; kept {sum(len(p) for p in pieces)}", flush=True)
    return pd.concat(pieces, ignore_index=True)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(drug_ids):
    workspace = Path(__file__).resolve().parent.parent
    real_dir = workspace / "dataset" / "real"
    results_dir = workspace / "scripts" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    drug_info = pd.read_csv(real_dir / "screened_compounds_rel_8.5.csv")
    drug_names = dict(zip(drug_info["DRUG_ID"], drug_info["DRUG_NAME"]))

    npz = np.load(real_dir / "data_rnaseq_svd_df.npz", allow_pickle=True)
    svd_df = pd.DataFrame(
        npz["data_rnaseq_svd_df"], index=npz["cosmic_ids"].astype(int)
    )
    print(f"Loaded SVD latents: {svd_df.shape}", flush=True)

    gdsc = load_drug_subset(real_dir, drug_ids)

    rows = []
    for drug_id in drug_ids:
        sub = gdsc[gdsc["DRUG_ID"] == drug_id].copy()
        sub = sub[sub["COSMIC_ID"].isin(svd_df.index)]
        tuples = sorted(
            sub.groupby(["COSMIC_ID", "DRUGSET_ID"]).groups.keys(),
            key=lambda t: (int(t[0]), int(t[1])),
        )

        n_done = 0
        for cosmic_id, drugset_id in tuples:
            ds = sub[
                (sub["COSMIC_ID"] == cosmic_id) & (sub["DRUGSET_ID"] == drugset_id)
            ].sort_values(by="CONC")
            if len(ds) < 3:
                continue

            x = np.log(ds["CONC"].values).astype(float)  # ln(µM)
            y = ds["INTENSITY"].values.astype(float)

            res = fit_ic50(x, y, like_std=NOISE_SIGMA)

            # `res.x[0] = k = -log_ic50` (script's internal x_fit = -x form).
            log_ic50 = -float(res.x[0])
            n_log_fit = float(res.x[1])  # raw param; n = exp(n_log_fit)
            base_fit = float(res.x[2])

            # Hessian-derived sigmas, where available.
            try:
                hess_diag = np.asarray(res.hess_inv).diagonal()
            except Exception:
                hess_diag = np.full(3, np.nan)

            # Post-fit residual std on the data points.
            resid = np.asarray(
                hill_predict(jnp.asarray(-x), res.x[0], np.exp(res.x[1]), res.x[2])
            ) - y

            row = {
                "DRUG_ID": int(drug_id),
                "DRUG_NAME": str(drug_names.get(drug_id, drug_id)),
                "COSMIC_ID": int(cosmic_id),
                "DRUGSET_ID": int(drugset_id),
                "n_doses": int(len(x)),
                "logic50": log_ic50,                # ln(µM), decreasing-y convention
                "logic50_sigma": float(hess_diag[0]),
                "n_log": n_log_fit,
                "n_log_sigma": float(hess_diag[1]),
                "base": base_fit,
                "base_sigma": float(hess_diag[2]),
                "success": bool(res.success),
                "fun": float(res.fun),
                "residual_std": float(np.std(resid)),
                "latent_logRNAseq_svd5": svd_df.loc[cosmic_id].values[:5].tolist(),
                "latent_logRNAseq_svd20": svd_df.loc[cosmic_id].values.tolist(),
            }
            rows.append(row)
            print(
                f"  drug {drug_id} cosmic={cosmic_id} drugset={drugset_id}: "
                f"logic50={log_ic50:.3f} (ln µM) base={base_fit:.0f} fun={res.fun:.2f}",
                flush=True,
            )
            n_done += 1
            if n_done >= N_DATASETS:
                break

    out = pd.DataFrame(rows)
    out_path = results_dir / "least_squares_fits.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {len(out)} rows to {out_path}", flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drug_id", type=int, default=None)
    args = parser.parse_args()
    drugs = [args.drug_id] if args.drug_id is not None else DRUGS
    main(drugs)
