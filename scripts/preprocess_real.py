"""
Preprocessor: Real GDSC2 IC50 Datasets (ln µM x-axis)
======================================================

Builds per-dataset folders under `dataset/real/drug_<id>/` with
`x = np.log(CONC_µM)`, `y = INTENSITY` (raw fluorescence), `latent`
(N_LATENT-dim TruncatedSVD of log2(TPM+1) RNAseq), plus `info.json`.

Usage:
    python3 scripts/preprocess_real.py            # both drugs
    python3 scripts/preprocess_real.py --drug_id 1003

Inputs:
    dataset/real/raw/GDSC2_public_raw_data_27Oct23.csv  (~2 GB, chunked)
    dataset/real/rnaseq_all_20250117.csv
    dataset/real/driver_genes_20241212.csv
    dataset/real/model_list_20250630.csv
    dataset/real/screened_compounds_rel_8.5.csv

Output:
    dataset/real/drug_<id>/dataset_<i>/{x,y,latent}.npy + info.json
    dataset/real/drug_<id>/_sample/preprocessing_config.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DRUGS = [1003, 1073]
N_DATASETS = 5
N_LATENT = 5
NOISE_SIGMA = 9099.0
GDSC_CHUNK_ROWS = 1_000_000


def load_drug_subset(real_dir, drug_ids):
    """Stream the 2 GB GDSC2 raw CSV in chunks, keeping only the rows for
    the drugs of interest. Memory-bounded: at any point we hold one chunk
    plus the accumulating filtered rows.
    """
    raw_path = real_dir / "raw" / "GDSC2_public_raw_data_27Oct23.csv"
    print(f"Streaming {raw_path} in {GDSC_CHUNK_ROWS}-row chunks...", flush=True)
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
    print(f"  scanned {rows_total} rows; kept {sum(len(p) for p in pieces)} for drugs={drug_ids}", flush=True)
    return pd.concat(pieces, ignore_index=True)


def build_svd_latents(real_dir, n_latent):
    """Load the cached RNAseq SVD from the self-contained .npz.

    The .npz holds the 1050 × 20 SVD projection of log2(TPM+1) RNAseq
    over driver genes (computed once upstream and committed) along with
    the matching `cosmic_ids` index. We rehydrate that into a DataFrame
    and truncate to `n_latent` columns. This avoids re-running the SVD
    (which would need the 5 GB `rnaseq_all_*.csv`) on every preprocess.
    """
    print("Loading cached RNAseq SVD from .npz...", flush=True)
    npz = np.load(real_dir / "data_rnaseq_svd_df.npz", allow_pickle=True)
    svd_df = pd.DataFrame(
        npz["data_rnaseq_svd_df"], index=npz["cosmic_ids"].astype(int)
    )
    if svd_df.shape[1] >= n_latent:
        svd_df = svd_df.iloc[:, :n_latent]
    print(f"  SVD latents shape: {svd_df.shape}", flush=True)
    return svd_df


def main(drug_ids):
    workspace = Path(__file__).resolve().parent.parent
    real_dir = workspace / "dataset" / "real"

    drug_info = pd.read_csv(real_dir / "screened_compounds_rel_8.5.csv")
    drug_names = dict(zip(drug_info["DRUG_ID"], drug_info["DRUG_NAME"]))

    svd_df = build_svd_latents(real_dir, N_LATENT)
    gdsc = load_drug_subset(real_dir, drug_ids)

    for drug_id in drug_ids:
        drug_name = drug_names.get(drug_id, str(drug_id))
        out_dir = real_dir / f"drug_{drug_id}"
        sample_dir = out_dir / "_sample"
        out_dir.mkdir(parents=True, exist_ok=True)
        sample_dir.mkdir(parents=True, exist_ok=True)

        sub = gdsc[gdsc["DRUG_ID"] == drug_id].copy()
        sub = sub[sub["COSMIC_ID"].isin(svd_df.index)]
        tuples = sorted(
            sub.groupby(["COSMIC_ID", "DRUGSET_ID"]).groups.keys(),
            key=lambda t: (int(t[0]), int(t[1])),
        )

        n_written = 0
        i = 0
        for cosmic_id, drugset_id in tuples:
            ds = sub[
                (sub["COSMIC_ID"] == cosmic_id) & (sub["DRUGSET_ID"] == drugset_id)
            ].sort_values(by="CONC")
            if len(ds) < 3:
                continue

            x = np.log(ds["CONC"].values).astype(float)  # ln(µM)
            y = ds["INTENSITY"].values.astype(float)
            latent = svd_df.loc[cosmic_id].values.astype(float)

            cell_dir = out_dir / f"dataset_{i}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            np.save(cell_dir / "x.npy", x)
            np.save(cell_dir / "y.npy", y)
            np.save(cell_dir / "latent.npy", latent)
            with open(cell_dir / "info.json", "w") as f:
                json.dump(
                    {
                        "domain": "cancer",
                        "noise_sigma": NOISE_SIGMA,
                        "n_latent": N_LATENT,
                        "n_doses": len(x),
                        "cosmic_id": int(cosmic_id),
                        "drugset_id": int(drugset_id),
                        "drug_id": int(drug_id),
                        "drug_name": str(drug_name),
                        "simulated": False,
                        "x_unit": "ln(µM)",
                    },
                    f,
                    indent=4,
                )

            n_written += 1
            i += 1
            if n_written >= N_DATASETS:
                break

        with open(sample_dir / "preprocessing_config.json", "w") as f:
            json.dump(
                {
                    "drug_id": int(drug_id),
                    "drug_name": str(drug_name),
                    "n_latent": N_LATENT,
                    "noise_sigma": NOISE_SIGMA,
                    "n_datasets": n_written,
                    "x_unit": "ln(µM)",
                },
                f,
                indent=4,
            )
        print(f"drug {drug_id} ({drug_name}): wrote {n_written} datasets to {out_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drug_id", type=int, default=None)
    args = parser.parse_args()
    drugs = [args.drug_id] if args.drug_id is not None else DRUGS
    main(drugs)
