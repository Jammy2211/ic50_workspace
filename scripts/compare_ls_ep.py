"""
Compare per-dataset log_ic50 from EP vs least-squares for the same 5
real GDSC2 datasets per drug.

Reads:
    scripts/results/least_squares_fits.csv             (LS, all drugs)
    scripts/results/ep_real_<drug_id>_summary.json     (EP, per drug)
    dataset/real/drug_<drug_id>/dataset_<i>/info.json  (cosmic_id, drugset_id)

Writes:
    scripts/results/compare_ls_ep_<drug_id>.txt        (per-drug table)
    scripts/results/compare_ls_ep_<drug_id>.png        (LS vs EP scatter)

Both LS and EP fits use `x = ln(CONC_µM)` and the canonical decreasing-y
sign convention, so logic50 (LS) and log_ic50 (EP) are in the same units
and directly comparable per (cosmic_id, drugset_id).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DRUGS = [1003, 1073]


def main():
    workspace = Path(__file__).resolve().parent.parent
    real_dir = workspace / "dataset" / "real"
    results_dir = workspace / "scripts" / "results"

    ls = pd.read_csv(results_dir / "least_squares_fits.csv")

    summary_lines = []

    for drug_id in DRUGS:
        ep_summary_path = results_dir / f"ep_real_{drug_id}_summary.json"
        if not ep_summary_path.exists():
            print(f"[skip] {ep_summary_path} not found yet")
            continue

        ep = json.load(open(ep_summary_path))
        n_datasets = ep["n_datasets"]
        ep_means = np.asarray(ep["hill_means"])
        ep_sigmas = np.asarray(ep["hill_sigmas"])

        # Get per-dataset (cosmic_id, drugset_id) ordering from info.json files
        ds_dir = real_dir / f"drug_{drug_id}"
        infos = []
        for i in range(n_datasets):
            with open(ds_dir / f"dataset_{i}" / "info.json") as f:
                infos.append(json.load(f))

        # Build comparison rows
        rows = []
        for i, info in enumerate(infos):
            cosmic = int(info["cosmic_id"])
            drugset = int(info["drugset_id"])
            ls_row = ls[
                (ls["DRUG_ID"] == drug_id)
                & (ls["COSMIC_ID"] == cosmic)
                & (ls["DRUGSET_ID"] == drugset)
            ]
            if len(ls_row) == 0:
                rows.append({
                    "dataset": f"dataset_{i}",
                    "cosmic_id": cosmic,
                    "drugset_id": drugset,
                    "ls_logic50": None,
                    "ls_fun": None,
                    "ep_log_ic50": float(ep_means[i, 0]),
                    "ep_log_ic50_sigma": float(ep_sigmas[i, 0]),
                    "delta": None,
                })
                continue
            ls_logic50 = float(ls_row.iloc[0]["logic50"])
            ls_fun = float(ls_row.iloc[0]["fun"])
            ep_log_ic50 = float(ep_means[i, 0])
            ep_log_ic50_sigma = float(ep_sigmas[i, 0])
            rows.append({
                "dataset": f"dataset_{i}",
                "cosmic_id": cosmic,
                "drugset_id": drugset,
                "ls_logic50": ls_logic50,
                "ls_fun": ls_fun,
                "ep_log_ic50": ep_log_ic50,
                "ep_log_ic50_sigma": ep_log_ic50_sigma,
                "delta": ep_log_ic50 - ls_logic50,
            })

        df = pd.DataFrame(rows)
        drug_name_lookup = ls[ls["DRUG_ID"] == drug_id]["DRUG_NAME"].iloc[0] if (ls["DRUG_ID"] == drug_id).any() else str(drug_id)

        # Text summary
        lines = []
        lines.append("=" * 100)
        lines.append(f"LS vs EP log_ic50 comparison — drug {drug_id} ({drug_name_lookup})")
        lines.append("=" * 100)
        lines.append(
            f"{'dataset':<12} {'cosmic_id':>10} {'drugset':>8} "
            f"{'LS logic50':>12} {'LS fun':>8} {'EP log_ic50':>13} "
            f"{'EP σ':>8} {'Δ (EP-LS)':>11}"
        )
        lines.append("-" * 100)
        for r in rows:
            ls_str = f"{r['ls_logic50']:>12.3f}" if r["ls_logic50"] is not None else f"{'n/a':>12}"
            fun_str = f"{r['ls_fun']:>8.3f}" if r["ls_fun"] is not None else f"{'n/a':>8}"
            d_str = f"{r['delta']:>+11.3f}" if r["delta"] is not None else f"{'n/a':>11}"
            lines.append(
                f"{r['dataset']:<12} {r['cosmic_id']:>10d} {r['drugset_id']:>8d} "
                f"{ls_str} {fun_str} {r['ep_log_ic50']:>13.3f} "
                f"{r['ep_log_ic50_sigma']:>8.3f} {d_str}"
            )
        lines.append("")

        # Aggregate metrics
        valid = df.dropna(subset=["ls_logic50", "delta"])
        if len(valid) > 0:
            mean_d = float(valid["delta"].abs().mean())
            rms_d = float(np.sqrt(((valid["delta"]) ** 2).mean()))
            lines.append(
                f"Mean |Δ| (LS↔EP): {mean_d:.3f} ln µM    RMS Δ: {rms_d:.3f} ln µM    "
                f"({len(valid)} comparable datasets)"
            )
        lines.append("")

        text = "\n".join(lines) + "\n"
        print(text)
        summary_lines.append(text)

        out_txt = results_dir / f"compare_ls_ep_{drug_id}.txt"
        out_txt.write_text(text)
        print(f"  wrote {out_txt}")

        # Scatter plot
        if len(valid) > 0:
            fig, ax = plt.subplots(figsize=(7, 6))
            xs = valid["ls_logic50"].to_numpy()
            ys = valid["ep_log_ic50"].to_numpy()
            yerrs = valid["ep_log_ic50_sigma"].to_numpy()
            ax.errorbar(xs, ys, yerr=yerrs, fmt="o", color="steelblue", capsize=4)

            # 1:1 line
            lo = float(min(xs.min(), ys.min())) - 1.0
            hi = float(max(xs.max(), ys.max())) + 1.0
            ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1, label="LS = EP")

            ax.set_xlabel("LS logic50 (ln µM)")
            ax.set_ylabel("EP log_ic50 (ln µM)")
            ax.set_title(
                f"drug {drug_id} ({drug_name_lookup}) — LS vs EP "
                f"(n={len(valid)}, RMS Δ = {rms_d:.2f})"
            )
            ax.set_aspect("equal", adjustable="datalim")
            ax.legend(fontsize=9)
            for _, r in valid.iterrows():
                ax.annotate(
                    str(r["cosmic_id"]),
                    (r["ls_logic50"], r["ep_log_ic50"]),
                    fontsize=7,
                    xytext=(4, 4),
                    textcoords="offset points",
                    color="dimgray",
                )
            fig.tight_layout()
            out_png = results_dir / f"compare_ls_ep_{drug_id}.png"
            fig.savefig(out_png, dpi=110)
            plt.close(fig)
            print(f"  wrote {out_png}")


if __name__ == "__main__":
    main()
