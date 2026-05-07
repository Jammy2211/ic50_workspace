"""
EP fit on real GDSC2 IC50 data.

Loads the first 5 datasets from `dataset/real/drug_<id>/` and fits the
same global linear + per-dataset Hill factor graph as `ep_sim.py`.
There is no ground truth so no recovery assertion fires — this script
just confirms the EP machinery runs end-to-end on real data and
produces a usable summary file plus per-dataset fit plots.

Datasets are written by `scripts/preprocess_real.py` with
`x = ln(CONC_µM)` so recovered `log_ic50` is directly in `ln(µM)`,
matching the simulator's convention and least_squares.py's output.

Usage:
    python3 scripts/ep_real.py                # fits BOTH drugs sequentially
    python3 scripts/ep_real.py --drug_id 1003

PYAUTO_TEST_MODE
----------------
Run with `PYAUTO_TEST_MODE=1` for fast development iteration. Dynesty
short-circuits sampling and the run finishes in seconds.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import autofit as af

try:
    here = Path(__file__).resolve().parent
except NameError:
    here = Path.cwd()
sys.path.insert(0, str(here))

import util
from util import (
    is_test_mode,
    load_dataset_list,
    plot_dataset,
    run_ep_fit,
    write_ep_summary,
)

DRUGS = [1003, 1073]
NLIVE = 50
MAX_STEPS = 3


def fit_drug(drug_id):
    workspace_root = here.parent
    sample_path = workspace_root / "dataset" / "real" / f"drug_{drug_id}"
    results_dir = here / "results"
    plot_dir = sample_path / "ep_results"
    plot_dir.mkdir(parents=True, exist_ok=True)

    name = f"real_{drug_id}"
    print(f"\n=== ic50_workspace EP fit ({name}) ===")
    print(f"PYAUTO_TEST_MODE = {is_test_mode()}")
    print(f"Sample path:      {sample_path}")
    print(f"Results out:      {results_dir}")
    print()

    loaded = load_dataset_list(sample_path, n_datasets=5, want_truth=False)
    print(f"Loaded {len(loaded['x_array'])} datasets")
    print(f"  noise_sigma     = {loaded['noise_sigma']}")
    print(f"  n_latent        = {loaded['n_latent']}")
    print()

    n_latent = loaded["n_latent"]
    n_datasets = len(loaded["x_array"])

    # Wide priors that work for both drugs (drug 1003 IC50s sit around ~ln(1e-3),
    # drug 1073 around ~ln(1) — log_ic50 can be anywhere in [-10, 10]).
    coef_mean_priors = [
        af.GaussianPrior(mean=0.0, sigma=5.0),       # log_ic50 (ln µM)
        af.GaussianPrior(mean=0.0, sigma=1.0),       # n_log
        af.GaussianPrior(mean=35000.0, sigma=10000.0),  # base
    ]
    coef_matrix_prior_sigmas = [1.0, 0.5, 6000.0]
    hill_priors_per_dataset = [
        (
            af.GaussianPrior(mean=0.0, sigma=5.0),
            af.GaussianPrior(mean=0.0, sigma=1.0),
            af.GaussianPrior(mean=35000.0, sigma=10000.0),
        )
        for _ in range(n_datasets)
    ]

    t_start = time.time()
    recovered = run_ep_fit(
        name=name,
        n_datasets=n_datasets,
        n_latent=n_latent,
        x_array=loaded["x_array"],
        y_array=loaded["y_array"],
        latent_array=loaded["latent_array"],
        noise_sigma=loaded["noise_sigma"],
        coef_mean_priors=coef_mean_priors,
        coef_matrix_prior_sigmas=coef_matrix_prior_sigmas,
        hill_priors_per_dataset=hill_priors_per_dataset,
        nlive=NLIVE,
        max_steps=MAX_STEPS,
    )
    wall_time = time.time() - t_start

    write_ep_summary(
        name=name,
        n_datasets=n_datasets,
        n_latent=n_latent,
        nlive=NLIVE,
        max_steps=MAX_STEPS,
        wall_time_s=wall_time,
        recovered=recovered,
        output_dir=results_dir,
        truth=None,
        test_mode=is_test_mode(),
    )

    for i in range(n_datasets):
        rec_i = {
            "log_ic50": float(recovered["hill_means"][i, 0]),
            "n_log": float(recovered["hill_means"][i, 1]),
            "base": float(recovered["hill_means"][i, 2]),
        }
        info_i = loaded["info_list"][i]
        plot_dataset(
            loaded["x_array"][i],
            loaded["y_array"][i],
            loaded["noise_sigma"],
            title=(
                f"drug {drug_id} dataset_{i} "
                f"(cosmic={info_i['cosmic_id']} drugset={info_i['drugset_id']})\n"
                f"log_ic50={rec_i['log_ic50']:.2f}  n={float(np.exp(rec_i['n_log'])):.2f}  "
                f"base={rec_i['base']:.0f}"
            ),
            fit_params=rec_i,
            output_path=plot_dir / f"dataset_{i}_ep_fit.png",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drug_id", type=int, default=None,
                        help=f"single drug; default loops both {DRUGS}")
    args = parser.parse_args()
    drugs = [args.drug_id] if args.drug_id is not None else DRUGS
    for d in drugs:
        fit_drug(d)

    if is_test_mode():
        print(
            "\nPYAUTO_TEST_MODE active — re-run without the env var for the proper run."
        )
    else:
        print("\nEP fit on real data complete.")


if __name__ == "__main__":
    main()
