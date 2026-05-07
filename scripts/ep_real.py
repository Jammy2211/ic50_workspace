"""
EP fit on real GDSC2 IC50 data.

Loads the first 5 datasets from `dataset/cancer_real__drug_1073/` and
fits the same global linear + per-dataset Hill factor graph as
`ep_sim.py`. There is no ground truth so no recovery assertion fires —
this script just confirms the EP machinery runs end-to-end on real
data and produces a usable summary file plus per-dataset fit plots.

Priors are the same as `ep_sim.py` because the simulator was tuned to
match real-data scales; broaden them in this file if a future dataset
or drug needs different ranges.

PYAUTO_TEST_MODE
----------------
Run with `PYAUTO_TEST_MODE=1` for fast development iteration. Dynesty
short-circuits sampling and the run finishes in seconds. Unset the
env var for the proper run that takes ~20–40 min.
"""

import sys
import time
from pathlib import Path

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

NAME = "real"
N_DATASETS = 5
NLIVE = 50
MAX_STEPS = 5

workspace_root = here.parent
real_path = workspace_root / "dataset" / "cancer_real__drug_1073"
results_dir = here / "results"
plot_dir = real_path / "ep_results"
plot_dir.mkdir(parents=True, exist_ok=True)

print(f"=== ic50_workspace EP fit ({NAME}) ===")
print(f"PYAUTO_TEST_MODE = {is_test_mode()}")
print(f"Sample path:      {real_path}")
print(f"Results out:      {results_dir}")
print()

loaded = load_dataset_list(real_path, n_datasets=N_DATASETS, want_truth=False)

print(f"Loaded {len(loaded['x_array'])} real datasets")
print(f"  noise_sigma = {loaded['noise_sigma']}")
print(f"  n_latent    = {loaded['n_latent']}")
print()

n_latent = loaded["n_latent"]
n_datasets = len(loaded["x_array"])

coef_mean_priors = [
    af.GaussianPrior(mean=1.0, sigma=2.0),
    af.GaussianPrior(mean=0.0, sigma=1.0),
    af.GaussianPrior(mean=35000.0, sigma=10000.0),
]
coef_matrix_prior_sigmas = [0.5, 0.5, 6000.0]
hill_priors_per_dataset = [
    (
        af.GaussianPrior(mean=1.0, sigma=1.0),
        af.GaussianPrior(mean=0.0, sigma=0.5),
        af.GaussianPrior(mean=35000.0, sigma=10000.0),
    )
    for _ in range(n_datasets)
]

t_start = time.time()
recovered = run_ep_fit(
    name=NAME,
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
    name=NAME,
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
    plot_dataset(
        loaded["x_array"][i],
        loaded["y_array"][i],
        loaded["noise_sigma"],
        title=(
            f"dataset_{i} (real, drug 1073)  |  "
            f"log_ic50={rec_i['log_ic50']:.2f}  "
            f"n={float(__import__('numpy').exp(rec_i['n_log'])):.2f}  "
            f"base={rec_i['base']:.0f}"
        ),
        fit_params=rec_i,
        output_path=plot_dir / f"dataset_{i}_ep_fit.png",
    )

if is_test_mode():
    print(
        "\nPYAUTO_TEST_MODE active — re-run without the env var for the "
        "proper run (~20–40 min)."
    )
else:
    print("\nEP fit on real data complete.")
