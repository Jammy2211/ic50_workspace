"""
EP fit on simulated IC50 data.

Loads the 5 simulated cell-line datasets in `dataset/ic50_sim/` along
with their saved ground truth, fits the workspace's standard global
linear + per-dataset Hill factor graph using AutoFit's expectation
propagation (`af.FactorGraphModel`), and asserts that the recovered
global `coef_mean` lies within 3σ of the simulator's truth.

Configuration matches `concr/scripts/cancer_sim/ep.py` (the reference
this script was ported from), except that priors are centred on this
workspace's simulator truth `[1.0, 0.0, 35000.0]` rather than concr's
`[1.36, 0.35, 86187.0]`.

PYAUTO_TEST_MODE
----------------
Run with `PYAUTO_TEST_MODE=1` for fast development iteration. Dynesty
short-circuits sampling, the run finishes in seconds, and the recovery
assertion is **skipped** (because the sampler hasn't converged). Unset
the env var for the proper run that exercises the assertion.
"""

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

NAME = "sim"
N_DATASETS = 5
NLIVE = 50
MAX_STEPS = 3

workspace_root = here.parent
sim_path = workspace_root / "dataset" / "ic50_sim"
results_dir = here / "results"
plot_dir = sim_path / "ep_results"
plot_dir.mkdir(parents=True, exist_ok=True)

print(f"=== ic50_workspace EP fit ({NAME}) ===")
print(f"PYAUTO_TEST_MODE = {is_test_mode()}")
print(f"Sample path:      {sim_path}")
print(f"Results out:      {results_dir}")
print()

loaded = load_dataset_list(sim_path, n_datasets=N_DATASETS, want_truth=True)

print(f"Loaded {len(loaded['x_array'])} simulated datasets")
print(f"  noise_sigma = {loaded['noise_sigma']}")
print(f"  n_latent    = {loaded['n_latent']}")
print(f"  true coef_mean = {loaded['coef_mean_true']}")
print()

n_latent = loaded["n_latent"]
n_datasets = len(loaded["x_array"])

coef_mean_priors = [
    af.GaussianPrior(mean=0.0, sigma=2.0),       # log_ic50 (ln µM)
    af.GaussianPrior(mean=0.0, sigma=1.0),       # n_log
    af.GaussianPrior(mean=35000.0, sigma=10000.0),  # base
]
coef_matrix_prior_sigmas = [0.5, 0.5, 6000.0]
hill_priors_per_dataset = [
    (
        af.GaussianPrior(mean=0.0, sigma=2.0),
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
    true_params_list=loaded["true_params_list"],
)
wall_time = time.time() - t_start

txt_path, json_path, failures = write_ep_summary(
    name=NAME,
    n_datasets=n_datasets,
    n_latent=n_latent,
    nlive=NLIVE,
    max_steps=MAX_STEPS,
    wall_time_s=wall_time,
    recovered=recovered,
    output_dir=results_dir,
    truth={
        "coef_mean_true": loaded["coef_mean_true"],
        "coef_matrix_true": loaded["coef_matrix_true"],
        "hill_params_true": loaded["hill_params_true"],
    },
    test_mode=is_test_mode(),
)

# Plot each dataset's recovered Hill curve overlaid on the data.
for i in range(n_datasets):
    truth_i = loaded["true_params_list"][i]
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
            f"dataset_{i}  |  log L recovered, "
            f"log_ic50={rec_i['log_ic50']:.2f} (true={truth_i['log_ic50']:.2f})"
        ),
        true_params={
            "log_ic50": truth_i["log_ic50"],
            "n_log": truth_i["n_log"],
            "base": truth_i["base"],
        },
        fit_params=rec_i,
        output_path=plot_dir / f"dataset_{i}_ep_fit.png",
    )

# Hard 3σ assertion on the global coef_mean — only fires in proper-run mode.
_, failures_coef_mean, _ = failures
if failures_coef_mean and not is_test_mode():
    print(
        f"\n{len(failures_coef_mean)} GLOBAL coef_mean assertion(s) FAILED "
        f"(>3σ from truth):"
    )
    for f in failures_coef_mean:
        print(
            f"  {f[0]}: true={f[1]:.4g}  recovered={f[2]:.4g}  "
            f"σ={f[3]:.4g}  σ-dist={f[4]:.2f}"
        )
    raise AssertionError(
        f"ic50_sim EP global coef_mean failed for "
        f"{len(failures_coef_mean)} channel(s)"
    )

if is_test_mode():
    print(
        "\nPYAUTO_TEST_MODE active — recovery assertion skipped. "
        "Re-run without the env var for the proper validation run."
    )
else:
    print(
        "\nAll global coef_mean assertions passed "
        "(recovered population means within 3σ of truth)."
    )
