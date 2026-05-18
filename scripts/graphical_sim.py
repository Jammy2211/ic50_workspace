"""
Graphical-model fit on simulated IC50 data.

Companion to `ep_sim.py`. Both scripts fit the same factor graph — the
same per-dataset Hill models linked to the same global linear regression
on the latent gene-expression features — but they sample that graph in
fundamentally different ways:

  * `ep_sim.py` (expectation propagation): the EP optimiser partitions
    the graph into per-dataset factors plus one global linear factor.
    Each factor's local search runs in a low-dimensional sub-space
    (3 params for a Hill factor, `n_latent*3 + 3` for the global one)
    while EP message passing reconciles the per-dataset Hill posteriors
    with the global linear constraint across iterations.
  * `graphical_sim.py` (this file): one non-linear search over the full
    factor graph. Every per-dataset Hill coefficient, every entry of
    `coef_matrix`, and every component of `coef_mean` is sampled
    jointly in **one** parameter space of size
    `n_datasets*3 + n_latent*3 + 3`. The shared-prior wiring in
    `build_model_linear` is what makes `hill_coef[i, j]` and the
    corresponding `hill[i].log_ic50` / `n_log` / `base` resolve to the
    same free variable.

Why have both? At small `n_datasets` the graphical fit is a perfectly
viable, conceptually simpler reference — it explores the joint posterior
of all parameters without any iterative approximation. As the sample
size grows the dimension scales linearly in `n_datasets`, and the
joint search becomes the bottleneck; EP scales as repeated local
sub-searches and is what we lean on for the full GDSC2 sample. Having
both available in the workspace lets us cross-check the EP result on
the same simulated data that `ep_sim.py` covers — they should recover
the same global `coef_mean` within 3σ.

The 3σ assertion on the global `coef_mean` mirrors `ep_sim.py` so the
script doubles as a sanity check on the graphical-fit path.

Configuration matches `ep_sim.py` (priors centred on the simulator
truth `[1.0, 0.0, 35000.0]`).

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
    DEFAULT_REGRESSION_SIGMAS,
    is_test_mode,
    load_dataset_list,
    plot_dataset,
    run_graphical_fit,
    write_graphical_summary,
)

NAME = "sim"
N_DATASETS = 5
NLIVE = 50

workspace_root = here.parent
sim_path = workspace_root / "dataset" / "ic50_sim"
results_dir = here / "results"
plot_dir = sim_path / "graphical_results"
plot_dir.mkdir(parents=True, exist_ok=True)

print(f"=== ic50_workspace graphical fit ({NAME}) ===")
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
recovered = run_graphical_fit(
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
    regression_sigmas=DEFAULT_REGRESSION_SIGMAS,
    true_params_list=loaded["true_params_list"],
)
wall_time = time.time() - t_start

txt_path, json_path, failures = write_graphical_summary(
    name=NAME,
    n_datasets=n_datasets,
    n_latent=n_latent,
    nlive=NLIVE,
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
            f"dataset_{i}  |  graphical fit, "
            f"log_ic50={rec_i['log_ic50']:.2f} (true={truth_i['log_ic50']:.2f})"
        ),
        true_params={
            "log_ic50": truth_i["log_ic50"],
            "n_log": truth_i["n_log"],
            "base": truth_i["base"],
        },
        fit_params=rec_i,
        output_path=plot_dir / f"dataset_{i}_graphical_fit.png",
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
        f"ic50_sim graphical global coef_mean failed for "
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
