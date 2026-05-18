"""
Profile ep_sim.py — timing breakdown by category.

Runs the same EP fit as `ep_sim.py` (5 simulated datasets, nlive=50,
max_steps=3) with explicit instrumentation hooks layered on top via
monkey-patching, then writes a structured timing report.

What gets timed
---------------

The total wall time is split into three top-level phases:

  Setup     — build_model_linear, per-dataset Hill models, HillAnalysis
              instances, AnalysisFactor / FactorGraphModel construction.
  Optimise  — `factor_graph.optimise(laplace, ...)`. The EP loop.
  Extract   — walking `ep_result.updated_ep_mean_field` to build
              `hill_means`, `hill_sigmas`, `coef_mean_means`, etc.

The Optimise phase is further broken into:

  (1) Local Hill fits        — sum of HillAnalysis.log_likelihood_function time
  (2) Global fit             — GlobalLinearAnalysis.log_likelihood_function time
  (3a) set_model_approx      — FixedHillCoefEPFactor.set_model_approx (the
                               prior-freeze hook) — runs once per EP
                               iteration on the global factor.
  (3b) PyAutoFit residual    — Optimise total minus (1) + (2) + (3a). This
                               bucket captures Dynesty wrapper overhead,
                               EP message updates, factor-graph traversal,
                               initial sample finding, etc. Drill-down is
                               left for follow-up work.

The instrumentation is via class-level monkey-patches on `HillAnalysis`,
`GlobalLinearAnalysis`, and `FixedHillCoefEPFactor` — so `ep_sim.py` and
`util.py` stay untouched.

Outputs
-------

  scripts/results/ep_sim_profile.md   — human-readable report
  scripts/results/ep_sim_profile.json — machine-readable sidecar

Cache warning
-------------

AutoFit's search-resume short-circuits proper runs when a previous
test-mode result lives at the same model-hash output dir. This script
deletes `output/ep_sim/` at startup so we always get a real run.
"""

import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import autofit as af

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))

import util
from util import (
    FixedHillCoefEPFactor,
    GlobalLinearAnalysis,
    HillAnalysis,
    build_model_linear,
    build_per_dataset_models,
    load_dataset_list,
)

NAME = "sim"
N_DATASETS = 5
NLIVE = 50
MAX_STEPS = 3


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------

TIMING = {
    "ll_calls": defaultdict(int),
    "ll_time": defaultdict(float),
    "set_model_approx_time": 0.0,
    "set_model_approx_calls": 0,
    "dynesty_fit_time": 0.0,
    "dynesty_fit_calls": 0,
}


def _install_instrumentation():
    """Monkey-patch the analysis + factor classes to count and time."""
    orig_hill_ll = HillAnalysis.log_likelihood_function
    orig_global_ll = GlobalLinearAnalysis.log_likelihood_function
    orig_set_model_approx = FixedHillCoefEPFactor.set_model_approx
    orig_dynesty_fit = af.DynestyStatic.fit

    def _wrapped_hill_ll(self, instance, xp=np):
        name = getattr(self, "_profile_name", "hill_unknown")
        TIMING["ll_calls"][name] += 1
        t0 = time.perf_counter()
        try:
            return orig_hill_ll(self, instance, xp=xp)
        finally:
            TIMING["ll_time"][name] += time.perf_counter() - t0

    def _wrapped_global_ll(self, instance, xp=np):
        TIMING["ll_calls"]["global"] += 1
        t0 = time.perf_counter()
        try:
            return orig_global_ll(self, instance, xp=xp)
        finally:
            TIMING["ll_time"]["global"] += time.perf_counter() - t0

    def _wrapped_set_model_approx(self, model_approx):
        t0 = time.perf_counter()
        try:
            return orig_set_model_approx(self, model_approx)
        finally:
            TIMING["set_model_approx_time"] += time.perf_counter() - t0
            TIMING["set_model_approx_calls"] += 1

    def _wrapped_dynesty_fit(self, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return orig_dynesty_fit(self, *args, **kwargs)
        finally:
            TIMING["dynesty_fit_time"] += time.perf_counter() - t0
            TIMING["dynesty_fit_calls"] += 1

    HillAnalysis.log_likelihood_function = _wrapped_hill_ll
    GlobalLinearAnalysis.log_likelihood_function = _wrapped_global_ll
    FixedHillCoefEPFactor.set_model_approx = _wrapped_set_model_approx
    af.DynestyStatic.fit = _wrapped_dynesty_fit


# ---------------------------------------------------------------------------
# Profiled EP run (inlines run_ep_fit so we can bracket internal phases)
# ---------------------------------------------------------------------------


def profiled_run():
    workspace_root = here.parent
    sim_path = workspace_root / "dataset" / "ic50_sim"

    # Clear AutoFit's checkpoint to avoid short-circuiting a proper run with a
    # previous test-mode cache. See feedback_autofit_cache_resume_pyauto_test_mode
    # in memory for the full story.
    cache_dir = workspace_root / "output" / f"ep_{NAME}"
    if cache_dir.exists():
        print(f"Removing cached AutoFit output: {cache_dir}")
        shutil.rmtree(cache_dir)

    loaded = load_dataset_list(sim_path, n_datasets=N_DATASETS, want_truth=True)
    n_latent = loaded["n_latent"]
    n_datasets = len(loaded["x_array"])

    coef_mean_priors = [
        af.GaussianPrior(mean=0.0, sigma=2.0),
        af.GaussianPrior(mean=0.0, sigma=1.0),
        af.GaussianPrior(mean=35000.0, sigma=10000.0),
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

    # ---- Phase 1: Setup ----
    t_setup_start = time.perf_counter()

    model_linear, hill_coef_priors = build_model_linear(
        n_latent=n_latent,
        n_datasets=n_datasets,
        coef_mean_priors=coef_mean_priors,
        coef_matrix_prior_sigmas=coef_matrix_prior_sigmas,
        hill_priors_per_dataset=hill_priors_per_dataset,
    )
    model_list = build_per_dataset_models(hill_coef_priors)

    analysis_list = []
    for i in range(n_datasets):
        a = HillAnalysis(
            x=loaded["x_array"][i],
            y=loaded["y_array"][i],
            noise_sigma=loaded["noise_sigma"],
            true_params=loaded["true_params_list"][i],
        )
        a._profile_name = f"dataset_{i}"
        analysis_list.append(a)

    fallback_sigmas = np.tile(np.array([1.0, 0.5, 10000.0]), (n_datasets, 1))
    analysis_global = GlobalLinearAnalysis(
        latents=np.asarray(loaded["latent_array"], dtype=float),
        fallback_sigmas=fallback_sigmas,
    )

    paths = af.DirectoryPaths(path_prefix=Path(f"ep_{NAME}"), name="ep")
    search_local = af.DynestyStatic(
        paths=paths, nlive=NLIVE, sample="rwalk", force_x1_cpu=True
    )
    search_global = af.DynestyStatic(
        paths=paths, nlive=NLIVE, sample="rwalk", force_x1_cpu=True
    )

    analysis_factor_list = []
    for i, (model, analysis) in enumerate(zip(model_list, analysis_list)):
        analysis_factor_list.append(
            af.AnalysisFactor(
                prior_model=model,
                analysis=analysis,
                optimiser=search_local,
                name=f"dataset_{i}",
            )
        )

    analysis_factor_global = FixedHillCoefEPFactor(
        prior_model=model_linear,
        analysis=analysis_global,
        hill_coef_priors=hill_coef_priors,
        local_factor_names=[f"dataset_{i}" for i in range(n_datasets)],
        optimiser=search_global,
        name="global",
    )

    factor_graph = af.FactorGraphModel(
        *analysis_factor_list, analysis_factor_global
    )
    laplace = af.LaplaceOptimiser()

    t_setup = time.perf_counter() - t_setup_start

    # ---- Phase 2: Optimise ----
    print(f"\nRunning EP: nlive={NLIVE}, max_steps={MAX_STEPS}, n_datasets={n_datasets}")
    t_optimise_start = time.perf_counter()
    ep_result = factor_graph.optimise(
        laplace,
        paths=paths,
        ep_history=af.EPHistory(kl_tol=1.0),
        max_steps=MAX_STEPS,
    )
    t_optimise = time.perf_counter() - t_optimise_start

    # ---- Phase 3: Extract ----
    t_extract_start = time.perf_counter()

    fmf = ep_result.updated_ep_mean_field.factor_mean_field
    mf = ep_result.updated_ep_mean_field.mean_field
    factor_by_name = {f.name: f for f in fmf if getattr(f, "name", None)}

    hill_means = np.empty((n_datasets, 3))
    hill_sigmas = np.empty((n_datasets, 3))
    for i, prior_row in enumerate(hill_coef_priors):
        ds_factor = factor_by_name[f"dataset_{i}"]
        ds_mf = fmf[ds_factor]
        for j, prior in enumerate(prior_row):
            msg = ds_mf[prior]
            hill_means[i, j] = float(msg.mean)
            hill_sigmas[i, j] = float(msg.sigma)

    _ = np.array([float(mf[model_linear.coef_mean[j]].mean) for j in range(3)])
    _ = np.array([float(mf[model_linear.coef_mean[j]].sigma) for j in range(3)])

    coef_matrix_means = np.empty((n_latent, 3))
    coef_matrix_sigmas = np.empty((n_latent, 3))
    for k in range(n_latent):
        for j in range(3):
            msg = mf[model_linear.coef_matrix[k, j]]
            coef_matrix_means[k, j] = float(msg.mean)
            coef_matrix_sigmas[k, j] = float(msg.sigma)

    t_extract = time.perf_counter() - t_extract_start

    return {
        "n_datasets": n_datasets,
        "n_latent": n_latent,
        "nlive": NLIVE,
        "max_steps": MAX_STEPS,
        "t_setup": t_setup,
        "t_optimise": t_optimise,
        "t_extract": t_extract,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_time(t):
    if t < 60:
        return f"{t:.1f} s"
    if t < 3600:
        return f"{t/60:.1f} min"
    if t < 86400:
        return f"{t/3600:.1f} h"
    return f"{t/86400:.1f} d"


def build_report(phase_times, timing, output_dir):
    n_datasets = phase_times["n_datasets"]
    n_latent = phase_times["n_latent"]
    nlive = phase_times["nlive"]
    max_steps = phase_times["max_steps"]
    t_setup = phase_times["t_setup"]
    t_optimise = phase_times["t_optimise"]
    t_extract = phase_times["t_extract"]
    t_total = t_setup + t_optimise + t_extract

    per_factor = []
    local_ll_time = 0.0
    local_ll_calls = 0
    for i in range(n_datasets):
        name = f"dataset_{i}"
        t = timing["ll_time"].get(name, 0.0)
        c = timing["ll_calls"].get(name, 0)
        local_ll_time += t
        local_ll_calls += c
        per_factor.append({"name": name, "calls": c, "time_s": t})

    global_ll_time = timing["ll_time"].get("global", 0.0)
    global_ll_calls = timing["ll_calls"].get("global", 0)
    sma_time = timing["set_model_approx_time"]
    sma_calls = timing["set_model_approx_calls"]
    dynesty_fit_time = timing["dynesty_fit_time"]
    dynesty_fit_calls = timing["dynesty_fit_calls"]

    total_ll_time = local_ll_time + global_ll_time

    # Residual = optimise minus everything we accounted for explicitly.
    pyautofit_overhead = (
        t_optimise - total_ll_time - sma_time
    )

    # Each EP iteration runs N local Dynesty fits + 1 global Dynesty fit.
    # `dynesty_fit_calls` should equal (N + 1) × max_steps in the worst case,
    # but EP terminates early when KL between iterations drops below `kl_tol`.
    fits_per_iter = (n_datasets + 1)
    expected_dynesty_fits_max = fits_per_iter * max_steps
    # Infer how many EP iterations actually ran from the observed fit count.
    iters_run_observed = (
        dynesty_fit_calls / fits_per_iter if dynesty_fit_calls else 0
    )

    # Per-Dynesty-fit wrapper overhead: time spent inside search.fit() that
    # ISN'T in our log_likelihood_function patches.
    dynesty_wrapper_time = dynesty_fit_time - total_ll_time
    dynesty_wrapper_per_fit = (
        dynesty_wrapper_time / dynesty_fit_calls if dynesty_fit_calls else 0.0
    )

    # Per-iteration costs — average across iterations that actually ran
    # (`iters_run_observed`), not the `max_steps` cap. This makes the
    # per-iter cost insensitive to early termination.
    iters_for_avg = max(1.0, iters_run_observed)
    local_ll_per_dataset_per_iter = local_ll_time / (n_datasets * iters_for_avg)
    global_ll_per_iter = global_ll_time / iters_for_avg
    sma_per_iter = sma_time / iters_for_avg
    # EP-loop orchestration overhead: optimise time spent outside any
    # search.fit() call — factor-graph traversal, message updates, etc.
    ep_orchestration_time = t_optimise - dynesty_fit_time - sma_time
    ep_orchestration_per_iter = ep_orchestration_time / iters_for_avg

    # Refined scaling model — splits the residual into linear-in-N (per-Dynesty-fit
    # wrapper overhead × (N+1) fits per iter) and constant-per-iteration buckets.
    #
    #   total(N, M) = a
    #               + M × [(N + 1) × dynesty_wrapper_per_fit]
    #               + M × N × local_ll_per_dataset_per_iter
    #               + M × global_ll_per_iter
    #               + M × sma_per_iter
    #               + M × ep_orchestration_per_iter
    #
    # We use `M = iters_run_observed` (≈ 2 in this run because EP converged
    # early via kl_tol=1.0). At larger N the convergence rate may differ —
    # the report flags this assumption explicitly.
    a = t_setup + t_extract
    M_proj = iters_for_avg

    def project(N, M=M_proj):
        return (
            a
            + M * (N + 1) * dynesty_wrapper_per_fit
            + M * N * local_ll_per_dataset_per_iter
            + M * global_ll_per_iter
            + M * sma_per_iter
            + M * ep_orchestration_per_iter
        )

    projections = {n: project(n) for n in [5, 100, 1000, 10000]}

    # Per-target breakdown so we can show users what fraction is "Dynesty wrapper"
    # vs "actual likelihood" at each N.
    projection_buckets = {}
    for n in [5, 100, 1000, 10000]:
        M = M_proj
        projection_buckets[n] = {
            "setup_extract": a,
            "dynesty_wrapper": M * (n + 1) * dynesty_wrapper_per_fit,
            "local_ll": M * n * local_ll_per_dataset_per_iter,
            "global_ll": M * global_ll_per_iter,
            "set_model_approx": M * sma_per_iter,
            "ep_orchestration": M * ep_orchestration_per_iter,
            "total": project(n, M),
        }

    lines = []
    lines.append("# EP Sim Profile — Timing Breakdown")
    lines.append("")
    lines.append(
        f"Run config: n_datasets={n_datasets}, n_latent={n_latent}, "
        f"nlive={nlive}, max_steps={max_steps}"
    )
    lines.append("")
    lines.append(f"**Total wall time: {t_total:.2f} s** ({t_total/60:.2f} min)")
    lines.append("")

    lines.append("## Top-level breakdown")
    lines.append("")
    lines.append("| Phase | Time (s) | % of total |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Setup (build model + analyses + factor graph) | {t_setup:.3f} | {100*t_setup/t_total:.1f} |")
    lines.append(f"| Optimise (`factor_graph.optimise`) | {t_optimise:.3f} | {100*t_optimise/t_total:.1f} |")
    lines.append(f"| Extract (walk ep_result → arrays) | {t_extract:.3f} | {100*t_extract/t_total:.1f} |")
    lines.append("")

    lines.append("## Optimise-phase breakdown")
    lines.append("")
    lines.append("| Category | Time (s) | % of optimise |")
    lines.append("|---|---:|---:|")
    lines.append(f"| (1) Local Hill fits — `HillAnalysis.log_likelihood_function` (5 factors × {max_steps} iter) | {local_ll_time:.3f} | {100*local_ll_time/t_optimise:.1f} |")
    lines.append(f"| (2) Global fit — `GlobalLinearAnalysis.log_likelihood_function` | {global_ll_time:.3f} | {100*global_ll_time/t_optimise:.1f} |")
    lines.append(f"| (3a) `set_model_approx` prior-freeze hook | {sma_time:.3f} | {100*sma_time/t_optimise:.1f} |")
    lines.append(f"| (3b) Dynesty wrapper overhead (search.fit minus LL evals) | {dynesty_wrapper_time:.3f} | {100*dynesty_wrapper_time/t_optimise:.1f} |")
    lines.append(f"| (3c) EP-loop orchestration (optimise minus search.fit minus set_model_approx) | {ep_orchestration_time:.3f} | {100*ep_orchestration_time/t_optimise:.1f} |")
    lines.append("")
    lines.append(
        f"Total Dynesty fits: {dynesty_fit_calls} "
        f"(worst case (N+1)×max_steps = {expected_dynesty_fits_max}; "
        f"observed implies ~{iters_run_observed:.1f} EP iterations ran before convergence)"
    )
    lines.append(f"Total search.fit wall time: {dynesty_fit_time:.3f} s — of which {100*total_ll_time/dynesty_fit_time:.1f}% was actual likelihood evaluation.")
    lines.append(f"Per-Dynesty-fit wrapper overhead: {dynesty_wrapper_per_fit:.3f} s/fit")
    lines.append("")
    lines.append("**Definitions:**")
    lines.append("- *Dynesty wrapper overhead* = time inside `search.fit(...)` not spent in `log_likelihood_function`. Covers sampler init, path/run setup, bound construction, weight/posterior post-processing, plot generation.")
    lines.append("- *EP-loop orchestration* = time inside `factor_graph.optimise(...)` not spent in any `search.fit(...)` call. Covers factor traversal, EP message updates between iterations, the LaplaceOptimiser bookkeeping.")
    lines.append("")

    lines.append("## Per-Hill-factor breakdown")
    lines.append("")
    lines.append("| Factor | LL calls | Total LL time (s) | Time/call (ms) | Per iteration (s) |")
    lines.append("|---|---:|---:|---:|---:|")
    for pf in per_factor:
        per_call_ms = 1000 * pf["time_s"] / pf["calls"] if pf["calls"] else 0.0
        per_iter_s = pf["time_s"] / max_steps
        lines.append(
            f"| {pf['name']} | {pf['calls']} | {pf['time_s']:.3f} | "
            f"{per_call_ms:.4f} | {per_iter_s:.3f} |"
        )
    lines.append("")

    lines.append("## Global factor breakdown")
    lines.append("")
    lines.append(f"- Likelihood calls: {global_ll_calls}")
    lines.append(f"- Total LL time: {global_ll_time:.3f} s")
    if global_ll_calls:
        lines.append(
            f"- Time per call: {1000*global_ll_time/global_ll_calls:.4f} ms"
        )
    lines.append(f"- Per iteration: {global_ll_time/max_steps:.3f} s")
    lines.append(
        f"- `set_model_approx` calls: {sma_calls} (≈ {sma_calls/max_steps:.1f} per EP iteration), "
        f"total {sma_time:.4f} s"
    )
    lines.append("")

    lines.append("## Scaling projection")
    lines.append("")
    lines.append("**Model components** (each per EP iteration, summed across `max_steps` iterations):")
    lines.append("")
    lines.append("| Bucket | Per-fit cost | Scales as |")
    lines.append("|---|---:|---|")
    lines.append(f"| Dynesty wrapper overhead | {dynesty_wrapper_per_fit:.3f} s/fit | (N + 1) per iteration |")
    lines.append(f"| Local LL evaluation | {local_ll_per_dataset_per_iter:.4f} s/dataset | N per iteration |")
    lines.append(f"| Global LL evaluation | {global_ll_per_iter:.3f} s | constant per iteration |")
    lines.append(f"| `set_model_approx` | {sma_per_iter:.4f} s | constant per iteration (walks N×3 priors) |")
    lines.append(f"| EP-loop orchestration | {ep_orchestration_per_iter:.3f} s | assumed constant per iteration |")
    lines.append(f"| Setup + extract | {a:.3f} s | one-off |")
    lines.append("")
    lines.append("**Assumptions** (each a verifiable prediction once we re-measure at larger N):")
    lines.append("1. Dynesty wrapper overhead is constant **per fit**. Likely true — paths/sampler init don't depend on `n_datasets`.")
    lines.append("2. Local LL cost scales linearly in N (each Hill is independent).")
    lines.append("3. Global LL cost is constant in N (`set_model_approx` freezes `hill_coef` to 18 free params).")
    lines.append("4. `set_model_approx` walks N×3 priors per call — should grow linearly. Lumped as constant here because it's <0.01% of total at N=5.")
    lines.append("5. EP-loop orchestration is constant per iteration. **Uncertain** — message updates may grow with `n_datasets`. Verify at N=100.")
    lines.append(f"6. EP converges in `M = {M_proj:.1f}` iterations at every N. **Most uncertain** — convergence rate depends on data and tolerance; larger samples may need more iterations to satisfy `kl_tol`.")
    lines.append("")
    lines.append(f"**Projection uses M = {M_proj:.1f} EP iterations** (observed in this run before `kl_tol=1.0` convergence; `max_steps={max_steps}` is the cap). Tighten `kl_tol` if you want the projection to assume full max_steps iterations.")
    lines.append("")
    lines.append("| n_datasets | M | Setup | Dynesty wrapper | Local LL | Global LL | sma | EP orch | **Total** |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for n in [5, 100, 1000, 10000]:
        b = projection_buckets[n]
        lines.append(
            f"| {n} | {M_proj:.0f} | "
            f"{_fmt_time(b['setup_extract'])} | "
            f"{_fmt_time(b['dynesty_wrapper'])} | "
            f"{_fmt_time(b['local_ll'])} | "
            f"{_fmt_time(b['global_ll'])} | "
            f"{_fmt_time(b['set_model_approx'])} | "
            f"{_fmt_time(b['ep_orchestration'])} | "
            f"**{_fmt_time(b['total'])}** |"
        )
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append("- Single-run measurement; sampling variance not captured. Re-run a couple of times if any bucket looks borderline.")
    lines.append("- `nlive=50` is small (`<= 2*ndim` per Dynesty's own warning for the global factor). Production runs would use higher nlive — both LL buckets scale roughly linearly in nlive, Dynesty wrapper overhead is mostly nlive-independent.")
    lines.append("- The (3b) Dynesty-wrapper bucket lumps together every non-LL cost inside `search.fit(...)`. The PyAutoFit log shows `corner_anesthetic` plot attempts on every fit, plus `Removing search internal folder` cleanup — both are obvious candidates for a `--profile` mode that skips them. A `cProfile`/`py-spy` pass on the same workload would attribute the wrapper bucket to specific functions.")
    lines.append("")

    lines.append("## Suggested follow-up optimisation targets")
    lines.append("")
    lines.append("Pick the largest bucket from the projection table at the target N:")
    lines.append("")
    lines.append("- **Dynesty wrapper dominates** (large per-fit overhead × `(N+1)·M` fits) → the biggest lever. Options: cache `paths` / sampler config so each per-factor fit doesn't re-initialise; share `DynestyStatic` instance reuse across iterations; suppress per-fit plot generation (`corner_anesthetic` log lines show this happens); skip `Removing search internal folder` cleanup on cached runs.")
    lines.append("- **Local LL dominates** (only at large N) → parallelise the N independent local fits across CPU cores per iteration; or JAX-vmap the Hill likelihood across datasets within one fused fit.")
    lines.append("- **Global LL dominates** → reduce global nlive (currently `nlive ≤ 2·ndim` so already at floor — try Laplace approximation for the global factor instead of full Dynesty).")
    lines.append("- **`set_model_approx` dominates** (only at very large N) → cache the `prior → message` lookup; current impl walks the whole mean-field dict each call.")
    lines.append("- **EP-loop orchestration dominates** → run `cProfile` / `py-spy` on the same workload and triage hot loops in `autofit/graphical/`.")
    lines.append("")

    md_path = output_dir / "ep_sim_profile.md"
    md_path.write_text("\n".join(lines) + "\n")

    json_data = {
        "config": {
            "n_datasets": n_datasets,
            "n_latent": n_latent,
            "nlive": nlive,
            "max_steps": max_steps,
        },
        "phases": {
            "setup_s": t_setup,
            "optimise_s": t_optimise,
            "extract_s": t_extract,
            "total_s": t_total,
        },
        "optimise_breakdown": {
            "local_ll_time_s": local_ll_time,
            "local_ll_calls": local_ll_calls,
            "global_ll_time_s": global_ll_time,
            "global_ll_calls": global_ll_calls,
            "set_model_approx_time_s": sma_time,
            "set_model_approx_calls": sma_calls,
            "dynesty_fit_time_s": dynesty_fit_time,
            "dynesty_fit_calls": dynesty_fit_calls,
            "dynesty_fit_calls_max_steps_worst_case": expected_dynesty_fits_max,
            "iters_run_observed": iters_run_observed,
            "dynesty_wrapper_time_s": dynesty_wrapper_time,
            "ep_orchestration_time_s": ep_orchestration_time,
            "unaccounted_residual_s": pyautofit_overhead - dynesty_wrapper_time - ep_orchestration_time,
        },
        "per_hill_factor": per_factor,
        "scaling_model": {
            "form": (
                "total(N, M) = a + M*[(N+1)*dynesty_wrapper_per_fit "
                "+ N*local_ll_per_dataset_per_iter + global_ll_per_iter "
                "+ sma_per_iter + ep_orchestration_per_iter]"
            ),
            "a_s": a,
            "dynesty_wrapper_per_fit_s": dynesty_wrapper_per_fit,
            "local_ll_per_dataset_per_iter_s": local_ll_per_dataset_per_iter,
            "global_ll_per_iter_s": global_ll_per_iter,
            "sma_per_iter_s": sma_per_iter,
            "ep_orchestration_per_iter_s": ep_orchestration_per_iter,
            "projections_s": projections,
            "projection_buckets_s": projection_buckets,
        },
    }
    json_path = output_dir / "ep_sim_profile.json"
    json_path.write_text(json.dumps(json_data, indent=2))

    return md_path, json_path


def main():
    _install_instrumentation()

    print("=== EP profile run ===")
    print(f"n_datasets={N_DATASETS}, nlive={NLIVE}, max_steps={MAX_STEPS}")

    phase_times = profiled_run()

    output_dir = here / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path, json_path = build_report(phase_times, TIMING, output_dir)

    t_total = phase_times["t_setup"] + phase_times["t_optimise"] + phase_times["t_extract"]
    print(f"\nTotal wall time: {t_total:.2f} s")
    print(f"  setup:    {phase_times['t_setup']:.2f} s")
    print(f"  optimise: {phase_times['t_optimise']:.2f} s")
    print(f"  extract:  {phase_times['t_extract']:.2f} s")
    print(f"\nMarkdown report: {md_path}")
    print(f"JSON sidecar:    {json_path}")


if __name__ == "__main__":
    main()
