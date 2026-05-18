"""
IC50 Workspace — Shared Hill-Curve, Plotting, and EP Fitting Utilities
=======================================================================

This module is the single source of truth for IC50 dose-response math,
plotting, and AutoFit-driven Expectation Propagation (EP) fits used
throughout `ic50_workspace`. Both simulated and real datasets pass
through the same `hill_curve`, `log_likelihood`, `plot_dataset`, and
`run_ep_fit` so sims and real data never visually or numerically drift.

Sign convention
---------------
This module uses the canonical dose-response sign convention,

    y = base / (1 + exp(n * (x - log_ic50))),    n = exp(n_log)

which is **monotonically decreasing in x** (high viability at low dose,
low viability at high dose). This matches the visual convention used by
the group, exemplified by `concr/cancer_legacy/real/viz_hill.py`.

JAX support
-----------
`hill_curve` and `log_likelihood` accept an `xp` keyword argument so the
same formula runs under either `numpy` or `jax.numpy`. Default is numpy
so simulator / likelihood-function tutorial paths are unchanged. The
AutoFit-side `HillAnalysis` and `GlobalLinearAnalysis` use JAX
(`use_jax=True`) and a `@jax.jit`-compiled inner function for fast EP
likelihood evaluation.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import autofit as af

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Core Hill curve and Gaussian log-likelihood — xp-aware
# ---------------------------------------------------------------------------


def hill_curve(x, log_ic50, n_log, base, xp=np):
    """Evaluate the Hill dose-response curve.

    Parameters
    ----------
    x : array_like
        Log-concentration grid in `ln(µM)` (e.g. `np.log(CONC_µM)` for
        real data, `np.linspace(-3.45, 3.45, 7)` for the simulator).
    log_ic50 : float
        Half-maximal log-concentration; the curve's inflection point.
    n_log : float
        Hill coefficient stored in log space; the actual slope is
        ``n = exp(n_log)``.
    base : float
        Upper plateau (intensity at low dose, where the cells are alive).
    xp : module, optional
        Array namespace; pass ``jax.numpy`` for JAX dispatch. Defaults
        to ``numpy``.

    Returns
    -------
    array
        Curve values evaluated at ``x``. Decreasing in ``x``: ``y -> base``
        as ``x -> -inf`` and ``y -> 0`` as ``x -> +inf``.
    """
    x = xp.asarray(x, dtype=float)
    n = xp.exp(n_log)
    nkx = xp.clip(n * (x - log_ic50), -500.0, 500.0)
    return base / (1.0 + xp.exp(nkx))


def log_likelihood(x, y, noise_sigma, log_ic50, n_log, base, xp=np):
    """Gaussian log-likelihood of ``y`` given Hill parameters.

    Returns ``-0.5 * sum( ((y_pred - y) / noise_sigma)**2 )``, the standard
    normal-likelihood up to additive constants. Use this rather than
    re-deriving it locally so every consumer in ic50_workspace agrees on
    the sign and the noise normalisation.
    """
    y = xp.asarray(y, dtype=float)
    y_pred = hill_curve(x, log_ic50, n_log, base, xp=xp)
    z = (y_pred - y) / noise_sigma
    return float(-0.5 * xp.dot(z, z))


# ---------------------------------------------------------------------------
# JIT'd helpers used inside AutoFit Analysis classes
# ---------------------------------------------------------------------------


@jax.jit
def _hill_log_likelihood_jit(x, y, sigma, log_ic50, n_log, base):
    n = jnp.exp(n_log)
    nkx = jnp.clip(n * (x - log_ic50), -500.0, 500.0)
    y_pred = base / (1.0 + jnp.exp(nkx))
    z = (y_pred - y) / sigma
    z = jnp.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return -0.5 * jnp.dot(z, z)


@jax.jit
def _global_log_likelihood_jit(latents, coef_matrix, coef_mean, ep_means, ep_sigmas):
    pred = latents @ coef_matrix + coef_mean
    z = (pred - ep_means) / ep_sigmas
    z = jnp.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return -0.5 * jnp.sum(z ** 2)


# ---------------------------------------------------------------------------
# Plotting (unchanged numpy path)
# ---------------------------------------------------------------------------


def plot_dataset(
    x,
    y,
    noise_sigma,
    *,
    title=None,
    true_params=None,
    fit_params=None,
    output_path=None,
):
    """Plot a single IC50 dataset in the group's standard style.

    Steel-blue scatter with capped errorbars; optional smooth Hill overlays
    when `true_params` (tomato) and/or `fit_params` (crimson) are supplied;
    optional dashed vertical line at the IC50 of whichever curve is
    overlaid.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x_min, x_max = float(x.min()), float(x.max())
    x_fine = np.linspace(x_min - 0.2, x_max + 0.2, 400)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        x, y, yerr=noise_sigma, fmt="o", color="steelblue", capsize=4, label="Data"
    )

    if true_params is not None:
        y_true = hill_curve(
            x_fine,
            true_params["log_ic50"],
            true_params["n_log"],
            true_params["base"],
        )
        ax.plot(x_fine, y_true, color="tomato", lw=2, label="True Hill curve")
        ax.axvline(
            true_params["log_ic50"],
            color="tomato",
            linestyle="--",
            linewidth=1,
            label=f"True log IC50 = {true_params['log_ic50']:.2f}",
        )

    if fit_params is not None:
        y_fit = hill_curve(
            x_fine,
            fit_params["log_ic50"],
            fit_params["n_log"],
            fit_params["base"],
        )
        ax.plot(x_fine, y_fit, color="crimson", lw=2, label="Best-fit Hill curve")
        ax.axvline(
            fit_params["log_ic50"],
            color="crimson",
            linestyle="--",
            linewidth=1,
            label=f"Fit log IC50 = {fit_params['log_ic50']:.2f}",
        )

    ax.set_xlabel("Log concentration (ln µM)")
    ax.set_ylabel("Intensity")
    if title is None and true_params is not None:
        title = (
            f"log_ic50={true_params['log_ic50']:.2f}  "
            f"n={np.exp(true_params['n_log']):.2f}  "
            f"base={true_params['base']:.0f}"
        )
    if title is not None:
        ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=110)
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Hill model class + AutoFit Analysis classes
# ---------------------------------------------------------------------------


class Hill:
    """AutoFit-friendly Hill model.

    Three parameters in the same convention as `hill_curve`:
    `log_ic50`, `n_log` (linear Hill coefficient is `exp(n_log)`), `base`.
    """

    def __init__(
        self,
        log_ic50: float = 1.0,
        n_log: float = 0.0,
        base: float = 35000.0,
    ):
        self.log_ic50 = log_ic50
        self.n_log = n_log
        self.base = base


class HillVisualizer(af.Visualizer):
    """Per-dataset diagnostic plotter wired into AutoFit's image path.

    AutoFit calls `visualize_before_fit` once before the local search and
    `visualize` after every search iteration. Both write into
    `paths.image_path`, which AutoFit places under each EP optimisation
    folder so refits across EP iterations accumulate as separate
    sub-folders. With `true_params` set on the analysis (sim case), the
    true Hill curve is overlaid; otherwise only data + best-fit are
    plotted (real-data case).
    """

    @staticmethod
    def visualize_before_fit(analysis, paths, model):
        x = np.asarray(analysis.x)
        y = np.asarray(analysis.y)
        os.makedirs(paths.image_path, exist_ok=True)
        plot_dataset(
            x,
            y,
            analysis.noise_sigma,
            title=(
                "Before fit"
                + (
                    f"  |  true log_ic50={analysis.true_params['log_ic50']:.2f}"
                    if analysis.true_params is not None
                    else ""
                )
            ),
            true_params=analysis.true_params,
            output_path=os.path.join(paths.image_path, "hill_curve_data.png"),
        )

    @staticmethod
    def visualize(analysis, paths, instance, during_analysis):
        x = np.asarray(analysis.x)
        y = np.asarray(analysis.y)
        fit = {
            "log_ic50": float(instance.hill.log_ic50),
            "n_log": float(instance.hill.n_log),
            "base": float(instance.hill.base),
        }
        os.makedirs(paths.image_path, exist_ok=True)
        kind = "During" if during_analysis else "After"
        title = (
            f"{kind} EP  |  fit log_ic50={fit['log_ic50']:.2f}"
        )
        if analysis.true_params is not None:
            title += f"  (true {analysis.true_params['log_ic50']:.2f})"
        plot_dataset(
            x,
            y,
            analysis.noise_sigma,
            title=title,
            true_params=analysis.true_params,
            fit_params=fit,
            output_path=os.path.join(paths.image_path, "hill_curve_fit.png"),
        )

    @staticmethod
    def visualize_combined(analyses, paths, instance, during_analysis, **kwargs):
        # Shim around an AutoFit interface mismatch where graphical / joint
        # update cycles pass extra kwargs the base Visualizer rejects. The
        # per-dataset plots above already cover what we need.
        pass


class HillAnalysis(af.Analysis):
    """Single-dataset Hill log-likelihood (Gaussian, scalar noise sigma).

    Inputs are stored as JAX arrays so the JIT'd inner function can run
    without retracing. `log_likelihood_function` returns a Python float
    so AutoFit handles it exactly like the numpy path.

    `true_params` is optional and is consumed only by `HillVisualizer`
    (purely informational — never read by `log_likelihood_function`).
    Pass it for simulator runs to overlay the true Hill curve on every
    per-iteration plot; leave as `None` for real data.
    """

    Visualizer = HillVisualizer

    def __init__(self, x, y, noise_sigma, true_params=None):
        # use_jax=False: AutoFit calls this analysis directly. We still
        # use JAX inside the likelihood for the jit'd hot path; the flag
        # only switches AutoFit's outer JIT/grad layer (which we don't
        # want here because dynesty doesn't need gradients).
        super().__init__(use_jax=False)
        self.x = jnp.asarray(x, dtype=float)
        self.y = jnp.asarray(y, dtype=float)
        self.noise_sigma = float(noise_sigma)
        self.true_params = true_params

    def log_likelihood_function(self, instance, xp=np):
        return float(
            _hill_log_likelihood_jit(
                self.x,
                self.y,
                self.noise_sigma,
                instance.hill.log_ic50,
                instance.hill.n_log,
                instance.hill.base,
            )
        )


class GlobalLinearAnalysis(af.Analysis):
    """EP-message comparison log-likelihood for the global linear model.

    The log-likelihood is

        log L = -0.5 * sum_i ‖ (latent_i @ coef_matrix + coef_mean - μ_i) / σ_i ‖²

    where `μ_i = instance.hill_coef[i, *]` and `σ_i = self._ep_sigmas[i, *]`.
    `_ep_sigmas` is populated by `FixedHillCoefEPFactor.set_model_approx`
    each EP iteration; on iteration zero we fall back to the broad
    `fallback_sigmas` constructed below so the global search still runs.
    """

    def __init__(self, latents, fallback_sigmas):
        # See HillAnalysis: use_jax=False because the JIT lives inside
        # the likelihood, not at the AutoFit-outer level.
        super().__init__(use_jax=False)
        self.latents = jnp.asarray(latents, dtype=float)
        self.fallback_sigmas = jnp.asarray(fallback_sigmas, dtype=float)

    def log_likelihood_function(self, instance, xp=np):
        ep_means = jnp.asarray(instance.hill_coef, dtype=float)
        ep_sigmas = getattr(self, "_ep_sigmas", self.fallback_sigmas)
        coef_matrix = jnp.asarray(instance.coef_matrix, dtype=float)
        coef_mean = jnp.asarray(instance.coef_mean, dtype=float)
        return float(
            _global_log_likelihood_jit(
                self.latents, coef_matrix, coef_mean, ep_means, ep_sigmas
            )
        )


class GraphicalLinearAnalysis(af.Analysis):
    """Graphical-fit twin of `GlobalLinearAnalysis` (no EP message passing).

    The log-likelihood is

        log L = -0.5 * sum_i ‖ (latent_i @ coef_matrix + coef_mean - hill_coef_i)
                              / regression_sigmas_i ‖²

    where `hill_coef` is read directly from the sampled instance — it is a
    **free** parameter in the graphical fit (shared via `build_model_linear`'s
    Prior wiring with each per-dataset Hill model), not an EP-frozen
    constant. The constraint scale `regression_sigmas` is fixed up-front
    (the EP version derives equivalent sigmas from the local factors'
    message posteriors each iteration, which has no analogue in a single
    global search).
    """

    def __init__(self, latents, regression_sigmas):
        # See HillAnalysis: use_jax=False because the JIT lives inside the
        # likelihood, not at the AutoFit-outer level.
        super().__init__(use_jax=False)
        self.latents = jnp.asarray(latents, dtype=float)
        n_datasets = int(self.latents.shape[0])
        sigmas = jnp.asarray(regression_sigmas, dtype=float)
        if sigmas.ndim == 1:
            sigmas = jnp.tile(sigmas, (n_datasets, 1))
        self.regression_sigmas = sigmas

    def log_likelihood_function(self, instance, xp=np):
        hill_coef = jnp.asarray(instance.hill_coef, dtype=float)
        coef_matrix = jnp.asarray(instance.coef_matrix, dtype=float)
        coef_mean = jnp.asarray(instance.coef_mean, dtype=float)
        return float(
            _global_log_likelihood_jit(
                self.latents,
                coef_matrix,
                coef_mean,
                hill_coef,
                self.regression_sigmas,
            )
        )


class FixedHillCoefEPFactor(af.EPAnalysisFactor):
    """Specialised EPAnalysisFactor for the global linear factor.

    The global `model_linear` Collection includes a `hill_coef
    (n_datasets, 3)` array whose elements share `Prior` instances with
    the per-dataset Hill factors. Mathematically `hill_coef` is **not**
    a free parameter of the global fit — it is fixed at each local
    factor's posterior mean and used as "data" for the EP message
    comparison.

    Each EP iteration this hook reads each local factor's posterior
    mean from `model_approx.factor_mean_field`, freezes the
    corresponding `hill_coef[i, j]` element at that mean (a constant),
    and stashes the local sigma on the wrapped Analysis at
    `_ep_sigmas` so the likelihood can read it.
    """

    def __init__(
        self,
        prior_model,
        analysis,
        hill_coef_priors,
        local_factor_names,
        optimiser=None,
        name=None,
    ):
        super().__init__(
            prior_model=prior_model,
            analysis=analysis,
            optimiser=optimiser,
            name=name,
        )
        self._hill_coef_priors = hill_coef_priors
        self._local_factor_names = local_factor_names

    def set_model_approx(self, model_approx):
        super().set_model_approx(model_approx)
        n_datasets = len(self._local_factor_names)
        ep_sigmas = np.empty((n_datasets, 3), dtype=np.float64)

        factor_by_name = {
            f.name: f
            for f in model_approx.factor_mean_field
            if getattr(f, "name", None)
        }

        for i, local_name in enumerate(self._local_factor_names):
            local_factor = factor_by_name[local_name]
            local_mf = model_approx.factor_mean_field[local_factor]
            for j, prior in enumerate(self._hill_coef_priors[i]):
                msg = local_mf[prior]
                self.prior_model.hill_coef[i, j] = float(msg.mean)
                ep_sigmas[i, j] = float(msg.sigma)

        self.analysis._ep_sigmas = jnp.asarray(ep_sigmas)
        print(
            f"  [{self.name}] post-freeze prior_count="
            f"{self.prior_model.prior_count}"
        )


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset_list(sample_path, n_datasets=None, want_truth=False):
    """Load per-dataset arrays from a workspace sample folder.

    Parameters
    ----------
    sample_path : str or Path
        Folder containing `dataset_<i>/` sub-directories.
    n_datasets : int, optional
        Number of datasets to load (sorted by index). ``None`` loads all.
    want_truth : bool
        If True, also load per-dataset `true_params.json` and the
        sample-level `_sample/` ground-truth arrays.

    Returns
    -------
    dict
        Keys: ``x_array``, ``y_array``, ``latent_array``, ``info_list``,
        ``noise_sigma``, ``n_latent``, ``dataset_dirs``. Plus, when
        ``want_truth`` is True, ``true_params_list``, ``coef_mean_true``,
        ``coef_matrix_true``, ``hill_params_true``, ``latent_array_true``.
    """
    sample_path = Path(sample_path)
    dataset_dirs = sorted(
        [
            d
            for d in sample_path.iterdir()
            if d.is_dir() and d.name.startswith("dataset_")
        ],
        key=lambda d: int(d.name.split("_")[1]),
    )
    if n_datasets is not None:
        dataset_dirs = dataset_dirs[:n_datasets]

    x_array, y_array, latent_array, info_list = [], [], [], []
    for d in dataset_dirs:
        x_array.append(np.load(d / "x.npy"))
        y_array.append(np.load(d / "y.npy"))
        latent_array.append(np.load(d / "latent.npy"))
        with open(d / "info.json") as f:
            info_list.append(json.load(f))

    noise_sigma = float(info_list[0]["noise_sigma"])
    n_latent = int(info_list[0]["n_latent"])

    out = dict(
        x_array=x_array,
        y_array=y_array,
        latent_array=latent_array,
        info_list=info_list,
        noise_sigma=noise_sigma,
        n_latent=n_latent,
        dataset_dirs=dataset_dirs,
    )

    if want_truth:
        true_params_list = []
        for d in dataset_dirs:
            with open(d / "true_params.json") as f:
                true_params_list.append(json.load(f))
        sample_meta = sample_path / "_sample"
        out["true_params_list"] = true_params_list
        out["coef_mean_true"] = np.load(sample_meta / "coef_mean_true.npy")
        out["coef_matrix_true"] = np.load(sample_meta / "coef_matrix_true.npy")
        out["hill_params_true"] = np.load(sample_meta / "hill_params_true.npy")
        out["latent_array_true"] = np.load(sample_meta / "latent_array.npy")

    return out


# ---------------------------------------------------------------------------
# Model wiring
# ---------------------------------------------------------------------------


def build_model_linear(
    *,
    n_latent,
    n_datasets,
    coef_mean_priors,
    coef_matrix_prior_sigmas,
    hill_priors_per_dataset,
):
    """Build the global linear `Collection` and return it plus the per-
    dataset hill_coef priors that link it to the local Hill models.
    """
    model_linear = af.Collection(
        coef_matrix=af.Array(
            shape=(n_latent, 3), prior=af.GaussianPrior(mean=0.0, sigma=0.5)
        ),
        coef_mean=af.Array(
            shape=(3,), prior=af.GaussianPrior(mean=0.0, sigma=1.0)
        ),
        hill_coef=af.Array(
            shape=(n_datasets, 3), prior=af.GaussianPrior(mean=0.0, sigma=1.0)
        ),
    )
    for j, p in enumerate(coef_mean_priors):
        model_linear.coef_mean[j] = p
    for k in range(n_latent):
        for j in range(3):
            model_linear.coef_matrix[k, j] = af.GaussianPrior(
                mean=0.0, sigma=coef_matrix_prior_sigmas[j]
            )

    hill_coef_priors = []
    for i in range(n_datasets):
        log_ic50_p, n_log_p, base_p = hill_priors_per_dataset[i]
        model_linear.hill_coef[i, 0] = log_ic50_p
        model_linear.hill_coef[i, 1] = n_log_p
        model_linear.hill_coef[i, 2] = base_p
        hill_coef_priors.append([log_ic50_p, n_log_p, base_p])

    return model_linear, hill_coef_priors


def build_per_dataset_models(hill_coef_priors):
    """Return a list of `Collection(hill=Hill(...))` linked to global priors."""
    model_list = []
    for prior_row in hill_coef_priors:
        log_ic50_p, n_log_p, base_p = prior_row
        hill = af.Model(Hill)
        hill.log_ic50 = log_ic50_p
        hill.n_log = n_log_p
        hill.base = base_p
        model_list.append(af.Collection(hill=hill))
    return model_list


# ---------------------------------------------------------------------------
# EP run + result extraction
# ---------------------------------------------------------------------------


def run_ep_fit(
    *,
    name,
    n_datasets,
    n_latent,
    x_array,
    y_array,
    latent_array,
    noise_sigma,
    coef_mean_priors,
    coef_matrix_prior_sigmas,
    hill_priors_per_dataset,
    nlive=50,
    max_steps=5,
    true_params_list=None,
):
    """Build the factor graph and call `factor_graph.optimise`.

    `true_params_list` is optional and only used by the per-dataset
    `HillVisualizer` to overlay the true Hill curve on each diagnostic
    plot. Pass it for simulator runs; leave None for real data.

    Returns a dict with `ep_result`, the recovered means/sigmas for
    `hill_coef`, `coef_mean`, `coef_matrix`, plus the constructed
    `model_linear` and `hill_coef_priors` for later introspection.
    """
    model_linear, hill_coef_priors = build_model_linear(
        n_latent=n_latent,
        n_datasets=n_datasets,
        coef_mean_priors=coef_mean_priors,
        coef_matrix_prior_sigmas=coef_matrix_prior_sigmas,
        hill_priors_per_dataset=hill_priors_per_dataset,
    )
    model_list = build_per_dataset_models(hill_coef_priors)

    analysis_list = [
        HillAnalysis(
            x=x_array[i],
            y=y_array[i],
            noise_sigma=noise_sigma,
            true_params=(
                true_params_list[i] if true_params_list is not None else None
            ),
        )
        for i in range(n_datasets)
    ]

    fallback_sigmas = np.tile(np.array([1.0, 0.5, 10000.0]), (n_datasets, 1))
    analysis_global = GlobalLinearAnalysis(
        latents=np.asarray(latent_array, dtype=float),
        fallback_sigmas=fallback_sigmas,
    )

    paths = af.DirectoryPaths(path_prefix=Path(f"ep_{name}"), name="ep")

    search_local = af.DynestyStatic(
        paths=paths, nlive=nlive, sample="rwalk", force_x1_cpu=True
    )
    search_global = af.DynestyStatic(
        paths=paths, nlive=nlive, sample="rwalk", force_x1_cpu=True
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

    print(
        f"\nGlobal model_linear prior_count BEFORE EP starts: "
        f"{model_linear.prior_count}"
    )
    print(
        f"  (= n_latent*3 + 3 + n_datasets*3 = "
        f"{n_latent * 3} + 3 + {n_datasets * 3} = "
        f"{n_latent * 3 + 3 + n_datasets * 3})"
    )
    print(
        f"  After first set_model_approx, this should drop to "
        f"{n_latent * 3 + 3} (coef_matrix + coef_mean only)."
    )
    print(
        f"\nRunning EP: max_steps={max_steps}, nlive={nlive}, "
        f"n_datasets={n_datasets}"
    )

    ep_result = factor_graph.optimise(
        laplace,
        paths=paths,
        ep_history=af.EPHistory(kl_tol=1.0),
        max_steps=max_steps,
    )

    fmf = ep_result.updated_ep_mean_field.factor_mean_field
    mf = ep_result.updated_ep_mean_field.mean_field
    factor_by_name = {
        f.name: f for f in fmf if getattr(f, "name", None)
    }

    hill_means = np.empty((n_datasets, 3))
    hill_sigmas = np.empty((n_datasets, 3))
    for i, prior_row in enumerate(hill_coef_priors):
        ds_factor = factor_by_name[f"dataset_{i}"]
        ds_mf = fmf[ds_factor]
        for j, prior in enumerate(prior_row):
            msg = ds_mf[prior]
            hill_means[i, j] = float(msg.mean)
            hill_sigmas[i, j] = float(msg.sigma)

    coef_mean_means = np.array(
        [float(mf[model_linear.coef_mean[j]].mean) for j in range(3)]
    )
    coef_mean_sigmas = np.array(
        [float(mf[model_linear.coef_mean[j]].sigma) for j in range(3)]
    )

    coef_matrix_means = np.empty((n_latent, 3))
    coef_matrix_sigmas = np.empty((n_latent, 3))
    for k in range(n_latent):
        for j in range(3):
            msg = mf[model_linear.coef_matrix[k, j]]
            coef_matrix_means[k, j] = float(msg.mean)
            coef_matrix_sigmas[k, j] = float(msg.sigma)

    return dict(
        ep_result=ep_result,
        hill_means=hill_means,
        hill_sigmas=hill_sigmas,
        coef_mean_means=coef_mean_means,
        coef_mean_sigmas=coef_mean_sigmas,
        coef_matrix_means=coef_matrix_means,
        coef_matrix_sigmas=coef_matrix_sigmas,
        model_linear=model_linear,
        hill_coef_priors=hill_coef_priors,
    )


# ---------------------------------------------------------------------------
# Graphical-model run (single non-linear search over the full factor graph)
# ---------------------------------------------------------------------------


DEFAULT_REGRESSION_SIGMAS = (0.5, 0.5, 6000.0)


def run_graphical_fit(
    *,
    name,
    n_datasets,
    n_latent,
    x_array,
    y_array,
    latent_array,
    noise_sigma,
    coef_mean_priors,
    coef_matrix_prior_sigmas,
    hill_priors_per_dataset,
    nlive=50,
    regression_sigmas=DEFAULT_REGRESSION_SIGMAS,
    true_params_list=None,
):
    """Build the factor graph and fit it in a single non-linear search.

    The model wiring is identical to ``run_ep_fit``: the same
    ``build_model_linear`` / ``build_per_dataset_models`` helpers wire
    `model_linear.hill_coef[i, j]` to each per-dataset `hill[i].log_ic50`
    / `.n_log` / `.base` via shared ``Prior`` instances. In a graphical
    fit those shared priors resolve to a *single* free variable per
    Hill parameter, so the search jointly samples
    ``n_datasets × 3 + n_latent × 3 + 3`` parameters — every per-dataset
    Hill coefficient, the global linear matrix, and the global linear
    offset — in one parameter space.

    The global linear factor is a Gaussian probabilistic constraint with
    fixed ``regression_sigmas`` (default ``[0.5, 0.5, 6000.0]``, matching
    ``coef_matrix_prior_sigmas`` from the EP sim run). The EP path
    derives equivalent sigmas from each iteration's message posteriors;
    a graphical fit has no such loop, so the scale is set up-front.

    Returns the same schema as ``run_ep_fit`` (``hill_means``,
    ``hill_sigmas``, ``coef_mean_means``, ``coef_mean_sigmas``,
    ``coef_matrix_means``, ``coef_matrix_sigmas``, ``model_linear``,
    ``hill_coef_priors``) plus the underlying ``result`` for callers
    that want to introspect samples directly.
    """
    model_linear, hill_coef_priors = build_model_linear(
        n_latent=n_latent,
        n_datasets=n_datasets,
        coef_mean_priors=coef_mean_priors,
        coef_matrix_prior_sigmas=coef_matrix_prior_sigmas,
        hill_priors_per_dataset=hill_priors_per_dataset,
    )
    model_list = build_per_dataset_models(hill_coef_priors)

    analysis_list = [
        HillAnalysis(
            x=x_array[i],
            y=y_array[i],
            noise_sigma=noise_sigma,
            true_params=(
                true_params_list[i] if true_params_list is not None else None
            ),
        )
        for i in range(n_datasets)
    ]

    analysis_global = GraphicalLinearAnalysis(
        latents=np.asarray(latent_array, dtype=float),
        regression_sigmas=np.asarray(regression_sigmas, dtype=float),
    )

    paths = af.DirectoryPaths(
        path_prefix=Path(f"graphical_{name}"), name="graphical"
    )

    analysis_factor_list = [
        af.AnalysisFactor(
            prior_model=model,
            analysis=analysis,
            name=f"dataset_{i}",
        )
        for i, (model, analysis) in enumerate(zip(model_list, analysis_list))
    ]
    analysis_factor_global = af.AnalysisFactor(
        prior_model=model_linear,
        analysis=analysis_global,
        name="global",
    )

    factor_graph = af.FactorGraphModel(
        *analysis_factor_list, analysis_factor_global
    )

    search = af.DynestyStatic(
        paths=paths, nlive=nlive, sample="rwalk", force_x1_cpu=True
    )

    print(
        f"\nGlobal factor graph free parameter count: "
        f"{factor_graph.global_prior_model.prior_count}"
    )
    print(
        f"  (= n_datasets*3 + n_latent*3 + 3 = "
        f"{n_datasets * 3} + {n_latent * 3} + 3 = "
        f"{n_datasets * 3 + n_latent * 3 + 3})"
    )
    print(f"\nRunning graphical fit: nlive={nlive}, n_datasets={n_datasets}")

    result = search.fit(
        model=factor_graph.global_prior_model, analysis=factor_graph
    )

    samples = result.samples
    median = samples.median_pdf()
    upper = samples.values_at_upper_sigma(sigma=1.0)
    lower = samples.values_at_lower_sigma(sigma=1.0)

    hill_means = np.empty((n_datasets, 3))
    hill_sigmas = np.empty((n_datasets, 3))
    for i in range(n_datasets):
        hill_i = median[i].hill
        hill_u = upper[i].hill
        hill_l = lower[i].hill
        for j, attr in enumerate(HILL_PARAM_NAMES):
            hill_means[i, j] = float(getattr(hill_i, attr))
            hill_sigmas[i, j] = (
                float(getattr(hill_u, attr)) - float(getattr(hill_l, attr))
            ) / 2.0

    global_median = median[n_datasets]
    global_upper = upper[n_datasets]
    global_lower = lower[n_datasets]

    coef_mean_means = np.asarray(global_median.coef_mean, dtype=float).reshape(3)
    coef_mean_sigmas = (
        np.asarray(global_upper.coef_mean, dtype=float).reshape(3)
        - np.asarray(global_lower.coef_mean, dtype=float).reshape(3)
    ) / 2.0

    coef_matrix_means = np.asarray(
        global_median.coef_matrix, dtype=float
    ).reshape(n_latent, 3)
    coef_matrix_sigmas = (
        np.asarray(global_upper.coef_matrix, dtype=float).reshape(n_latent, 3)
        - np.asarray(global_lower.coef_matrix, dtype=float).reshape(n_latent, 3)
    ) / 2.0

    return dict(
        result=result,
        hill_means=hill_means,
        hill_sigmas=hill_sigmas,
        coef_mean_means=coef_mean_means,
        coef_mean_sigmas=coef_mean_sigmas,
        coef_matrix_means=coef_matrix_means,
        coef_matrix_sigmas=coef_matrix_sigmas,
        model_linear=model_linear,
        hill_coef_priors=hill_coef_priors,
    )


# ---------------------------------------------------------------------------
# Summary writers
# ---------------------------------------------------------------------------


HILL_PARAM_NAMES = ["log_ic50", "n_log", "base"]


def _sigma_dist(rec, true, sig):
    return abs(float(rec) - float(true)) / max(float(sig), 1e-12)


def write_ep_summary(
    *,
    name,
    n_datasets,
    n_latent,
    nlive,
    max_steps,
    wall_time_s,
    recovered,
    output_dir,
    truth=None,
    test_mode=False,
    method="ep",
):
    """Write `<method>_<name>_summary.txt` and `<method>_<name>_summary.json`.

    Parameters
    ----------
    recovered : dict
        Output of ``run_ep_fit`` (`hill_means`, `hill_sigmas`,
        `coef_mean_means`, `coef_mean_sigmas`, `coef_matrix_means`,
        `coef_matrix_sigmas`). The graphical-fit path returns the same
        schema so the same writer covers both.
    truth : dict, optional
        For sim runs: ``coef_mean_true``, ``coef_matrix_true``,
        ``hill_params_true``. Triggers per-element σ-distance columns
        and a `failures` list. None for real-data runs.
    test_mode : bool
        If True, marks the summary as a test-mode (non-converged) run
        and skips raising on failure.
    method : str, optional
        Either ``"ep"`` (default) or ``"graphical"``. Controls the
        output filenames (`<method>_<name>_summary.{txt,json}`),
        the summary header, and the JSON ``"method"`` field. The
        ``max_steps`` value is irrelevant for the graphical fit but is
        still recorded for parity with the EP summaries.

    Returns
    -------
    (txt_path, json_path, failures)
        Paths of the two written files plus a list of >3σ failure tuples
        (empty if `truth` is None or test_mode is True).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hill_means = recovered["hill_means"]
    hill_sigmas = recovered["hill_sigmas"]
    coef_mean_means = recovered["coef_mean_means"]
    coef_mean_sigmas = recovered["coef_mean_sigmas"]
    coef_matrix_means = recovered["coef_matrix_means"]
    coef_matrix_sigmas = recovered["coef_matrix_sigmas"]

    failures_hill = []
    failures_coef_mean = []
    failures_coef_matrix = []

    method_title = "EP" if method == "ep" else method.capitalize()

    lines = []
    lines.append("=" * 100)
    lines.append(f"IC50 {method_title} Summary ({name})")
    if test_mode:
        lines.append("[PYAUTO_TEST_MODE active — sampler short-circuited; values are not converged.]")
    lines.append("=" * 100)
    lines.append(f"Datasets:    {n_datasets}")
    lines.append(f"n_latent:    {n_latent}")
    lines.append(f"nlive:       {nlive}")
    lines.append(f"max_steps:   {max_steps}")
    lines.append(
        f"Wall time:   {wall_time_s:.1f} s ({wall_time_s / 60:.1f} min)"
    )
    lines.append("")

    if truth is not None and not test_mode:
        # Pass/fail header
        n_hill = n_datasets * 3
        n_matrix = n_latent * 3
        hill_warns = sum(
            1
            for i in range(n_datasets)
            for j in range(3)
            if _sigma_dist(
                hill_means[i, j],
                truth["hill_params_true"][i, j],
                hill_sigmas[i, j],
            )
            > 3.0
        )
        coef_mean_warns = sum(
            1
            for j in range(3)
            if _sigma_dist(
                coef_mean_means[j], truth["coef_mean_true"][j], coef_mean_sigmas[j]
            )
            > 3.0
        )
        coef_matrix_warns = sum(
            1
            for k in range(n_latent)
            for j in range(3)
            if _sigma_dist(
                coef_matrix_means[k, j],
                truth["coef_matrix_true"][k, j],
                coef_matrix_sigmas[k, j],
            )
            > 3.0
        )
        lines.append("--- Pass/fail at 3σ ---")
        lines.append(
            f"hill_coef:    {n_hill - hill_warns}/{n_hill} within 3σ        "
            f"(3 params × {n_datasets} datasets)"
        )
        lines.append(
            f"coef_mean:    {3 - coef_mean_warns}/3 within 3σ          "
            "(population-mean Hill params)"
        )
        lines.append(
            f"coef_matrix:  {n_matrix - coef_matrix_warns}/{n_matrix} within 3σ        "
            f"(n_latent × 3 linear coefficients)"
        )
        lines.append("")

    # Per-dataset hill_coef
    lines.append("--- Per-dataset hill_coef ---")
    if truth is not None:
        lines.append(
            f"{'dataset':<10}  {'param':<10}  {'true':>11}  "
            f"{'mean':>11}  {'σ':>11}  {'σ-dist':>7}  flag"
        )
    else:
        lines.append(
            f"{'dataset':<10}  {'param':<10}  {'mean':>11}  {'σ':>11}"
        )
    lines.append("-" * 100)
    for i in range(n_datasets):
        for j, label in enumerate(HILL_PARAM_NAMES):
            rec = hill_means[i, j]
            sig = hill_sigmas[i, j]
            if truth is not None:
                true = truth["hill_params_true"][i, j]
                sd = _sigma_dist(rec, true, sig)
                flag = "OK" if sd <= 3.0 else f"WARN {sd:.1f}σ"
                lines.append(
                    f"dataset_{i:<3} {label:<10}  {true:>11.4g}  "
                    f"{rec:>11.4g}  {sig:>11.4g}  {sd:>7.2f}  {flag}"
                )
                if sd > 3.0 and not test_mode:
                    failures_hill.append((f"dataset_{i}", label, true, rec, sig, sd))
            else:
                lines.append(
                    f"dataset_{i:<3} {label:<10}  {rec:>11.4g}  {sig:>11.4g}"
                )
        lines.append("-" * 100)
    lines.append("")

    # Global coef_mean
    lines.append("--- Global coef_mean ---")
    if truth is not None:
        lines.append(
            f"{'channel':<12}  {'true':>11}  {'mean':>11}  "
            f"{'σ':>11}  {'σ-dist':>7}  flag"
        )
    else:
        lines.append(f"{'channel':<12}  {'mean':>11}  {'σ':>11}")
    lines.append("-" * 100)
    for j, label in enumerate(HILL_PARAM_NAMES):
        rec = coef_mean_means[j]
        sig = coef_mean_sigmas[j]
        if truth is not None:
            true = truth["coef_mean_true"][j]
            sd = _sigma_dist(rec, true, sig)
            flag = "OK" if sd <= 3.0 else f"WARN {sd:.1f}σ"
            lines.append(
                f"{label:<12}  {true:>11.4g}  {rec:>11.4g}  "
                f"{sig:>11.4g}  {sd:>7.2f}  {flag}"
            )
            if sd > 3.0 and not test_mode:
                failures_coef_mean.append((label, true, rec, sig, sd))
        else:
            lines.append(f"{label:<12}  {rec:>11.4g}  {sig:>11.4g}")
    lines.append("")

    # Global coef_matrix
    lines.append(f"--- Global coef_matrix ({n_latent * 3} elements) ---")
    if truth is not None:
        lines.append(
            f"{'latent':<7}  {'param':<10}  {'true':>11}  "
            f"{'mean':>11}  {'σ':>11}  {'σ-dist':>7}  flag"
        )
    else:
        lines.append(
            f"{'latent':<7}  {'param':<10}  {'mean':>11}  {'σ':>11}"
        )
    lines.append("-" * 100)
    for k in range(n_latent):
        for j, label in enumerate(HILL_PARAM_NAMES):
            rec = coef_matrix_means[k, j]
            sig = coef_matrix_sigmas[k, j]
            if truth is not None:
                true = truth["coef_matrix_true"][k, j]
                sd = _sigma_dist(rec, true, sig)
                flag = "OK" if sd <= 3.0 else f"WARN {sd:.1f}σ"
                lines.append(
                    f"{k:<7}  {label:<10}  {true:>11.4g}  "
                    f"{rec:>11.4g}  {sig:>11.4g}  {sd:>7.2f}  {flag}"
                )
                if sd > 3.0 and not test_mode:
                    failures_coef_matrix.append(
                        (k, label, true, rec, sig, sd)
                    )
            else:
                lines.append(
                    f"{k:<7}  {label:<10}  {rec:>11.4g}  {sig:>11.4g}"
                )
        lines.append("-" * 100)
    lines.append("")

    if truth is not None and not test_mode:
        lines.append("--- Failures (>3σ) ---")
        if not (failures_hill or failures_coef_mean or failures_coef_matrix):
            lines.append(
                "(none — all parameters recovered within 3σ of simulator truth)"
            )
        else:
            for w in failures_hill:
                lines.append(
                    f"hill_coef    {w[0]}.{w[1]}: true={w[2]:.4g} "
                    f"fit={w[3]:.4g} σ-dist={w[5]:.2f}"
                )
            for f in failures_coef_mean:
                lines.append(
                    f"coef_mean    {f[0]}: true={f[1]:.4g} fit={f[2]:.4g} "
                    f"σ={f[3]:.4g} σ-dist={f[4]:.2f}"
                )
            for c in failures_coef_matrix:
                lines.append(
                    f"coef_matrix  latent_{c[0]}.{c[1]}: true={c[2]:.4g} "
                    f"fit={c[3]:.4g} σ={c[4]:.4g} σ-dist={c[5]:.2f}"
                )

    summary = "\n".join(lines) + "\n"

    txt_path = output_dir / f"{method}_{name}_summary.txt"
    with open(txt_path, "w") as f:
        f.write(summary)
    print(f"\nSummary written to: {txt_path}")

    sidecar = {
        "method": method,
        "name": name,
        "n_datasets": n_datasets,
        "n_latent": n_latent,
        "nlive": nlive,
        "max_steps": max_steps,
        "wall_time_s": wall_time_s,
        "test_mode": test_mode,
        "hill_means": hill_means.tolist(),
        "hill_sigmas": hill_sigmas.tolist(),
        "coef_mean_means": coef_mean_means.tolist(),
        "coef_mean_sigmas": coef_mean_sigmas.tolist(),
        "coef_matrix_means": coef_matrix_means.tolist(),
        "coef_matrix_sigmas": coef_matrix_sigmas.tolist(),
    }
    if truth is not None:
        sidecar["coef_mean_true"] = truth["coef_mean_true"].tolist()
        sidecar["coef_matrix_true"] = truth["coef_matrix_true"].tolist()
        sidecar["hill_params_true"] = truth["hill_params_true"].tolist()

    json_path = output_dir / f"{method}_{name}_summary.json"
    with open(json_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"Sidecar JSON written to: {json_path}")

    failures = (failures_hill, failures_coef_mean, failures_coef_matrix)
    return txt_path, json_path, failures


def write_graphical_summary(
    *,
    name,
    n_datasets,
    n_latent,
    nlive,
    wall_time_s,
    recovered,
    output_dir,
    truth=None,
    test_mode=False,
):
    """Graphical-fit shim around ``write_ep_summary``.

    Mirrors the EP writer's interface but with no ``max_steps`` (the
    graphical fit is a single non-linear search, not an EP loop) and
    pins ``method="graphical"`` so the output files become
    ``graphical_<name>_summary.{txt,json}``.
    """
    return write_ep_summary(
        name=name,
        n_datasets=n_datasets,
        n_latent=n_latent,
        nlive=nlive,
        max_steps=0,
        wall_time_s=wall_time_s,
        recovered=recovered,
        output_dir=output_dir,
        truth=truth,
        test_mode=test_mode,
        method="graphical",
    )


def is_test_mode():
    """Return True if PYAUTO_TEST_MODE is set non-zero."""
    try:
        return bool(int(os.environ.get("PYAUTO_TEST_MODE", "0")))
    except ValueError:
        return False
