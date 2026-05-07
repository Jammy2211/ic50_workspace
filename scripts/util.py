"""
IC50 Workspace — Shared Hill-Curve and Plotting Utilities
=========================================================

This module is the single source of truth for IC50 dose-response math and
plotting in `ic50_workspace`. Both simulated and real datasets pass through
the same `hill_curve`, `log_likelihood`, and `plot_dataset` functions, so
sims and real data never visually drift.

Sign convention
---------------
This module uses the canonical dose-response sign convention,

    y = base / (1 + exp(n * (x - log_ic50))),    n = exp(n_log)

which is **monotonically decreasing in x** (high viability at low dose, low
viability at high dose). This matches the visual convention used by the
group, exemplified by `concr/cancer_legacy/real/viz_hill.py`.

Caveat: concr's *production* fitting code (`concr/scripts/cancer/*.py`,
`concr/simulators/cancer_sim.py`) uses the **inverted** convention
`n*(log_ic50 - x)`, which gives a monotonically *increasing* curve. If
fits run by that production code are visualised here, the inferred
`log_ic50` may need a sign-aware reinterpretation. Track this whenever
results cross the sim/concr boundary. A separate caveat: concr's
`model_api.py:245-246` has a stale log-likelihood sign + missing-noise
bug; do not import or copy that formula.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def hill_curve(x, log_ic50, n_log, base):
    """Evaluate the Hill dose-response curve.

    Parameters
    ----------
    x : array_like
        Log-concentration grid (e.g. ``log(rank+1)``).
    log_ic50 : float
        Half-maximal log-concentration; the curve's inflection point.
    n_log : float
        Hill coefficient stored in log space; the actual slope is
        ``n = exp(n_log)``.
    base : float
        Upper plateau (intensity at low dose, where the cells are alive).

    Returns
    -------
    np.ndarray
        Curve values evaluated at ``x``. Decreasing in ``x``: ``y -> base``
        as ``x -> -inf`` and ``y -> 0`` as ``x -> +inf``.
    """
    x = np.asarray(x, dtype=float)
    n = np.exp(n_log)
    nkx = n * (x - log_ic50)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = base / (1.0 + np.exp(nkx))
    result = np.where(nkx > 500, 0.0, result)
    result = np.where(nkx < -500, base, result)
    return result


def log_likelihood(x, y, noise_sigma, log_ic50, n_log, base):
    """Gaussian log-likelihood of ``y`` given Hill parameters.

    Returns ``-0.5 * sum( ((y_pred - y) / noise_sigma)**2 )``, the standard
    normal-likelihood up to additive constants. Use this rather than
    re-deriving it locally so every consumer in ic50_workspace agrees on
    the sign and the noise normalisation.
    """
    y = np.asarray(y, dtype=float)
    y_pred = hill_curve(x, log_ic50, n_log, base)
    z = (y_pred - y) / noise_sigma
    return float(-0.5 * np.dot(z, z))


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

    Designed to render simulated and real data identically. Steel-blue
    scatter with capped errorbars; optional smooth Hill overlays when
    `true_params` (tomato) and/or `fit_params` (crimson) are supplied;
    optional dashed vertical line at the IC50 of whichever curve is
    overlaid.

    Parameters
    ----------
    x, y : array_like
        Log-concentration grid and observed intensities.
    noise_sigma : float
        Per-point noise std-dev (used as the y-errorbar).
    title : str, optional
        Plot title; if omitted a short auto-title is generated.
    true_params, fit_params : dict, optional
        Either may be a dict with keys ``"log_ic50"``, ``"n_log"``,
        ``"base"``. Both can be provided to overlay both curves.
    output_path : str or Path, optional
        If supplied, the figure is written via ``savefig(dpi=110)`` and
        closed. If omitted the figure is shown interactively (only useful
        in a notebook context, since this module forces the Agg backend).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x_min, x_max = float(x.min()), float(x.max())
    x_fine = np.linspace(x_min - 0.2, x_max + 0.2, 400)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        x, y, yerr=noise_sigma, fmt="o", color="steelblue", capsize=4, label="Data"
    )

    overlay_ic50 = None
    if true_params is not None:
        y_true = hill_curve(
            x_fine,
            true_params["log_ic50"],
            true_params["n_log"],
            true_params["base"],
        )
        ax.plot(x_fine, y_true, color="tomato", lw=2, label="True Hill curve")
        overlay_ic50 = ("tomato", true_params["log_ic50"], "True")

    if fit_params is not None:
        y_fit = hill_curve(
            x_fine,
            fit_params["log_ic50"],
            fit_params["n_log"],
            fit_params["base"],
        )
        ax.plot(x_fine, y_fit, color="crimson", lw=2, label="Best-fit Hill curve")
        overlay_ic50 = ("crimson", fit_params["log_ic50"], "Fit")

    if overlay_ic50 is not None:
        color, log_ic50_val, kind = overlay_ic50
        ax.axvline(
            log_ic50_val,
            color=color,
            linestyle="--",
            linewidth=1,
            label=f"{kind} log IC50 = {log_ic50_val:.2f}",
        )

    ax.set_xlabel("Log concentration rank (ln(rank+1))")
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
