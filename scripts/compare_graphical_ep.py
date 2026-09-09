"""
Cross-Method Comparison: Graphical vs EP
========================================

Reads the JSON sidecars written by ``graphical_sim.py`` / ``graphical_real.py``
and ``ep_sim.py`` / ``ep_real.py`` and writes a side-by-side comparison to
``results/graphical_ep_comparison_<name>.{txt,json}``.

This is the trust-building deliverable for the EP campaign: EP is only worth
scaling to 10 000+ datasets if it demonstrably matches the graphical joint fit
at small N.

Per parameter the comparison shows simulator truth (when the run is simulated),
the graphical estimate (mean +/- sigma), the EP estimate (mean +/- sigma) and
three signals:

  * sigma-distance from truth, graphical
  * sigma-distance from truth, EP
  * cross-method disagreement, |g - e| / sqrt(sigma_g^2 + sigma_e^2)

The third is the one the per-method 3-sigma checks cannot give you. Two methods
can each sit within 3 sigma of truth and still disagree with each other well
beyond their combined error bars -- which means at least one of them has an
understated error. A cross-method disagreement above 1.0 is flagged DISAGREE.

Ported from concr/scripts/cancer_sim/compare_graphical_ep.py and rewritten for
this workspace's sidecar schema (concr writes `<method>_summary_n<N>.json` with
a different layout; here it is `<method>_<name>_summary.json`). No Hill maths is
involved, so the sign-convention hazard that applies to the rest of concr's
cancer code does not apply to this file.

Usage:
    python3 scripts/compare_graphical_ep.py [--name sim] [--disagree_threshold 1.0]
"""

import argparse
import json
import math
import sys
from pathlib import Path

try:
    here = Path(__file__).resolve().parent
except NameError:
    here = Path.cwd()

workspace_root = here.parent
results_dir = workspace_root / "results"

PARAMS = ["log_ic50", "n_log", "base"]


def load_sidecar(method, name):
    path = results_dir / f"{method}_{name}_summary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"missing sidecar: {path}\n"
            f"Run scripts/{method}_{name}.py first."
        )
    with open(path) as f:
        return json.load(f)


def sigma_dist(rec, true, sigma):
    return abs(rec - true) / max(abs(sigma), 1e-12)


def cross_dist(g_rec, g_sig, e_rec, e_sig):
    return abs(g_rec - e_rec) / max(math.sqrt(g_sig**2 + e_sig**2), 1e-12)


def rows_for_block(block, g, e):
    """Yield (label, true, g_mean, g_sig, e_mean, e_sig) for one parameter block."""
    if block == "coef_mean":
        for j, pname in enumerate(PARAMS):
            yield (
                f"coef_mean[{pname}]",
                g["coef_mean_true"][j],
                g["coef_mean_means"][j], g["coef_mean_sigmas"][j],
                e["coef_mean_means"][j], e["coef_mean_sigmas"][j],
            )
    elif block == "coef_matrix":
        for i in range(g["n_latent"]):
            for j, pname in enumerate(PARAMS):
                yield (
                    f"coef_matrix[{i}][{pname}]",
                    g["coef_matrix_true"][i][j],
                    g["coef_matrix_means"][i][j], g["coef_matrix_sigmas"][i][j],
                    e["coef_matrix_means"][i][j], e["coef_matrix_sigmas"][i][j],
                )
    elif block == "hill_coef":
        for i in range(g["n_datasets"]):
            for j, pname in enumerate(PARAMS):
                yield (
                    f"dataset_{i}[{pname}]",
                    g["hill_params_true"][i][j],
                    g["hill_means"][i][j], g["hill_sigmas"][i][j],
                    e["hill_means"][i][j], e["hill_sigmas"][i][j],
                )
    else:
        raise ValueError(f"unknown block: {block}")


def main():
    parser = argparse.ArgumentParser(description="Compare graphical vs EP sidecars.")
    parser.add_argument("--name", default="sim", help="Run name (default: %(default)s).")
    parser.add_argument(
        "--disagree_threshold", type=float, default=1.0,
        help="Cross-method sigma threshold above which a row is flagged (default: %(default)s).",
    )
    args = parser.parse_args()

    g = load_sidecar("graphical", args.name)
    e = load_sidecar("ep", args.name)

    for field in ("n_datasets", "n_latent"):
        if g[field] != e[field]:
            raise ValueError(
                f"sidecar mismatch on {field}: graphical={g[field]} vs ep={e[field]}. "
                f"The two runs did not fit the same problem; comparison is meaningless."
            )

    lines = []
    out = lines.append
    out("=" * 108)
    out(f"Graphical vs EP cross-method comparison  ({args.name})")
    out("=" * 108)
    out(f"Datasets:  {g['n_datasets']}      n_latent: {g['n_latent']}")
    out(f"graphical: nlive={g['nlive']}  wall={g['wall_time_s']:.1f}s  test_mode={g['test_mode']}")
    out(f"EP:        nlive={e['nlive']}  max_steps={e.get('max_steps')}  "
        f"wall={e['wall_time_s']:.1f}s  test_mode={e['test_mode']}")
    if g["test_mode"] or e["test_mode"]:
        out("")
        out("!! One or both runs are TEST MODE — sampling was short-circuited.")
        out("!! These numbers are a plumbing check, not a result.")
    out("")

    totals = {}
    summary_rows = []

    for block in ("coef_mean", "coef_matrix", "hill_coef"):
        out("-" * 108)
        out(f"--- {block} ---")
        out(
            f"{'parameter':<26}{'true':>12}{'graphical':>13}{'sig_g':>11}"
            f"{'EP':>13}{'sig_e':>11}{'d_g':>7}{'d_e':>7}{'cross':>8}  flag"
        )
        out("-" * 108)
        n_g_fail = n_e_fail = n_disagree = n_rows = 0
        for label, true, gm, gs, em, es in rows_for_block(block, g, e):
            dg = sigma_dist(gm, true, gs)
            de = sigma_dist(em, true, es)
            cx = cross_dist(gm, gs, em, es)
            flags = []
            if dg > 3.0:
                flags.append("G>3sig")
                n_g_fail += 1
            if de > 3.0:
                flags.append("E>3sig")
                n_e_fail += 1
            if cx > args.disagree_threshold:
                flags.append("DISAGREE")
                n_disagree += 1
            n_rows += 1
            out(
                f"{label:<26}{true:>12.4g}{gm:>13.4g}{gs:>11.4g}"
                f"{em:>13.4g}{es:>11.4g}{dg:>7.2f}{de:>7.2f}{cx:>8.2f}  "
                f"{' '.join(flags) if flags else 'OK'}"
            )
            summary_rows.append({
                "block": block, "parameter": label, "true": true,
                "graphical_mean": gm, "graphical_sigma": gs,
                "ep_mean": em, "ep_sigma": es,
                "sigma_dist_graphical": dg, "sigma_dist_ep": de,
                "cross_disagreement": cx,
            })
        totals[block] = {
            "rows": n_rows,
            "graphical_within_3sigma": n_rows - n_g_fail,
            "ep_within_3sigma": n_rows - n_e_fail,
            "disagreements": n_disagree,
        }
        out("")

    out("=" * 108)
    out("VERDICT")
    out("=" * 108)
    grand = {"rows": 0, "graphical_within_3sigma": 0, "ep_within_3sigma": 0, "disagreements": 0}
    for block, t in totals.items():
        for k in grand:
            grand[k] += t[k]
        out(
            f"  {block:<14} graphical {t['graphical_within_3sigma']}/{t['rows']} within 3sig   "
            f"EP {t['ep_within_3sigma']}/{t['rows']} within 3sig   "
            f"cross-method disagreements: {t['disagreements']}/{t['rows']}"
        )
    out("")
    out(
        f"  TOTAL          graphical {grand['graphical_within_3sigma']}/{grand['rows']}   "
        f"EP {grand['ep_within_3sigma']}/{grand['rows']}   "
        f"disagreements {grand['disagreements']}/{grand['rows']}"
    )
    out("")
    if grand["disagreements"] == 0:
        out("  EP and the graphical joint fit agree within their combined error bars"
            " on every parameter.")
    else:
        out(f"  {grand['disagreements']} parameter(s) disagree beyond "
            f"{args.disagree_threshold} sigma of the combined error bar. At least one"
            " method's error is understated on those rows — investigate before"
            " trusting EP at larger N.")

    # A median sigma per block, so a 'pass' produced by enormous error bars is
    # visible rather than hidden. A wide sigma makes the 3-sigma test vacuous.
    out("")
    out("  Median recovered sigma (a large value makes the 3-sigma test weak):")
    for block in totals:
        gs_all = sorted(r["graphical_sigma"] for r in summary_rows if r["block"] == block)
        es_all = sorted(r["ep_sigma"] for r in summary_rows if r["block"] == block)
        med = lambda v: v[len(v) // 2] if v else float("nan")
        out(f"    {block:<14} graphical {med(gs_all):>12.4g}   EP {med(es_all):>12.4g}")

    text = "\n".join(lines)
    print(text)

    results_dir.mkdir(parents=True, exist_ok=True)
    txt_path = results_dir / f"graphical_ep_comparison_{args.name}.txt"
    json_path = results_dir / f"graphical_ep_comparison_{args.name}.json"
    txt_path.write_text(text + "\n")
    with open(json_path, "w") as f:
        json.dump(
            {
                "name": args.name,
                "n_datasets": g["n_datasets"],
                "n_latent": g["n_latent"],
                "disagree_threshold": args.disagree_threshold,
                "graphical_run": {k: g[k] for k in ("nlive", "wall_time_s", "test_mode")},
                "ep_run": {k: e[k] for k in ("nlive", "max_steps", "wall_time_s", "test_mode")},
                "totals": totals,
                "grand_total": grand,
                "rows": summary_rows,
            },
            f, indent=2,
        )
    print(f"\nWrote {txt_path}")
    print(f"Wrote {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
