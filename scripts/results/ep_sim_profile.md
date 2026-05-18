# EP Sim Profile — Timing Breakdown

Run config: n_datasets=5, n_latent=5, nlive=50, max_steps=3

**Total wall time: 65.87 s** (1.10 min)

## Top-level breakdown

| Phase | Time (s) | % of total |
|---|---:|---:|
| Setup (build model + analyses + factor graph) | 0.101 | 0.2 |
| Optimise (`factor_graph.optimise`) | 65.760 | 99.8 |
| Extract (walk ep_result → arrays) | 0.010 | 0.0 |

## Optimise-phase breakdown

| Category | Time (s) | % of optimise |
|---|---:|---:|
| (1) Local Hill fits — `HillAnalysis.log_likelihood_function` (5 factors × 3 iter) | 3.349 | 5.1 |
| (2) Global fit — `GlobalLinearAnalysis.log_likelihood_function` | 2.744 | 4.2 |
| (3a) `set_model_approx` prior-freeze hook | 0.002 | 0.0 |
| (3b) Dynesty wrapper overhead (search.fit minus LL evals) | 56.013 | 85.2 |
| (3c) EP-loop orchestration (optimise minus search.fit minus set_model_approx) | 3.651 | 5.6 |

Total Dynesty fits: 12 (worst case (N+1)×max_steps = 18; observed implies ~2.0 EP iterations ran before convergence)
Total search.fit wall time: 62.106 s — of which 9.8% was actual likelihood evaluation.
Per-Dynesty-fit wrapper overhead: 4.668 s/fit

**Definitions:**
- *Dynesty wrapper overhead* = time inside `search.fit(...)` not spent in `log_likelihood_function`. Covers sampler init, path/run setup, bound construction, weight/posterior post-processing, plot generation.
- *EP-loop orchestration* = time inside `factor_graph.optimise(...)` not spent in any `search.fit(...)` call. Covers factor traversal, EP message updates between iterations, the LaplaceOptimiser bookkeeping.

## Per-Hill-factor breakdown

| Factor | LL calls | Total LL time (s) | Time/call (ms) | Per iteration (s) |
|---|---:|---:|---:|---:|
| dataset_0 | 4133 | 0.681 | 0.1647 | 0.227 |
| dataset_1 | 4461 | 0.744 | 0.1667 | 0.248 |
| dataset_2 | 4465 | 0.451 | 0.1010 | 0.150 |
| dataset_3 | 4530 | 0.741 | 0.1636 | 0.247 |
| dataset_4 | 4315 | 0.733 | 0.1699 | 0.244 |

## Global factor breakdown

- Likelihood calls: 4935
- Total LL time: 2.744 s
- Time per call: 0.5560 ms
- Per iteration: 0.915 s
- `set_model_approx` calls: 2 (≈ 0.7 per EP iteration), total 0.0022 s

## Scaling projection

**Model components** (each per EP iteration, summed across `max_steps` iterations):

| Bucket | Per-fit cost | Scales as |
|---|---:|---|
| Dynesty wrapper overhead | 4.668 s/fit | (N + 1) per iteration |
| Local LL evaluation | 0.3349 s/dataset | N per iteration |
| Global LL evaluation | 1.372 s | constant per iteration |
| `set_model_approx` | 0.0011 s | constant per iteration (walks N×3 priors) |
| EP-loop orchestration | 1.826 s | assumed constant per iteration |
| Setup + extract | 0.111 s | one-off |

**Assumptions** (each a verifiable prediction once we re-measure at larger N):
1. Dynesty wrapper overhead is constant **per fit**. Likely true — paths/sampler init don't depend on `n_datasets`.
2. Local LL cost scales linearly in N (each Hill is independent).
3. Global LL cost is constant in N (`set_model_approx` freezes `hill_coef` to 18 free params).
4. `set_model_approx` walks N×3 priors per call — should grow linearly. Lumped as constant here because it's <0.01% of total at N=5.
5. EP-loop orchestration is constant per iteration. **Uncertain** — message updates may grow with `n_datasets`. Verify at N=100.
6. EP converges in `M = 2.0` iterations at every N. **Most uncertain** — convergence rate depends on data and tolerance; larger samples may need more iterations to satisfy `kl_tol`.

**Projection uses M = 2.0 EP iterations** (observed in this run before `kl_tol=1.0` convergence; `max_steps=3` is the cap). Tighten `kl_tol` if you want the projection to assume full max_steps iterations.

| n_datasets | M | Setup | Dynesty wrapper | Local LL | Global LL | sma | EP orch | **Total** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 2 | 0.1 s | 56.0 s | 3.3 s | 2.7 s | 0.0 s | 3.7 s | **1.1 min** |
| 100 | 2 | 0.1 s | 15.7 min | 1.1 min | 2.7 s | 0.0 s | 3.7 s | **16.9 min** |
| 1000 | 2 | 0.1 s | 2.6 h | 11.2 min | 2.7 s | 0.0 s | 3.7 s | **2.8 h** |
| 10000 | 2 | 0.1 s | 1.1 d | 1.9 h | 2.7 s | 0.0 s | 3.7 s | **1.2 d** |

## Caveats

- Single-run measurement; sampling variance not captured. Re-run a couple of times if any bucket looks borderline.
- `nlive=50` is small (`<= 2*ndim` per Dynesty's own warning for the global factor). Production runs would use higher nlive — both LL buckets scale roughly linearly in nlive, Dynesty wrapper overhead is mostly nlive-independent.
- The (3b) Dynesty-wrapper bucket lumps together every non-LL cost inside `search.fit(...)`. The PyAutoFit log shows `corner_anesthetic` plot attempts on every fit, plus `Removing search internal folder` cleanup — both are obvious candidates for a `--profile` mode that skips them. A `cProfile`/`py-spy` pass on the same workload would attribute the wrapper bucket to specific functions.

## Suggested follow-up optimisation targets

Pick the largest bucket from the projection table at the target N:

- **Dynesty wrapper dominates** (large per-fit overhead × `(N+1)·M` fits) → the biggest lever. Options: cache `paths` / sampler config so each per-factor fit doesn't re-initialise; share `DynestyStatic` instance reuse across iterations; suppress per-fit plot generation (`corner_anesthetic` log lines show this happens); skip `Removing search internal folder` cleanup on cached runs.
- **Local LL dominates** (only at large N) → parallelise the N independent local fits across CPU cores per iteration; or JAX-vmap the Hill likelihood across datasets within one fused fit.
- **Global LL dominates** → reduce global nlive (currently `nlive ≤ 2·ndim` so already at floor — try Laplace approximation for the global factor instead of full Dynesty).
- **`set_model_approx` dominates** (only at very large N) → cache the `prior → message` lookup; current impl walks the whole mean-field dict each call.
- **EP-loop orchestration dominates** → run `cProfile` / `py-spy` on the same workload and triage hot loops in `autofit/graphical/`.

