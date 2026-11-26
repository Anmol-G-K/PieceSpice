"""Compare two solver implementations with a paired experimental design.

A timing comparison of two fast programs is easy to get wrong: the difference
you measure is often the machine's state, not the code's.  This script uses the
standard controls for that, and applies them to the numerical question as well:

**Timing**
  * warm-up before any measurement (JIT, caches, page faults);
  * **paired interleaving** A,B,A,B... so a drift in CPU state hits both arms;
  * median of the per-pair ratio, not the ratio of the medians;
  * bootstrap 95 % confidence interval on that ratio;
  * Wilcoxon signed-rank test on the paired differences (non-parametric, no
    normality assumption, which timing data never satisfies).

**Numerics**
  * equivalence is a tolerance question, not a statistical one: report the
    maximum relative difference over every trace.

Usage::

    uv run python scripts/compare_kernels.py            # reference vs compiled
    uv run python scripts/compare_kernels.py --kernels   # solve kernels incl. JAX
"""

from __future__ import annotations

import argparse
import statistics
import time

import numba as nb
import numpy as np
from scipy import stats

from engine.ir.parse import parse_circuit
from engine.solver.transient import solve_transient

REPEATS = 8
BOOTSTRAP = 4000
SEED = 20261126


# ---------------------------------------------------------------------------
# Standard paired comparison
# ---------------------------------------------------------------------------

def paired_ratio(a: list[float], b: list[float]) -> dict:
    """Compare two paired samples of run times.

    ``ratio = median(a_i / b_i)`` -- a paired ratio, so the machine's state
    cancels between the two members of each pair instead of being attributed to
    whichever arm happened to run while the CPU was boosting.
    """
    ratios = [x / y for x, y in zip(a, b)]
    rng = np.random.default_rng(SEED)
    n = len(ratios)
    boot = np.array([
        statistics.median(rng.choice(ratios, size=n, replace=True))
        for _ in range(BOOTSTRAP)])
    # Wilcoxon needs a non-zero difference to be meaningful.
    try:
        p_value = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        p_value = float("nan")
    return {
        "median_a_s": statistics.median(a),
        "median_b_s": statistics.median(b),
        "paired_ratio_median": statistics.median(ratios),
        "ratio_ci95": (float(np.percentile(boot, 2.5)),
                       float(np.percentile(boot, 97.5))),
        "wilcoxon_p": p_value,
        "n_pairs": n,
    }


def equivalence(reference, other) -> dict:
    """Maximum relative difference, both arms sampled onto the reference grid.

    The two engines choose different step grids, so comparing sample i to
    sample i would measure the grids, not the answer.  Interpolation onto the
    reference timeline is what makes this an equivalence test.
    """
    t_ref = reference.get(reference.sweep_var)
    t_other = other.get(other.sweep_var)
    # Compare on **common time points only**.  Two fixed-step runs can take a
    # different number of shortened steps, and interpolating across a
    # switching edge then reports volts of difference that are really a
    # misalignment, not a disagreement.
    common = np.intersect1d(t_ref, t_other, assume_unique=False)
    if common.size < 3:
        return {"max_rel_diff": float("nan"), "worst_trace": "no common points",
                "max_abs_diff": float("nan"), "circuit_scale": float("nan"),
                "common_points": int(common.size)}
    idx_ref = np.searchsorted(t_ref, common)
    idx_oth = np.searchsorted(t_other, common)
    shared = [k for k in sorted(set(reference.traces) & set(other.traces))
              if k != reference.sweep_var]
    # One circuit-wide scale.  Normalising per trace is wrong here: a body-diode
    # current sits at 1e-15 A, so any tiny difference reads as 100 % (the trap
    # the repository's own metrics notes warn about).
    scale = max(max(float(np.max(np.abs(reference.get(k)))) for k in shared),
                1e-12)
    worst = 0.0
    worst_key = ""
    worst_abs = 0.0
    for key in shared:
        a = reference.get(key)[idx_ref]
        b = other.get(key)[idx_oth]
        error = float(np.max(np.abs(a - b)))
        if error > worst_abs:
            worst_abs = error
        if error / scale > worst:
            worst, worst_key = error / scale, key
    return {"max_rel_diff": worst, "worst_trace": worst_key,
            "max_abs_diff": worst_abs, "circuit_scale": scale,
            "common_points": int(common.size)}


# ---------------------------------------------------------------------------
# Kernels compared at the same job
# ---------------------------------------------------------------------------

@nb.njit(cache=True)
def _numba_solve(a, b, x, work):
    """Dense LU with partial pivoting, as the solver uses it."""
    n = a.shape[0]
    for i in range(n):
        for j in range(n):
            work[i, j] = a[i, j]
        work[i, n] = b[i]
    for c in range(n):
        pivot = c
        best = abs(work[c, c])
        for r in range(c + 1, n):
            if abs(work[r, c]) > best:
                best = abs(work[r, c])
                pivot = r
        if pivot != c:
            for j in range(n + 1):
                work[c, j], work[pivot, j] = work[pivot, j], work[c, j]
        piv = work[c, c]
        for r in range(c + 1, n):
            f = work[r, c] / piv
            for j in range(c, n + 1):
                work[r, j] -= f * work[c, j]
    for i in range(n - 1, -1, -1):
        acc = work[i, n]
        for j in range(i + 1, n):
            acc -= work[i, j] * x[j]
        x[i] = acc / work[i, i]
    return x


@nb.njit(cache=True)
def _numba_solve_loop(a, b, x, work, steps):
    for _ in range(steps):
        _numba_solve(a, b, x, work)
    return x


def bench_kernels(steps: int = 7000, n: int = 9) -> None:
    """Same job -- ``steps`` dense solves of an ``n x n`` system -- five ways."""
    rng = np.random.default_rng(SEED)
    a = rng.normal(size=(n, n))
    a = a @ a.T + n * np.eye(n)
    b = rng.normal(size=n)
    x = np.zeros(n)
    work = np.zeros((n, n + 1))

    rows = {}

    def timed(fn, repeats: int = 5) -> float:
        fn()
        best = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            best.append(time.perf_counter() - t0)
        return statistics.median(best)

    rows["numpy (python loop)"] = timed(
        lambda: [np.linalg.solve(a, b) for _ in range(steps)]) / steps

    _numba_solve_loop(a, b, x, work, 1)
    rows["numba in-place"] = timed(
        lambda: _numba_solve_loop(a, b, x, work, steps)) / steps

    try:
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        aj = jnp.asarray(a)
        bj = jnp.asarray(b)

        @jax.jit
        def jax_batch(a_, b_):
            return jax.vmap(lambda bi: jnp.linalg.solve(a_, bi))(b_)

        @jax.jit
        def jax_scan(a_, b_):
            # The right-hand side must depend on the carry.  With a
            # loop-invariant solve XLA hoists it out of the loop and the
            # benchmark times an empty scan -- the classic JAX trap.
            def body(carry, _):
                out = jnp.linalg.solve(a_, b_ + 1e-3 * carry)
                return out, out[0]

            _, trace = jax.lax.scan(body, jnp.zeros(n, dtype=b_.dtype), None,
                                    length=steps)
            return trace

        batch = jnp.tile(bj, (steps, 1))
        jax_batch(aj, batch).block_until_ready()
        jax_scan(aj, bj).block_until_ready()
        rows["jax vmap batched"] = timed(
            lambda: jax_batch(aj, batch).block_until_ready()) / steps
        rows["jax scan sequential"] = timed(
            lambda: jax_scan(aj, bj).block_until_ready()) / steps
    except Exception as exc:                     # noqa: BLE001 - reported
        rows[f"jax unavailable: {type(exc).__name__}"] = float("nan")

    print(f"\n{steps} dense solves of a {n}x{n} system (median of 5):")
    base = rows.get("numba in-place")
    for name, per_step in rows.items():
        if per_step != per_step:
            print(f"  {name:26s}    n/a")
            continue
        ratio = base / per_step if base else float("nan")
        print(f"  {name:26s} {per_step * 1e6:8.2f} us/solve   "
              f"{ratio:5.1f}x vs numba")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernels", action="store_true",
                        help="also compare the linear-solve kernels")
    parser.add_argument("--vendor", default="epc2216")
    args = parser.parse_args(argv)

    from pipeline.fitting import build_pwl_spec
    from engine.rom import SolverRom
    from engine.rom.bench import DptConditions, dpt

    rom = SolverRom.from_spec(build_pwl_spec(args.vendor))
    circuit = dpt(rom, DptConditions())

    # Warm-up is part of the method: the compiled arm pays JIT compilation on
    # its first call, and letting that land inside the measurement would make
    # the comparison measure the compiler, not the kernel.
    for kwargs in (dict(engine="reference", max_step=2e-9),
                   dict(engine="compiled", adaptive=True)):
        solve_transient(circuit, **kwargs)

    runtimes = {"reference": [], "compiled": []}
    traces = {}
    for _ in range(REPEATS):
        for arm, kwargs in (("reference", dict(engine="reference",
                                               max_step=2e-9)),
                            ("compiled", dict(engine="compiled", adaptive=True))):
            t0 = time.perf_counter()
            result = solve_transient(circuit, **kwargs)
            runtimes[arm].append(time.perf_counter() - t0)
            traces[arm] = result

    stats_out = paired_ratio(runtimes["reference"], runtimes["compiled"])
    # Implementation equivalence needs the *same* step policy; comparing a
    # fixed-step arm against an adaptive one measures the integrator, not the
    # implementation.
    fixed = solve_transient(circuit, engine="compiled", adaptive=False,
                            max_step=2e-9)
    same = equivalence(traces["reference"], fixed)
    different_policy = equivalence(traces["reference"], traces["compiled"])
    print(f"\n{args.vendor}, {REPEATS} interleaved pairs, same circuit:")
    print(f"  reference  median {stats_out['median_a_s'] * 1e3:8.2f} ms")
    print(f"  compiled   median {stats_out['median_b_s'] * 1e3:8.2f} ms")
    low, high = stats_out["ratio_ci95"]
    print(f"  paired speedup {stats_out['paired_ratio_median']:5.2f}x "
          f"(95% CI {low:.2f}-{high:.2f}), Wilcoxon p={stats_out['wilcoxon_p']:.4f}")
    print(f"  equivalence (same step policy): max rel diff "
          f"{same['max_rel_diff']:.2e} on {same['worst_trace']} "
          f"over {same['common_points']} common time points")
    print(f"  discretisation (fixed vs adaptive): max rel diff "
          f"{different_policy['max_rel_diff']:.2e} on "
          f"{different_policy['worst_trace']} "
          f"(absolute {different_policy['max_abs_diff']:.2e} V)")

    if args.kernels:
        bench_kernels()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
