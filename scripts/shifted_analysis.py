"""Shifted asymptotic family analysis: closed-form Arb cross-check & alpha optimization."""

import math
import time
import numpy as np
import mpmath as mp
from scipy.optimize import minimize_scalar
from flint import acb, arb, arb_mat, ctx

import os as _os, sys as _sys
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _SCRIPT_DIR not in _sys.path:
    _sys.path.insert(0, _SCRIPT_DIR)

from limit_xspace import XSpaceEvaluator, eval_nu


def compute_U_alpha_arb(max_n: int, alpha_val: float | str, panels: list[float], work_dps: int = 80) -> list[arb]:
    """Compute U_n(alpha) = (n!/pi) Re int_0^inf exp(z*(gamma + alpha + log z))/z^{n+1} dtau."""
    ctx.dps = work_dps
    g = arb.const_euler()
    alpha = arb(str(alpha_val))
    pi = arb.pi()
    T = arb(panels[-1])
    tol = arb(10) ** (-(work_dps - 25))
    shift = g + alpha
    U = []
    for n in range(max_n + 1):
        def f(t, analytic, n=n):
            z = 1 + acb(0, 1) * t
            return (z * (shift + z.log(analytic=analytic))).exp() / (z ** (n + 1))

        total = acb(0)
        for a, b in zip(panels, panels[1:]):
            total += acb.integral(f, arb(a), arb(b), rel_tol=tol, abs_tol=tol)
        assert total.is_finite(), f"Arb integration failed at n={n}, alpha={alpha_val}"
        tail_mag = (2 * (shift + 1).exp() * math.factorial(n) * (-pi * T / 2).exp() / (pi ** 2 * T ** n)).upper()
        tail_ball = arb(0).union(tail_mag).union(-tail_mag)
        U.append(total.real * math.factorial(n) / pi + tail_ball)
    return U


def build_shifted_forms(U: list[arb], deg: int, alpha: arb) -> tuple[arb_mat, arb_mat]:
    """Build shifted I(alpha) and net D(alpha) matrices for degree <= deg."""
    H = [sum((arb(1) / j for j in range(1, n + 1)), arb(0)) for n in range(2 * deg + 2)]
    I_mat = arb_mat([[U[i + j] for j in range(deg + 1)] for i in range(deg + 1)])
    D_mat = arb_mat([
        [(2 * H[i + j] + 2 + alpha - H[i] - H[j]) * U[i + j] - 2 * U[i + j + 1] for j in range(deg + 1)]
        for i in range(deg + 1)
    ])
    return I_mat, D_mat


def solve_arb_midpoint(U: list[arb], deg: int, alpha: arb, dps: int = 80) -> float:
    """Solve generalized eigenvalue problem at high precision from Arb moment midpoints."""
    I_mat, D_mat = build_shifted_forms(U, deg, alpha)
    with mp.workdps(dps):
        cv = lambda x: mp.mpf(x.mid().str(dps - 10, radius=False))
        A = mp.matrix([[cv(I_mat[i, j]) for j in range(deg + 1)] for i in range(deg + 1)])
        B = mp.matrix([[cv(D_mat[i, j]) for j in range(deg + 1)] for i in range(deg + 1)])
        Linv = mp.cholesky(A) ** -1
        ev, _ = mp.eigsy(Linv * B * Linv.T)
        return float(max(ev))


def cross_check_shifted() -> None:
    """Compare Arb closed-form eta(alpha) against XSpaceEvaluator at alpha in {-0.5, -0.337, 0.0, 0.3}."""
    print("=== Cross-checking Arb closed-form vs x-space evaluator ===")
    ev = XSpaceEvaluator(max_deg=20, X=8.0, n_x=32, n_w=48)
    panels = [0, 1, 3, 7, 15, 35, 70, 140, 250]
    for a_val in (-0.5, -0.337, 0.0, 0.3):
        U = compute_U_alpha_arb(33, a_val, panels, work_dps=65)
        alpha_arb = arb(str(a_val))
        nu_a = eval_nu(ev.x, alpha=a_val)
        for d in (0, 4, 10, 16):
            eta_arb = solve_arb_midpoint(U, d, alpha_arb, dps=65)
            eta_x, _ = ev.solve_eta(d, alpha=a_val, nu_vals=nu_a)
            diff = abs(eta_arb - eta_x)
            print(f"  alpha={a_val:6.3f} deg={d:2d}: Arb={eta_arb:.9f}, x-space={eta_x:.9f}, diff={diff:.2e}")
            assert diff < 1e-6, f"Mismatch at alpha={a_val}, deg={d}: diff={diff}"
    print("PASS: Closed-form Arb matches x-space to < 1e-6 at all test points.\n")


def optimize_alpha_table() -> None:
    """Optimize alpha for deg = 0..24 using XSpaceEvaluator, and check deg 16 & 20 in Arb."""
    print("=== Optimizing alpha across degrees 0..24 ===")
    t0 = time.time()
    ev = XSpaceEvaluator(max_deg=24, X=8.0, n_x=32, n_w=48)
    nu0 = eval_nu(ev.x, alpha=0.0)
    degs = [0, 1, 2, 3, 4, 5, 8, 10, 12, 16, 20, 24]
    results = []
    for d in degs:
        eta_base, _ = ev.solve_eta(d, alpha=0.0, nu_vals=nu0)
        res = minimize_scalar(
            lambda a: -ev.solve_eta(d, alpha=a)[0],
            bounds=(-0.6, 0.1),
            method="bounded",
            options={"xatol": 5e-5},
        )
        a_opt = float(res.x)
        eta_opt = float(-res.fun)
        results.append((d, eta_base, a_opt, eta_opt, eta_opt - eta_base))
        print(
            f"  deg {d:2d}: base={eta_base:.8f}, alpha*={a_opt:.6f}, "
            f"eta*={eta_opt:.8f}, gain={eta_opt - eta_base:+.8f}"
        )

    # Confirm deg 16, 20, 24 in Arb at alpha* and print degree extrapolation
    panels = [0, 1, 3, 7, 15, 35, 70, 140, 260]
    a_ref = round(results[-1][2], 4)
    U_hi = compute_U_alpha_arb(49, a_ref, panels, work_dps=150)
    for d, _, a_opt, eta_opt, _ in results:
        if d in (16, 20, 24):
            eta_arb = solve_arb_midpoint(U_hi, d, arb(str(a_ref)), dps=150)
            print(f"  Arb check deg {d:2d} at alpha={a_ref:.4f}: eta_arb={eta_arb:.10f} (x-space at opt={eta_opt:.10f})")
    # Simple Shanks / power-law extrapolation from deg 16, 20, 24
    c16, c20, c24 = [-r[3] for r in results if r[0] in (16, 20, 24)]
    d1, d2 = c16 - c20, c20 - c24
    r_est = d2 / d1
    c_inf_shanks = c24 - d2 * r_est / (1 - r_est)
    print(f"  Extrapolated c* (deg 16,20,24 Shanks): {c_inf_shanks:.6f} (c20={c20:.6f}, c24={c24:.6f})")
    print(f"Done in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    cross_check_shifted()
    optimize_alpha_table()
