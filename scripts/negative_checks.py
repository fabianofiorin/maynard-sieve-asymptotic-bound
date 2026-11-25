"""Short numerical confirmations of negative / structural results (A2), (A3)."""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from flint import arb

import os as _os, sys as _sys
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _SCRIPT_DIR not in _sys.path:
    _sys.path.insert(0, _SCRIPT_DIR)

from shifted_analysis import compute_U_alpha_arb, solve_arb_midpoint

EULER_GAMMA = np.euler_gamma


def make_macro_u_grid(k: int, n_gl: int = 24, w_max: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """Build log-spaced u = k * w quadrature grid covering [0, k * w_max] through the u ~ 1 boundary layer."""
    # Breakpoints in macro variable w = u/k: cover 0 to 1/k to w_max cleanly
    pts_w = [0.0, 0.2 / k, 1.0 / k, 4.0 / k, 20.0 / k, 100.0 / k, 500.0 / k]
    pts_w = [w for w in pts_w if w < 0.05] + [0.05, 0.2, 0.8, 2.5, 8.0, 22.0, w_max]
    pts_w = sorted(set(pts_w))
    ut, uw = leggauss(n_gl)
    ws = np.concatenate([0.5 * (b - a) * ut + 0.5 * (a + b) for a, b in zip(pts_w, pts_w[1:])])
    ww = np.concatenate([0.5 * (b - a) * uw for a, b in zip(pts_w, pts_w[1:])])
    return k * ws, k * ww


def check_A2_microscale() -> None:
    """Check (A2) kappa formula & finite-k prediction for g(u) = 1/(1+u) + 1/(1+u)^2."""
    print("=== (A2) Microscale Optimality Check: g(u) = h(u) + h(u)^2 ===")
    h = lambda u: 1.0 / (1.0 + u)
    g = lambda u: h(u) + h(u) ** 2

    m0, _ = quad(lambda u: g(u) ** 2, 0.0, np.inf)
    g_lam = lambda u: g(u / m0) / m0

    beta_g = 1.0
    beta_lam = beta_g - np.log(m0)

    diff_int, _ = quad(lambda u: u * (g_lam(u) ** 2 - h(u) ** 2), 0.0, np.inf, limit=200)
    c1_lam = EULER_GAMMA - diff_int
    kappa_minus_gamma = 2.0 * beta_lam + c1_lam - EULER_GAMMA

    sq_int, _ = quad(lambda u: -(1.0 + u) * ((g_lam(u) - h(u)) ** 2), 0.0, np.inf, limit=200)
    print(f"  m0(g) = {m0:.8f}, beta_lam = {beta_lam:.8f}, c1_lam - gamma = {c1_lam - EULER_GAMMA:.8f}")
    print(f"  kappa - gamma = {kappa_minus_gamma:.10f}")
    print(f"  -int(1+u)(g_lam - h)^2 = {sq_int:.10f}, diff = {abs(kappa_minus_gamma - sq_int):.2e}")

    alpha_p = c1_lam - EULER_GAMMA
    panels = [0, 1, 3, 7, 15, 35, 70, 150]
    U = compute_U_alpha_arb(1, alpha_p, panels, work_dps=50)
    eta0_ap = solve_arb_midpoint(U, 0, arb(repr(alpha_p)), dps=50)
    pred_limit = (kappa_minus_gamma) + (eta0_ap - 0.0)  # wait: check exact identity!
    # In fact: g_lam has int_0^{k x} g_lam(u) du = L + log x + beta_lam + o(1),
    # and k log(W(z)/W(sig)) = i tau (c1_lam - L) + z log z + o(1) = i tau (gamma + alpha' - L) + z log z,
    # where alpha' = c1_lam - gamma!
    # So the shift in the density is alpha' = c1_lam - gamma, giving eta_deg0(alpha') + alpha' for 2<1, A1>
    # plus 2 * beta_lam from the inner integral constant beta_lam!
    pred_limit_exact = 2.0 * beta_lam + (eta0_ap + alpha_p)
    print(f"  Predicted Q=1 limit: 2*beta_lam + eta_deg0(alpha') + alpha' = {pred_limit_exact:.8f}")
    print(f"  Notice (kappa - gamma) + baseline_deg0 = {kappa_minus_gamma - 0.33652278:.8f}")

    x_t, x_w = leggauss(48)
    pts_x = [0.0, 1e-4, 1e-2, 0.1, 0.5, 2.0, 6.0, 15.0, 35.0]
    xx = np.concatenate([0.5 * (b - a) * x_t + 0.5 * (a + b) for a, b in zip(pts_x, pts_x[1:])])
    wx = np.concatenate([0.5 * (b - a) * x_w for a, b in zip(pts_x, pts_x[1:])])

    for k in (1000, 10000, 100000, 1000000):
        uu, ww = make_macro_u_grid(k, n_gl=24, w_max=60.0)
        gl_u = g_lam(uu)
        L = np.log(k)
        sig = L
        a = k * L
        T = 40.0
        dt = 0.04
        tau = (np.arange(int(T / dt)) + 0.5) * dt
        tau_full = np.concatenate([-tau[::-1], tau])
        s = sig * (1.0 + 1j * tau_full)
        Wa = np.sum(ww[None, :] * (gl_u[None, :] ** 2) * np.exp(-(s[:, None] / a) * uu[None, :]), axis=1)
        Wa_sig = float(np.sum(ww * (gl_u ** 2) * np.exp(-(sig / a) * uu)))
        log_ratio = np.log(Wa / Wa_sig)
        fI = np.exp(k * log_ratio)
        fJ = np.exp((k - 1) * log_ratio)
        K_vals = (np.exp(1j * sig * tau_full) - np.exp(-sig)) / (1.0 + 1j * tau_full) / sig
        I_norm = np.real(np.sum(K_vals * fI)) * (sig * dt / (2.0 * np.pi))

        U_grid = k * xx
        inner = np.log(1.0 + U_grid / m0) + 1.0 - 1.0 / (1.0 + U_grid / m0)
        Va2 = (1.0 / sig) * np.sum(wx[None, :] * (inner[None, :] ** 2) * np.exp(-(1.0 + 1j * tau_full[:, None]) * xx[None, :]), axis=1)
        J_norm = np.real(np.sum(np.exp(1j * sig * tau_full) * fJ * Va2)) * (sig * dt / (2.0 * np.pi))
        Mk = (J_norm / I_norm) / (L * Wa_sig)
        print(f"  k={k:7d}: M_k(g_lam) - log k = {Mk - L:.6f} (limit pred = {pred_limit_exact:.6f})")


def check_A3_power_tail() -> None:
    """Check (A3) p = 0.8: at the stable scale a = k^{1/(2p-1)}, M_k = O(1), so M_k - log k -> -inf."""
    print("\n=== (A3) Power-law tail g(u) = (1 + u)^(-0.8), Q=1, a = k^(5/3), sigma=1 ===")
    p = 0.8
    x_t, x_w = leggauss(48)
    pts_x = [0.0, 1e-4, 1e-2, 0.1, 0.5, 2.0, 6.0, 15.0, 40.0]
    xx = np.concatenate([0.5 * (b - a) * x_t + 0.5 * (a + b) for a, b in zip(pts_x, pts_x[1:])])
    wx = np.concatenate([0.5 * (b - a) * x_w for a, b in zip(pts_x, pts_x[1:])])

    for k in (1000, 10000, 100000, 1000000):
        L = np.log(k)
        sig = 1.0
        a = k ** (1.0 / (2.0 * p - 1.0))
        # Macro variable w = u / a (so s*u/a = s*w):
        pts_w = [0.0, 1.0 / a, 10.0 / a, 100.0 / a, 1e-3, 0.02, 0.2, 1.0, 5.0, 25.0, 150.0]
        pts_w = sorted(set(w for w in pts_w if w >= 0.0))
        ut, uw = leggauss(24)
        ww_pts = np.concatenate([0.5 * (b - a0) * ut + 0.5 * (a0 + b) for a0, b in zip(pts_w, pts_w[1:])])
        ww_wt = np.concatenate([0.5 * (b - a0) * uw for a0, b in zip(pts_w, pts_w[1:])])
        uu = a * ww_pts
        gp = (1.0 + uu) ** (-p)

        T = 40.0
        dt = 0.04
        tau = (np.arange(int(T / dt)) + 0.5) * dt
        tau_full = np.concatenate([-tau[::-1], tau])
        s = sig * (1.0 + 1j * tau_full)
        Wa = a * np.sum(ww_wt[None, :] * (gp[None, :] ** 2) * np.exp(-s[:, None] * ww_pts[None, :]), axis=1)
        Wa_sig = float(a * np.sum(ww_wt * (gp ** 2) * np.exp(-sig * ww_pts)))
        log_ratio = np.log(Wa / Wa_sig)
        fI = np.exp(k * log_ratio)
        fJ = np.exp((k - 1) * log_ratio)
        K_vals = (np.exp(1j * sig * tau_full) - np.exp(-sig)) / (1.0 + 1j * tau_full) / sig
        I_norm = np.real(np.sum(K_vals * fI)) * (sig * dt / (2.0 * np.pi))

        U_grid = a * (xx / sig)
        inner = (((1.0 + U_grid) ** (1.0 - p)) - 1.0) / (1.0 - p)
        Va2 = (1.0 / sig) * np.sum(wx[None, :] * (inner[None, :] ** 2) * np.exp(-(1.0 + 1j * tau_full[:, None]) * xx[None, :]), axis=1)
        J_norm = np.real(np.sum(np.exp(1j * sig * tau_full) * fJ * Va2)) * (sig * dt / (2.0 * np.pi))
        Mk = k * (J_norm / I_norm) / (a * a * Wa_sig / a)  # i.e. k / (a * Wa_sig) * (J_norm / I_norm)
        print(f"  k={k:7d}: M_k = {Mk:.5f}, M_k - log k = {Mk - L:.5f}")


if __name__ == "__main__":
    check_A2_microscale()
    check_A3_power_tail()
