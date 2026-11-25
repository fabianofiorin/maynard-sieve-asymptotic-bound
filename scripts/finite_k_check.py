"""Step 4: Finite-k Laplace cross-checks for Baseline, Theorem A, Theorem B (P2), and Theorem C (Lambda_9)."""

from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import json
import math
import os
import sys
import time
from functools import lru_cache

import mpmath as mp
import numpy as np
from numpy.polynomial.legendre import leggauss
from flint import acb, acb_mat, arb, arb_mat, arb_poly, ctx, fmpq

import os as _os, sys as _sys
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _SCRIPT_DIR not in _sys.path:
    _sys.path.insert(0, _SCRIPT_DIR)

from limit_xspace import EULER_GAMMA, XSpaceEvaluator, eval_nu, build_orth_poly_vals
from shifted_analysis import compute_U_alpha_arb, build_shifted_forms


WITNESS_CACHE = os.path.join(os.path.dirname(__file__), "witness_cache.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "finite_k_results.json")


def _arb_frac(s: str) -> arb:
    """Parse decimal or 'p/q' string into an exact Arb ball."""
    fr = Fraction(s)
    return arb(fmpq(fr.numerator, fr.denominator))


def build_B_forms(U: list[arb], d0: int, d1: int, alpha: arb) -> tuple[arb_mat, arb_mat]:
    """Build block Gram I and net D matrices for F = [Q0(x) + Q1(x) * R2] prod (1+at_i)^{-1}."""
    max_n = max(2 * d0 + 1, d0 + d1 + 2, 2 * d1 + 4)
    H = [sum((arb(1) / j for j in range(1, m + 1)), arb(0)) for m in range(max_n + 5)]

    def fac_ratio(n: int, s: int) -> arb:
        fac = arb(1)
        for t in range(n + 1, n + s + 1):
            fac /= t
        return fac

    def L(n: int, s: int) -> arb:
        return fac_ratio(n, s) * ((H[n] + 1 + alpha) * U[n + s] - U[n + s + 1])

    N = d0 + 1 + d1 + 1
    I_mat = arb_mat(N, N)
    M_mat = arb_mat(N, N)
    o = d0 + 1
    for i in range(d0 + 1):
        for j in range(d0 + 1):
            n = i + j
            I_mat[i, j] = U[n]
            M_mat[i, j] = L(n, 0) - H[j] * U[n]
    for i in range(d0 + 1):
        for j in range(d1 + 1):
            n = i + j
            I_val = fac_ratio(n, 1) * U[n + 1]
            I_mat[i, o + j] = I_val
            I_mat[o + j, i] = I_val
            M_mat[i, o + j] = L(n, 1) - H[j] * I_val + U[n + 2] / ((j + 1) * (j + 2))
            M_mat[o + j, i] = L(n, 1) - H[i] * I_val
    for i in range(d1 + 1):
        for j in range(d1 + 1):
            n = i + j
            I_val = fac_ratio(n, 2) * U[n + 2] + 2 * fac_ratio(n, 3) * U[n + 3]
            I_mat[o + i, o + j] = I_val
            M_mat[o + i, o + j] = (L(n, 2) + 2 * L(n, 3)) - H[j] * I_val + U[n + 3] / ((j + 1) * (j + 2) * (n + 3))

    D_mat = arb_mat(N, N)
    for r in range(N):
        for c in range(N):
            D_mat[r, c] = M_mat[r, c] + M_mat[c, r] - alpha * I_mat[r, c]
    return I_mat, D_mat


def solve_top_eigenpair(I_mat: arb_mat, D_mat: arb_mat, dps: int = 140) -> tuple[float, list[str]]:
    """Solve top generalized eigenpair at high precision and return (eta, normalized coeff strings)."""
    n = I_mat.nrows()
    with mp.workdps(dps):
        cv = lambda x: mp.mpf(x.mid().str(dps - 10, radius=False))
        A = mp.matrix([[cv(I_mat[i, j]) for j in range(n)] for i in range(n)])
        B = mp.matrix([[cv(D_mat[i, j]) for j in range(n)] for i in range(n)])
        Linv = mp.cholesky(A) ** -1
        ev, V = mp.eigsy(Linv * B * Linv.T)
        top = max(range(n), key=lambda j: ev[j])
        c = Linv.T * V[:, top]
        c0 = c[0]
        return float(ev[top]), [mp.nstr(c[i] / c0, 60) for i in range(n)]


def compute_U_alpha_fast_gl(max_n: int, alpha_val: float | str, panels: list[float], n_gl: int = 64, work_dps: int = 140) -> list[arb]:
    """Compute U_n(alpha) simultaneously via composite GL in Arb (evaluates exp once per node)."""
    old_prec = ctx.dps
    ctx.dps = work_dps
    try:
        g = arb.const_euler()
        shift = g + arb(str(alpha_val))
        pi = arb.pi()
        nodes = _gauss(n_gl, ctx.prec)
        acc = [arb(0)] * (max_n + 1)
        for a, b in zip(panels, panels[1:]):
            half = arb(b - a) / 2
            mid = arb(a + b) / 2
            for x_gl, w_gl in nodes:
                t = mid + half * x_gl
                wt = half * w_gl
                z = 1 + acb(0, 1) * t
                cur = (z * (shift + z.log())).exp() / z
                inv_z = 1 / z
                for n in range(max_n + 1):
                    acc[n] += wt * cur.real
                    cur *= inv_z
        return [acc[n] * math.factorial(n) / pi for n in range(max_n + 1)]
    finally:
        ctx.dps = old_prec


def get_witnesses(refresh: bool = False) -> dict:
    """Compute or load asymptotic witnesses for Baseline, Theorem A, and Theorem B."""
    if not refresh and os.path.exists(WITNESS_CACHE):
        with open(WITNESS_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)

    t0 = time.time()
    panels = [0, 0.5, 1.5, 3, 6, 10, 16, 24, 35, 50, 70, 100, 140, 190, 250]
    work_dps = 200
    ctx.dps = work_dps

    # Baseline: alpha=0, deg=20
    U0 = compute_U_alpha_fast_gl(41, "0", panels, n_gl=96, work_dps=work_dps)
    I0, D0 = build_shifted_forms(U0, 20, arb(0))
    eta0, c0 = solve_top_eigenpair(I0, D0, dps=work_dps)

    # Theorem A: alpha=-0.3367, deg=20
    alphaA = "-0.3367"
    UA = compute_U_alpha_fast_gl(41, alphaA, panels, n_gl=96, work_dps=work_dps)
    IA, DA = build_shifted_forms(UA, 20, arb(alphaA))
    etaA, cA = solve_top_eigenpair(IA, DA, dps=work_dps)

    # Theorem B: alpha=-0.2596, d0=20, d1=16
    alphaB = "-0.2596"
    UB = compute_U_alpha_fast_gl(46, alphaB, panels, n_gl=96, work_dps=work_dps)
    IB, DB = build_B_forms(UB, 20, 16, arb(alphaB))
    etaB, cB = solve_top_eigenpair(IB, DB, dps=work_dps)

    data = {
        "baseline": {"alpha": "0", "deg": 20, "eta": eta0, "c": c0},
        "theorem_A": {"alpha": alphaA, "deg": 20, "eta": etaA, "c": cA},
        "theorem_B": {"alpha": alphaB, "d0": 20, "d1": 16, "eta": etaB, "q0": cB[:21], "q1": cB[21:]},
    }
    with open(WITNESS_CACHE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Generated witnesses in {time.time() - t0:.2f}s: eta0={eta0:.8f}, etaA={etaA:.8f}, etaB={etaB:.8f}")
    return data


# ---- Exact finite-k Laplace evaluator in Arb (supports both radial and P2) ----

@lru_cache(None)
def _gauss(n: int, prec: int) -> list[tuple[arb, arb]]:
    return [arb.legendre_p_root(n, i, weight=True) for i in range(n)]


def _K_log(z: acb) -> acb:
    lz, g, az = z.log(), arb.const_euler(), abs(z).upper()
    out = lz * lz / 2 + g * lz + g * g / 2 + arb.pi() ** 2 / 12
    term, n, goal = acb(1), 0, 2.0 ** (-ctx.prec - 10)
    while True:
        n += 1
        term *= -z / n
        out += term / (n ** 2)
        if n + 2 > 2 * az:
            tail = abs(term).upper() * az / ((n + 1) ** 3) / (1 - az / (n + 2))
            if tail < goal * (1 + abs(out).upper()):
                return out + acb(tail.union(-tail), tail.union(-tail))


def _moments(s: acb, a: arb, M: int) -> tuple[list[acb], list[acb], list[acb]]:
    """Return A_m, B_m, C_m = int_0^inf e^{-sy} y^m {1, log(1+ay), log^2(1+ay)} dy for m=0..M."""
    A = [acb(math.factorial(m)) / (s ** (m + 1)) for m in range(M + 1)]
    z = s / a
    ez = z.exp()
    e1 = z.expint(1)
    R = [ez * e1 / a]
    B = [ez * e1 / s]
    C = [2 * ez * _K_log(z) / s]
    D = [s * C[0] / (2 * a)]
    for m in range(1, M + 1):
        R.append((A[m - 1] - R[m - 1]) / a)
        B.append((m * B[m - 1] + a * R[m]) / s)
        D.append((B[m - 1] - D[m - 1]) / a)
        C.append((m * C[m - 1] + 2 * a * D[m]) / s)
    return A, B, C


def _W_tuple(s: acb, a: arb) -> tuple[acb, acb, acb]:
    """Return exact (W0(s;a), W2(s;a), W4(s;a)) via partial fractions."""
    z = s / a
    E = z.exp() * z.expint(1) / a  # int_0^inf e^{-st}/(1+at) dt
    W0 = 1 / a - (s / a) * E
    W2 = (1 / s - 2 * E + W0) / (a ** 2)
    W4 = (2 * a ** 2 / s ** 3 - 2 * a / s ** 2 + 3 / s - 4 * E + W0) / (a ** 4)
    return W0, W2, W4


def _poly_v_parts(q_coeffs: list[arb], a: arb) -> tuple[arb_poly, arb_poly]:
    """Express v(Y) = int_0^Y Q(Y-t)/(1+at) dt as P(Y) + R(Y)*log(1+aY)."""
    Yb, aY = arb_poly([1 / a, 1]), arb_poly([1, a])
    P = arb_poly([0])
    R = arb_poly([0])
    for j, qj in enumerate(q_coeffs):
        if qj == 0:
            continue
        Rj = Yb ** j * (1 / a)
        Pj = arb_poly([0])
        for r in range(1, j + 1):
            Pj += math.comb(j, r) * (-1 / a) ** r / (r * a) * Yb ** (j - r) * (aY ** r - 1)
        P += qj * Pj
        R += qj * Rj
    return P, R


def _coeffs_pad(p: arb_poly, n: int) -> list[arb]:
    c = p.coeffs()
    return c + [arb(0)] * (n - len(c))


DEFAULT_PANELS = [0.0, 0.5, 1.5, 3.0, 6.0, 10.0, 16.0, 24.0, 35.0, 50.0, 70.0]


def eval_finite_k_B(
    k: int,
    a: arb,
    sigma: arb,
    q0_Y: list[arb],
    q1_Y: list[arb],
    panels: list[float] | None = None,
    n_gl: int = 64,
    prec: int = 448,
) -> tuple[arb, arb, arb]:
    """Evaluate exact finite-k (I/W0(sig)^k, kJ/W0(sig)^k, M_k) for F=[Q0(Y)+Q1(Y)*sig^2*P2] prod (1+at_j)^{-1}."""
    if panels is None:
        panels = DEFAULT_PANELS
    old_prec = ctx.prec
    ctx.prec = prec
    try:
        d_eff = max(len(q0_Y) - 1, len(q1_Y) + 1)
        M = 2 * d_eff

        Q0p = arb_poly(q0_Y)
        Q1p = arb_poly(q1_Y)
        cI_00 = _coeffs_pad(Q0p * Q0p, M + 1)
        cI_01 = _coeffs_pad(2 * Q0p * Q1p, M + 1)
        cI_11 = _coeffs_pad(Q1p * Q1p, M + 1)

        P0, R0 = _poly_v_parts(q0_Y, a)
        P1, R1 = _poly_v_parts(q1_Y, a)
        sig2 = sigma ** 2
        # v_{12}(Y) = (1/a^2) v_1(Y) + (1/a^2) sum q1_j (a Y^{j+2}/((j+1)(j+2)) - Y^{j+1}/(j+1))
        poly_extra_coeffs = [arb(0)] * (len(q1_Y) + 2)
        for j, qj in enumerate(q1_Y):
            if qj == 0:
                continue
            poly_extra_coeffs[j + 1] -= qj / (j + 1)
            poly_extra_coeffs[j + 2] += a * qj / ((j + 1) * (j + 2))
        poly_extra = arb_poly(poly_extra_coeffs) * (sig2 / (a ** 2))

        PA = P0 + (sig2 / (a ** 2)) * P1 + poly_extra
        RA = R0 + (sig2 / (a ** 2)) * R1
        PB = sig2 * P1
        RB = sig2 * R1

        cJ_AA = [_coeffs_pad(p, M + 1) for p in (PA * PA, 2 * PA * RA, RA * RA)]
        cJ_AB = [_coeffs_pad(p, M + 1) for p in (2 * PA * PB, 2 * (PA * RB + PB * RA), 2 * RA * RB)]
        cJ_BB = [_coeffs_pad(p, M + 1) for p in (PB * PB, 2 * PB * RB, RB * RB)]

        W0_sig = _W_tuple(acb(sigma), a)[0].real
        nodes = _gauss(n_gl, prec)

        I_acc = arb(0)
        kJ_acc = arb(0)
        has_q1 = any(x != 0 for x in q1_Y)

        for a0, b0 in zip(panels, panels[1:]):
            left = arb(a0)
            dtau = arb(b0 - a0)
            for x_gl, w_gl in nodes:
                tau = left + dtau * (x_gl + 1) / 2
                s = acb(sigma, sigma * tau)
                wt = sigma * dtau * w_gl / (2 * arb.pi())

                W0, W2, W4 = _W_tuple(s, a)
                phi0 = W0 / W0_sig
                phi2 = W2 / W0_sig
                phi4 = W4 / W0_sig

                A_m, B_m, C_m = _moments(s, a, M)
                phase = (acb(0, 1) * sigma * tau).exp()

                phi_km3 = phi0 ** (k - 3)
                phi_km2 = phi_km3 * phi0
                phi_km1 = phi_km2 * phi0
                phi_k = phi_km1 * phi0

                # I integrand (divided by e^sigma * W0_sig^k)
                gI_00 = sum((c * m for c, m in zip(cI_00, A_m)), acb(0))
                val_I = gI_00 * phi_k
                if has_q1:
                    gI_01 = sum((c * m for c, m in zip(cI_01, A_m)), acb(0))
                    gI_11 = sum((c * m for c, m in zip(cI_11, A_m)), acb(0))
                    val_I += gI_01 * sig2 * k * phi2 * phi_km1
                    val_I += gI_11 * (sig2 ** 2) * (k * phi4 * phi_km1 + k * (k - 1) * (phi2 ** 2) * phi_km2)
                I_acc += wt * (phase * val_I).real

                # kJ integrand (divided by e^sigma * W0_sig^k)
                def eval_V(c_triple):
                    return sum((c * m for cs, X in zip(c_triple, (A_m, B_m, C_m)) for c, m in zip(cs, X)), acb(0))

                V_AA = eval_V(cJ_AA)
                val_J = V_AA * phi_km1
                if has_q1:
                    V_AB = eval_V(cJ_AB)
                    V_BB = eval_V(cJ_BB)
                    val_J += V_AB * (k - 1) * phi2 * phi_km2
                    val_J += V_BB * ((k - 1) * phi4 * phi_km2 + (k - 1) * (k - 2) * (phi2 ** 2) * phi_km3)
                kJ_acc += wt * (k / W0_sig) * (phase * val_J).real

        return I_acc, kJ_acc, kJ_acc / I_acc
    finally:
        ctx.prec = old_prec


def validate_k4_direct() -> None:
    """Validate eval_finite_k_B at k=4 against direct 4D Duffy quadrature in real space."""
    print("=== Validating exact finite-k Laplace P2 formula at k=4 vs direct 4D quadrature ===")
    k = 4
    alpha = -0.2596
    sigma = math.log(k) + alpha
    a = k * sigma
    q0_c = [1.0, -0.4, 0.05]
    q1_c = [0.25, -0.08, 0.015]
    q0_Y = [q0_c[j] * (sigma ** j) for j in range(len(q0_c))]
    q1_Y = [q1_c[j] * (sigma ** j) for j in range(len(q1_c))]

    # Direct 4D Duffy quadrature
    n_gl = 22
    ut, uw = leggauss(n_gl)
    u = 0.5 * (ut + 1.0)
    w = 0.5 * uw

    # 3D outer grid (u1, u2, u3) on R_3
    u1, u2, u3 = np.meshgrid(u, u, u, indexing="ij")
    w1, w2, w3 = np.meshgrid(w, w, w, indexing="ij")
    t1 = u1
    t2 = (1.0 - u1) * u2
    t3 = (1.0 - u1) * (1.0 - u2) * u3
    jac3 = ((1.0 - u1) ** 2) * (1.0 - u2)
    wt3 = w1 * w2 * w3 * jac3
    rem = 1.0 - t1 - t2 - t3  # = (1-u1)*(1-u2)*(1-u3)
    p2_3 = t1 ** 2 + t2 ** 2 + t3 ** 2
    g3 = 1.0 / ((1.0 + a * t1) * (1.0 + a * t2) * (1.0 + a * t3))

    # Integrate t4 = rem * u4
    t4 = rem[..., None] * u[None, None, None, :]
    y4 = rem[..., None] - t4
    x4 = sigma * y4
    Q0_v = q0_c[0] + q0_c[1] * x4 + q0_c[2] * (x4 ** 2)
    Q1_v = q1_c[0] + q1_c[1] * x4 + q1_c[2] * (x4 ** 2)
    p2_4 = p2_3[..., None] + t4 ** 2
    F_v = (Q0_v + Q1_v * (sigma ** 2) * p2_4) * g3[..., None] / (1.0 + a * t4)

    # I_4D = int (F^2 * rem) du4 dR3
    I_4D = float(np.sum(wt3[..., None] * rem[..., None] * (F_v ** 2) * w[None, None, None, :]))
    # J_4D = int (int_0^rem F dt4)^2 dR3
    inner = np.sum(rem[..., None] * F_v * w[None, None, None, :], axis=-1)
    J_4D = float(np.sum(wt3 * (inner ** 2)))
    M_4D = k * J_4D / I_4D

    # Exact Laplace in Arb (k=4 needs larger T_norm for 1e-10 precision)
    ctx.prec = 256
    a_arb = arb(k) * (arb(k).log() + arb(str(alpha)))
    sig_arb = a_arb / k
    I_norm, kJ_norm, M_lap = eval_finite_k_B(
        k, a_arb, sig_arb,
        [arb(repr(x)) for x in q0_Y], [arb(repr(x)) for x in q1_Y],
        panels=[0.0, 0.5, 1.5, 3.0, 6.0, 10.0, 16.0, 25.0, 40.0, 60.0, 90.0, 130.0, 170.0],
        n_gl=48, prec=256,
    )
    W0_sig = _W_tuple(acb(sig_arb), a_arb)[0].real
    scale_I = sig_arb.exp() * (W0_sig ** k)
    I_lap = float((I_norm * scale_I).mid())
    kJ_lap = float((kJ_norm * scale_I).mid())
    M_lap_f = float(M_lap.mid())
    rel_I = abs(I_lap - I_4D) / I_4D
    rel_J = abs(kJ_lap - k * J_4D) / (k * J_4D)
    rel_M = abs(M_lap_f - M_4D) / M_4D
    print(f"  4D direct: I={I_4D:.10f}, kJ={k*J_4D:.10f}, M_4={M_4D:.10f}")
    print(f"  Laplace  : I={I_lap:.10f}, kJ={kJ_lap:.10f}, M_4={M_lap_f:.10f}")
    print(f"  Rel errs : I={rel_I:.2e}, kJ={rel_J:.2e}, M_4={rel_M:.2e}")
    assert rel_M < 1e-6 and rel_I < 1e-6 and rel_J < 1e-6, "k=4 validation failed!"
    print("PASS: k=4 finite-k Laplace P2 formula matches 4D quadrature to < 1e-6.\n")
    return {
        "k": k,
        "I_4D": I_4D,
        "I_lap": I_lap,
        "kJ_4D": k * J_4D,
        "kJ_lap": kJ_lap,
        "M_4D": M_4D,
        "M_lap": M_lap_f,
        "rel_err_I": rel_I,
        "rel_err_kJ": rel_J,
        "rel_err_M": rel_M,
    }


def partitions_ge2(W: int) -> list[tuple[int, ...]]:
    """Return all integer partitions of weight 0 <= w <= W with parts >= 2."""
    out: list[tuple[int, ...]] = [()]

    def gen_w(w: int, max_p: int, cur: tuple[int, ...]) -> None:
        if w == 0:
            out.append(cur)
            return
        for p in range(min(w, max_p), 1, -1):
            gen_w(w - p, p, cur + (p,))

    for w in range(2, W + 1):
        gen_w(w, w, ())
    return out


@lru_cache(None)
def set_partition_sums(L: tuple[int, ...]) -> dict[tuple[int, ...], int]:
    """Return {sorted_tuple_of_block_sums: multiplicity} over all set partitions of multiset L."""
    if not L:
        return {(): 1}
    first, rest = L[0], L[1:]
    n = len(rest)
    acc: dict[tuple[int, ...], int] = {}
    for mask in range(1 << n):
        s = first
        rem = []
        for i in range(n):
            if (mask >> i) & 1:
                s += rest[i]
            else:
                rem.append(rest[i])
        for sub_t, cnt in set_partition_sums(tuple(rem)).items():
            key = tuple(sorted(sub_t + (s,)))
            acc[key] = acc.get(key, 0) + cnt
    return acc


@lru_cache(None)
def _W_part_asymp(L: tuple[int, ...]) -> dict[int, int]:
    """Asymptotic PPP moment weights sum c_r z^{-r} for sorted multiset L."""
    if not L:
        return {0: 1}
    first, rest = L[0], L[1:]
    n = len(rest)
    acc: dict[int, int] = {}
    for mask in range(1 << n):
        s = first
        rem = []
        for i in range(n):
            if (mask >> i) & 1:
                s += rest[i]
            else:
                rem.append(rest[i])
        k_pow = s - 1
        k_coef = math.factorial(s - 2)
        for r, c in _W_part_asymp(tuple(rem)).items():
            acc[r + k_pow] = acc.get(r + k_pow, 0) + k_coef * c
    return acc


def W_all_partial_frac(s: acb, a: arb, max_m: int) -> list[acb]:
    """Exact W_m(s; a) = int_0^inf t^m (1+at)^{-2} e^{-st} dt for m = 0..max_m via partial fractions."""
    z = s / a
    E0 = z.exp() * z.expint(1) / a
    W0 = 1 / a - (s / a) * E0
    W = [W0]
    if max_m >= 1:
        W.append((E0 - W0) / a)
    inv_s = 1 / s
    a_over_s = a * inv_s
    terms = [inv_s]
    for r in range(1, max(1, max_m - 1)):
        terms.append(terms[-1] * r * a_over_s)
    inv_a = 1 / a
    inv_a_pow = inv_a
    for m in range(2, max_m + 1):
        inv_a_pow *= inv_a
        poly_sum = acb(0)
        for r in range(m - 1):
            sgn = 1 if ((m - r) % 2 == 0) else -1
            poly_sum += sgn * (m - 1 - r) * terms[r]
        sgn_m = 1 if (m % 2 == 0) else -1
        W.append(inv_a_pow * (poly_sum - sgn_m * m * E0 + sgn_m * W0))
    return W


def poly_v_shift(
    q_coeffs: list[arb], a: arb, s_pow: int, P_base: arb_poly, R_base: arb_poly
) -> tuple[arb_poly, arb_poly]:
    """Express int_0^Y Q(Y-t) t^{s_pow} / (1 + at) dt as P_s(Y) + R_s(Y) * log(1 + aY)."""
    sgn_s = 1 if (s_pow % 2 == 0) else -1
    inv_as = arb(1) / (a ** s_pow)
    P_extra_coeffs = [arb(0)] * (len(q_coeffs) + s_pow)
    for r in range(s_pow):
        sgn_r = 1 if ((s_pow - 1 - r) % 2 == 0) else -1
        ar = (a ** r) * sgn_r
        fr = math.factorial(r)
        for j, qj in enumerate(q_coeffs):
            if qj == 0:
                continue
            coef = qj * ar * (math.factorial(j) * fr) / math.factorial(j + r + 1)
            P_extra_coeffs[j + r + 1] += coef
    P_out = (sgn_s * P_base + arb_poly(P_extra_coeffs)) * inv_as
    R_out = (sgn_s * R_base) * inv_as
    return P_out, R_out


def eval_finite_k_W(
    k: int,
    a: arb,
    sigma: arb,
    parts: list[tuple[int, ...]],
    q_Y_list: list[list[arb]],
    panels: list[float] | None = None,
    n_gl: int = 48,
    prec: int = 448,
) -> tuple[arb, arb, arb]:
    """Exact finite-k Laplace evaluator in Arb for complete Lambda_W symmetric hierarchy."""
    if panels is None:
        panels = DEFAULT_PANELS
    old_prec = ctx.prec
    ctx.prec = prec
    try:
        W_max = max((sum(p) for p in parts), default=0)
        max_m = 2 * W_max
        part_idx = {p: idx for idx, p in enumerate(parts)}
        n_p = len(parts)
        M = max(len(q) - 1 + sum(p) for p, q in zip(parts, q_Y_list)) * 2

        I_polys: dict[tuple[int, ...], arb_poly] = {}
        for i in range(n_p):
            Qi = arb_poly(q_Y_list[i])
            for j in range(i, n_p):
                Qj = arb_poly(q_Y_list[j])
                L = tuple(sorted(parts[i] + parts[j], reverse=True))
                mult = 1 if i == j else 2
                prod = mult * (Qi * Qj)
                I_polys[L] = I_polys[L] + prod if L in I_polys else prod
        I_coeffs = {L: p.coeffs() for L, p in I_polys.items()}

        P_raw, R_raw = [], []
        for q in q_Y_list:
            Pr, Rr = _poly_v_parts(q, a)
            P_raw.append(Pr)
            R_raw.append(Rr)

        P_eff = [P_raw[i] for i in range(n_p)]
        R_eff = [R_raw[i] for i in range(n_p)]
        for idx_lam, lam in enumerate(parts):
            m_lam = len(lam)
            if m_lam == 0:
                continue
            sdict: dict[tuple[int, tuple[int, ...]], int] = {}
            for mask in range(1, 1 << m_lam):
                s_pow = 0
                rem = []
                for b in range(m_lam):
                    if (mask >> b) & 1:
                        s_pow += lam[b]
                    else:
                        rem.append(lam[b])
                mu = tuple(sorted(rem, reverse=True))
                sdict[(s_pow, mu)] = sdict.get((s_pow, mu), 0) + 1
            for (s_pow, mu), mult in sdict.items():
                Ps, Rs = poly_v_shift(q_Y_list[idx_lam], a, s_pow, P_raw[idx_lam], R_raw[idx_lam])
                fac = mult * (sigma ** s_pow)
                j_mu = part_idx[mu]
                P_eff[j_mu] = P_eff[j_mu] + fac * Ps
                R_eff[j_mu] = R_eff[j_mu] + fac * Rs

        J_polys: dict[tuple[int, ...], tuple[arb_poly, arb_poly, arb_poly]] = {}
        for i in range(n_p):
            for j in range(i, n_p):
                L = tuple(sorted(parts[i] + parts[j], reverse=True))
                mult = 1 if i == j else 2
                pAA = mult * (P_eff[i] * P_eff[j])
                pAB = mult * (P_eff[i] * R_eff[j] + P_eff[j] * R_eff[i])
                pBB = mult * (R_eff[i] * R_eff[j])
                if L in J_polys:
                    o0, o1, o2 = J_polys[L]
                    J_polys[L] = (o0 + pAA, o1 + pAB, o2 + pBB)
                else:
                    J_polys[L] = (pAA, pAB, pBB)
        J_coeffs = {L: (t[0].coeffs(), t[1].coeffs(), t[2].coeffs()) for L, t in J_polys.items()}

        all_L = sorted(set(I_coeffs.keys()) | set(J_coeffs.keys()))
        max_parts_L = max(len(L) for L in all_L)
        ff_k = [arb(1)]
        ff_km1 = [arb(1)]
        for t in range(1, max_parts_L + 1):
            ff_k.append(ff_k[-1] * max(0, k - t + 1))
            ff_km1.append(ff_km1[-1] * max(0, k - 1 - t + 1))

        unique_bsums: dict[tuple[int, ...], int] = {}
        bsum_list: list[tuple[int, ...]] = []
        sig_pows = {L: sigma ** sum(L) for L in all_L}
        I_Ls = sorted(I_coeffs.keys())
        J_Ls = sorted(J_coeffs.keys())
        n_I, n_J = len(I_Ls), len(J_Ls)

        for L in all_L:
            for bsums in set_partition_sums(L):
                if bsums not in unique_bsums:
                    unique_bsums[bsums] = len(bsum_list)
                    bsum_list.append(bsums)
        n_B = len(bsum_list)

        max_len_I = max((len(I_coeffs[L]) for L in I_Ls), default=1)
        max_len_J = max((max(len(J_coeffs[L][0]), len(J_coeffs[L][1]), len(J_coeffs[L][2])) for L in J_Ls), default=1)

        mat_cI = acb_mat(n_I, max_len_I)
        mat_spI = acb_mat(n_I, n_B)
        for row, L in enumerate(I_Ls):
            spL = sig_pows[L]
            for j, c in enumerate(I_coeffs[L]):
                mat_cI[row, j] = acb(c * spL)
            for bsums, cnt in set_partition_sums(L).items():
                if len(bsums) <= k:
                    mat_spI[row, unique_bsums[bsums]] += acb(arb(cnt) * ff_k[len(bsums)])

        mat_cA = acb_mat(n_J, max_len_J)
        mat_cB = acb_mat(n_J, max_len_J)
        mat_cC = acb_mat(n_J, max_len_J)
        mat_spJ = acb_mat(n_J, n_B)
        for row, L in enumerate(J_Ls):
            spL = sig_pows[L]
            cA, cB, cC = J_coeffs[L]
            for j, c in enumerate(cA):
                mat_cA[row, j] = acb(c * spL)
            for j, c in enumerate(cB):
                mat_cB[row, j] = acb(c * spL)
            for j, c in enumerate(cC):
                mat_cC[row, j] = acb(c * spL)
            for bsums, cnt in set_partition_sums(L).items():
                if len(bsums) <= k - 1:
                    mat_spJ[row, unique_bsums[bsums]] += acb(arb(cnt) * ff_km1[len(bsums)])

        W0_sig = W_all_partial_frac(acb(sigma), a, 0)[0].real
        nodes = _gauss(n_gl, prec)
        I_acc = arb(0)
        kJ_acc = arb(0)
        pi2 = 2 * arb.pi()

        for a0, b0 in zip(panels, panels[1:]):
            left = arb(a0)
            dtau = arb(b0 - a0)
            for x_gl, w_gl in nodes:
                tau = left + dtau * (x_gl + 1) / 2
                s = acb(sigma, sigma * tau)
                wt = sigma * dtau * w_gl / pi2
                Ws = W_all_partial_frac(s, a, max_m)
                phi0 = Ws[0] / W0_sig
                inv_W0 = 1 / Ws[0]
                rat = [w * inv_W0 for w in Ws]
                bvals = acb_mat(n_B, 1)
                for idx_b, bsums in enumerate(bsum_list):
                    pr = acb(1)
                    for bs in bsums:
                        pr *= rat[bs]
                    bvals[idx_b, 0] = pr

                A_m, B_m, C_m = _moments(s, a, M)
                col_AI = acb_mat(max_len_I, 1, A_m[:max_len_I])
                col_AJ = acb_mat(max_len_J, 1, A_m[:max_len_J])
                col_BJ = acb_mat(max_len_J, 1, B_m[:max_len_J])
                col_CJ = acb_mat(max_len_J, 1, C_m[:max_len_J])

                gI_vec = mat_cI * col_AI
                GI_vec = mat_spI * bvals
                val_I = (gI_vec.transpose() * GI_vec)[0, 0]

                gJ_vec = mat_cA * col_AJ + mat_cB * col_BJ + mat_cC * col_CJ
                GJ_vec = mat_spJ * bvals
                val_J = (gJ_vec.transpose() * GJ_vec)[0, 0]

                phase = (acb(0, 1) * sigma * tau).exp()
                phi_km1 = phi0 ** (k - 1)
                phi_k = phi_km1 * phi0
                I_acc += wt * (phase * val_I * phi_k).real
                kJ_acc += wt * (k / W0_sig) * (phase * val_J * phi_km1).real

        return I_acc, kJ_acc, kJ_acc / I_acc
    finally:
        ctx.prec = old_prec


def validate_k5_direct_C() -> dict:
    """Validate eval_finite_k_W at k=5 for all 30 blocks (Lambda_9) vs direct 5D Duffy quadrature."""
    print("=== Validating exact finite-k set-partition Laplace at k=5 vs 5D Duffy (Lambda_9, 30 blocks) ===")
    k = 5
    alpha = -0.2615
    sigma = math.log(k) + alpha
    a = k * sigma
    parts9 = partitions_ge2(9)
    rng = np.random.default_rng(42)
    q_x = [[float(rng.uniform(-0.3, 0.3)) for _ in range(3)] for _ in parts9]
    q_x[0] = [1.0, -0.3, 0.05]

    n_gl = 14
    ut, uw = leggauss(n_gl)
    u = 0.5 * (ut + 1.0)
    w = 0.5 * uw
    u1, u2, u3, u4 = np.meshgrid(u, u, u, u, indexing="ij")
    w1, w2, w3, w4 = np.meshgrid(w, w, w, w, indexing="ij")
    t1 = u1
    t2 = (1.0 - u1) * u2
    t3 = (1.0 - u1) * (1.0 - u2) * u3
    t4 = (1.0 - u1) * (1.0 - u2) * (1.0 - u3) * u4
    jac4 = ((1.0 - u1) ** 3) * ((1.0 - u2) ** 2) * (1.0 - u3)
    wt4 = w1 * w2 * w3 * w4 * jac4
    rem = 1.0 - t1 - t2 - t3 - t4
    g4 = 1.0 / ((1.0 + a * t1) * (1.0 + a * t2) * (1.0 + a * t3) * (1.0 + a * t4))
    P_prev = {r: (t1 ** r + t2 ** r + t3 ** r + t4 ** r) for r in range(2, 10)}

    t5 = rem[..., None] * u[None, None, None, None, :]
    x5 = sigma * (rem[..., None] - t5)
    P_full = {r: (sigma ** r) * (P_prev[r][..., None] + t5 ** r) for r in range(2, 10)}
    F_sum = np.zeros_like(x5)
    for lam, qc in zip(parts9, q_x):
        Qv = qc[0] + qc[1] * x5 + qc[2] * (x5 ** 2)
        mon = np.ones_like(x5)
        for r in lam:
            mon = mon * P_full[r]
        F_sum += Qv * mon
    F_v = F_sum * g4[..., None] / (1.0 + a * t5)
    I_5D = float(np.sum(wt4[..., None] * rem[..., None] * (F_v ** 2) * w[None, None, None, None, :]))
    inner = np.sum(rem[..., None] * F_v * w[None, None, None, None, :], axis=-1)
    J_5D = float(np.sum(wt4 * (inner ** 2)))
    M_5D = k * J_5D / I_5D

    ctx.prec = 320
    a_arb = arb(k) * (arb(k).log() + arb(str(alpha)))
    sig_arb = a_arb / k
    q_Y_arb = [[arb(repr(float(c))) * (sig_arb ** j) for j, c in enumerate(qc)] for qc in q_x]
    I_n, kJ_n, M_lap = eval_finite_k_W(
        k, a_arb, sig_arb, parts9, q_Y_arb,
        panels=[0.0, 0.5, 1.5, 3.0, 6.0, 10.0, 16.0, 25.0, 40.0, 65.0, 100.0, 140.0],
        n_gl=40, prec=320,
    )
    W0_s = W_all_partial_frac(acb(sig_arb), a_arb, 0)[0].real
    sc = sig_arb.exp() * (W0_s ** k)
    I_l = float((I_n * sc).mid())
    kJ_l = float((kJ_n * sc).mid())
    M_l = float(M_lap.mid())
    rel_I = abs(I_l - I_5D) / I_5D
    rel_J = abs(kJ_l - k * J_5D) / (k * J_5D)
    rel_M = abs(M_l - M_5D) / M_5D
    print(f"  5D direct: I={I_5D:.10e}, kJ={k*J_5D:.10e}, M_5={M_5D:.10f}")
    print(f"  Laplace  : I={I_l:.10e}, kJ={kJ_l:.10e}, M_5={M_l:.10f}")
    print(f"  Rel errs : I={rel_I:.2e}, kJ={rel_J:.2e}, M_5={rel_M:.2e}")
    assert rel_M < 1e-6 and rel_I < 1e-6 and rel_J < 1e-6, "k=5 validation failed!"
    print("PASS: k=5 finite-k Laplace Lambda_9 formula matches 5D quadrature to < 1e-6.\n")
    return {
        "k": k,
        "I_5D": I_5D,
        "I_lap": I_l,
        "kJ_5D": k * J_5D,
        "kJ_lap": kJ_l,
        "M_5D": M_5D,
        "M_lap": M_l,
        "rel_err_I": rel_I,
        "rel_err_kJ": rel_J,
        "rel_err_M": rel_M,
    }


def compute_L_exact_radial(alpha: float, c_list: list[float]) -> tuple[float, float, float]:
    """Return (<Q,Q>, <Q,AQ>, <AQ,AQ>) in x-space for L-exact prediction."""
    ev = XSpaceEvaluator(max_deg=len(c_list) - 1, X=9.0, n_x=28, n_w=48)
    nu0 = eval_nu(ev.x, alpha=alpha, dtau=0.025)
    wt = nu0 * ev.w
    # Evaluate Q(x) and Q(x-w) directly in monomial basis (in float64 with Horner)
    # Wait: monomial c_list of degree 20 can have large cancellations in float64!
    # So use high-precision mpmath at the grid nodes!
    with mp.workdps(80):
        c_mp = [mp.mpf(str(x)) for x in c_list]
        def eval_poly(vals):
            out = np.empty_like(vals, dtype=np.float64)
            for idx, xv in np.ndenumerate(vals):
                xm = mp.mpf(float(xv))
                acc = mp.mpf(0)
                for coef in reversed(c_mp):
                    acc = acc * xm + coef
                out[idx] = float(acc)
            return out
        Qx = eval_poly(ev.x)
        Qsh = eval_poly(ev.x[:, None] - ev.WW)
    AQ = Qx * np.log(ev.x) + np.sum(ev.WT * (Qsh - Qx[:, None]) / ev.WW, axis=1)
    qq = float(np.sum(wt * Qx * Qx))
    qaq = float(np.sum(wt * Qx * AQ))
    aqaq = float(np.sum(wt * AQ * AQ))
    return qq, qaq, aqaq


def compute_L_exact_B(alpha: float, q0_list: list[str], q1_list: list[str]) -> tuple[float, float, float]:
    """Return (I_norm, M_sym_half, N_AA) for Theorem B L-exact prediction."""
    ev = XSpaceEvaluator(max_deg=max(len(q0_list), len(q1_list)) - 1, X=9.0, n_x=28, n_w=48)
    # Precompute nu_{0,0}, nu_{1,0}, nu_{2,0} via Bromwich
    from powersum_screen import bell_weights_coeffs, compute_nu_pq_grid
    wc = bell_weights_coeffs(2, 0)
    nu_pq = compute_nu_pq_grid(ev.x, alpha, 2, 0, wc, dtau=0.025)
    w0 = nu_pq[(0, 0)] * ev.w
    w1 = nu_pq[(1, 0)] * ev.w
    w2 = nu_pq[(2, 0)] * ev.w

    with mp.workdps(100):
        def eval_poly(c_strs, vals):
            c_mp = [mp.mpf(str(x)) for x in c_strs]
            out = np.empty_like(vals, dtype=np.float64)
            for idx, xv in np.ndenumerate(vals):
                xm = mp.mpf(float(xv))
                acc = mp.mpf(0)
                for coef in reversed(c_mp):
                    acc = acc * xm + coef
                out[idx] = float(acc)
            return out
        Q0x = eval_poly(q0_list, ev.x)
        Q0sh = eval_poly(q0_list, ev.x[:, None] - ev.WW)
        Q1x = eval_poly(q1_list, ev.x)
        Q1sh = eval_poly(q1_list, ev.x[:, None] - ev.WW)

    AQ0 = Q0x * np.log(ev.x) + np.sum(ev.WT * (Q0sh - Q0x[:, None]) / ev.WW, axis=1)
    AQ1 = Q1x * np.log(ev.x) + np.sum(ev.WT * (Q1sh - Q1x[:, None]) / ev.WW, axis=1)
    BQ1 = np.sum(ev.WT * Q1sh * ev.WW, axis=1)

    I_val = float(np.sum(w0 * Q0x * Q0x + 2.0 * w1 * Q0x * Q1x + w2 * Q1x * Q1x))
    G0 = AQ0 + BQ1
    G1 = AQ1
    M_half = float(np.sum(w0 * Q0x * G0 + w1 * (Q0x * G1 + Q1x * G0) + w2 * Q1x * G1))
    N_AA = float(np.sum(w0 * G0 * G0 + 2.0 * w1 * G0 * G1 + w2 * G1 * G1))
    return I_val, M_half, N_AA


def compute_L_exact_C(alpha_str: str, blocks_with_coeffs: list[dict]) -> tuple[float, float, float]:
    """Return exact 384-bit Arb (I_C, M_half_C, N_AA_C) for Theorem C L-exact prediction using L^(2)_n IBP."""
    for cand_dir in (
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "verifier")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    ):
        if os.path.exists(os.path.join(cand_dir, "verify_asymptotic.py")) and cand_dir not in sys.path:
            sys.path.insert(0, cand_dir)
    from verify_asymptotic import compute_independent_moments

    old_prec = ctx.prec
    ctx.prec = 384
    try:
        alpha_fr = Fraction(str(alpha_str))
        alpha_arb = _arb_frac(str(alpha_str))
        parts = [tuple(b["partition"]) for b in blocks_with_coeffs]
        degs = [int(b["deg"]) for b in blocks_with_coeffs]
        max_r = max(
            da + db + max(_W_part_asymp(tuple(sorted(pa + pb, reverse=True))).keys())
            for pa, da in zip(parts, degs)
            for pb, db in zip(parts, degs)
        )
        max_u = max_r + 2
        ver_panels = [0.0, 0.35, 0.9, 2.0, 4.5, 10.0, 22.0, 45.0, 85.0, 145.0, 225.0]
        U, L_mom = compute_independent_moments(max_u, alpha_fr, ver_panels, 384, m_gl=160, n_arc=32)
        L2 = [L_mom[n + 1] - (1 + alpha_arb) * L_mom[n] - U[n + 1] / (n + 1) for n in range(max_r + 1)]

        H = [arb(0)]
        H2 = [arb(0)]
        for m in range(1, max_r + 5):
            H.append(H[-1] + arb(1) / m)
            H2.append(H2[-1] + arb(1) / (m * m))
        zeta2 = (arb.pi() ** 2) / 6

        def inv_poch(n: int, r: int) -> arb:
            val = arb(1)
            for t in range(n + 1, n + r + 1):
                val /= t
            return val

        m0_c, m1_c, m2_c = {}, {}, {}

        def m0(n: int, Lt: tuple[int, ...]) -> arb:
            ck = (n, Lt)
            if ck not in m0_c:
                m0_c[ck] = sum((c * inv_poch(n, r) * U[n + r] for r, c in _W_part_asymp(Lt).items()), arb(0))
            return m0_c[ck]

        def m1(n: int, Lt: tuple[int, ...]) -> arb:
            ck = (n, Lt)
            if ck not in m1_c:
                m1_c[ck] = sum((c * inv_poch(n, r) * (H[n] * U[n + r] - L_mom[n + r]) for r, c in _W_part_asymp(Lt).items()), arb(0))
            return m1_c[ck]

        def m2(n: int, Lt: tuple[int, ...]) -> arb:
            ck = (n, Lt)
            if ck not in m2_c:
                c_const = H[n] ** 2 + zeta2 - H2[n]
                m2_c[ck] = sum(
                    (c * inv_poch(n, r) * (L2[n + r] - 2 * H[n] * L_mom[n + r] + c_const * U[n + r]) for r, c in _W_part_asymp(Lt).items()),
                    arb(0),
                )
            return m2_c[ck]

        n_p = len(parts)
        part_idx = {p: i for i, p in enumerate(parts)}
        q_coeffs = [[_arb_frac(s) for s in b["coeffs"]] for b in blocks_with_coeffs]
        p_coeffs = [[-H[j] * c for j, c in enumerate(qc)] for qc in q_coeffs]
        for idx_lam, lam in enumerate(parts):
            if not lam:
                continue
            m_lam = len(lam)
            sdict: dict[tuple[int, tuple[int, ...]], int] = {}
            for mask in range(1, 1 << m_lam):
                s_pow = sum(lam[b] for b in range(m_lam) if (mask >> b) & 1)
                mu = tuple(sorted([lam[b] for b in range(m_lam) if not ((mask >> b) & 1)], reverse=True))
                sdict[(s_pow, mu)] = sdict.get((s_pow, mu), 0) + 1
            for (s_pow, mu), mult in sdict.items():
                j_mu = part_idx[mu]
                need_deg = len(q_coeffs[idx_lam]) - 1 + s_pow
                if len(p_coeffs[j_mu]) <= need_deg:
                    p_coeffs[j_mu].extend([arb(0)] * (need_deg + 1 - len(p_coeffs[j_mu])))
                fs = math.factorial(s_pow - 1)
                for j, qj in enumerate(q_coeffs[idx_lam]):
                    p_coeffs[j_mu][j + s_pow] += mult * (arb(math.factorial(j) * fs) / math.factorial(j + s_pow)) * qj

        I_v = arb(0)
        M_h = arb(0)
        N_AA = arb(0)
        for bi in range(n_p):
            for bj in range(bi, n_p):
                mult = 1 if bi == bj else 2
                L_ab = tuple(sorted(parts[bi] + parts[bj], reverse=True))
                Qi, Qj = q_coeffs[bi], q_coeffs[bj]
                Pi, Pj = p_coeffs[bi], p_coeffs[bj]
                sI = arb(0)
                for u_deg, cu in enumerate(Qi):
                    for v_deg, cv in enumerate(Qj):
                        sI += cu * cv * m0(u_deg + v_deg, L_ab)
                I_v += mult * sI

                def pair_QG(Qa: list[arb], Qb: list[arb], Pb: list[arb]) -> arb:
                    acc = arb(0)
                    for u_deg, cu in enumerate(Qa):
                        for v_deg, cv in enumerate(Qb):
                            acc += cu * cv * m1(u_deg + v_deg, L_ab)
                        for v_deg, pv in enumerate(Pb):
                            acc += cu * pv * m0(u_deg + v_deg, L_ab)
                    return acc

                M_h += pair_QG(Qi, Qj, Pj) + (0 if bi == bj else pair_QG(Qj, Qi, Pi))
                sN = arb(0)
                for u_deg, cu in enumerate(Qi):
                    for v_deg, cv in enumerate(Qj):
                        sN += cu * cv * m2(u_deg + v_deg, L_ab)
                    for v_deg, pv in enumerate(Pj):
                        sN += cu * pv * m1(u_deg + v_deg, L_ab)
                for u_deg, pu in enumerate(Pi):
                    for v_deg, cv in enumerate(Qj):
                        sN += pu * cv * m1(u_deg + v_deg, L_ab)
                    for v_deg, pv in enumerate(Pj):
                        sN += pu * pv * m0(u_deg + v_deg, L_ab)
                N_AA += mult * sN

        return float(I_v.mid()), float(M_h.mid()), float(N_AA.mid())
    finally:
        ctx.prec = old_prec


def _finite_k_worker(args: tuple) -> tuple[str, int, int, float, str]:
    """Top-level worker for parallel finite-k Arb Laplace evaluations."""
    kind, k, alpha_str, payload, n_gl, prec = args
    ctx.prec = prec
    L_arb = arb(k).log()
    alpha_arb = _arb_frac(alpha_str)
    sig_arb = L_arb + alpha_arb
    a_arb = k * sig_arb
    if kind in ("baseline", "A"):
        c_arbs = [_arb_frac(s) for s in payload]
        q_Y = [c * (sig_arb ** j) for j, c in enumerate(c_arbs)]
        _, _, Mk_arb = eval_finite_k_B(
            k, a_arb, sig_arb, q_Y, [arb(0)],
            panels=DEFAULT_PANELS, n_gl=n_gl, prec=prec,
        )
    elif kind == "B":
        q0_arbs = [_arb_frac(s) for s in payload["q0"]]
        q1_arbs = [_arb_frac(s) for s in payload["q1"]]
        q0_Y = [c * (sig_arb ** j) for j, c in enumerate(q0_arbs)]
        q1_Y = [c * (sig_arb ** j) for j, c in enumerate(q1_arbs)]
        _, _, Mk_arb = eval_finite_k_B(
            k, a_arb, sig_arb, q0_Y, q1_Y,
            panels=DEFAULT_PANELS, n_gl=n_gl, prec=prec,
        )
    elif kind == "C":
        parts = [tuple(b["partition"]) for b in payload]
        q_Y_list = [[_arb_frac(s) * (sig_arb ** j) for j, s in enumerate(b["coeffs"])] for b in payload]
        _, _, Mk_arb = eval_finite_k_W(
            k, a_arb, sig_arb, parts, q_Y_list,
            panels=DEFAULT_PANELS, n_gl=n_gl, prec=prec,
        )
    else:
        raise ValueError(kind)
    return kind, k, n_gl, float(Mk_arb.mid()), Mk_arb.str(60)


def run_k_sweep(witnesses: dict, k4_res: dict | None = None, k5_res: dict | None = None) -> dict:
    """Run k = 10^3..10^8 sweep for Baseline, Theorem A, Theorem B, and Theorem C."""
    ks = [10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7, 10 ** 8]

    # Precompute L-exact inner products
    lex_base = compute_L_exact_radial(0.0, witnesses["baseline"]["c"])
    lex_A = compute_L_exact_radial(float(Fraction(witnesses["theorem_A"]["alpha"])), witnesses["theorem_A"]["c"])
    lex_B = compute_L_exact_B(float(Fraction(witnesses["theorem_B"]["alpha"])), witnesses["theorem_B"]["q0"], witnesses["theorem_B"]["q1"])
    wC = witnesses["theorem_C"]
    alphaC_f = float(Fraction(wC["alpha"]))
    lex_C = compute_L_exact_C(wC["alpha"], wC["blocks"])
    eta_L_exact_C = 2.0 * lex_C[1] / lex_C[0] - alphaC_f
    assert abs(eta_L_exact_C - wC["eta"]) < 1e-6, f"L-exact limit mismatch for C: {eta_L_exact_C} vs {wC['eta']}"

    # Submit all finite-k tasks in parallel (max 4 workers)
    tasks = []
    for k in ks:
        tasks.append(("baseline", k, witnesses["baseline"]["alpha"], witnesses["baseline"]["c"], 64, 448))
        tasks.append(("A", k, witnesses["theorem_A"]["alpha"], witnesses["theorem_A"]["c"], 64, 448))
        tasks.append(("B", k, witnesses["theorem_B"]["alpha"], {"q0": witnesses["theorem_B"]["q0"], "q1": witnesses["theorem_B"]["q1"]}, 64, 448))
        tasks.append(("C", k, wC["alpha"], wC["blocks"], 64, 448))
    # n_gl=80 spot-check for Theorem C at k=10^6
    tasks.append(("C", 10 ** 6, wC["alpha"], wC["blocks"], 80, 448))

    res_map: dict[tuple[str, int, int], float] = {}
    str_map: dict[tuple[str, int, int], str] = {}
    with ProcessPoolExecutor(max_workers=4) as pool:
        for kind, k, ngl, val, val_str in pool.map(_finite_k_worker, tasks):
            res_map[(kind, k, ngl)] = val
            str_map[(kind, k, ngl)] = val_str

    spot_64 = res_map[("C", 10 ** 6, 64)]
    spot_80 = res_map[("C", 10 ** 6, 80)]
    spot_diff = abs(spot_64 - spot_80)
    with mp.workdps(100):
        def _parse_arb_mid(s: str):
            core = s.strip().lstrip("[").rstrip("]")
            return mp.mpf(core.split("+/-")[0].strip())
        arb_diff_mp = abs(_parse_arb_mid(str_map[("C", 10 ** 6, 64)]) - _parse_arb_mid(str_map[("C", 10 ** 6, 80)]))
        arb_diff_f = float(arb_diff_mp)
    print(f"=== Theorem C quadrature spot-check at k=10^6: n_gl=64 vs 80 float64_diff={spot_diff:.2e}, arb448_diff={arb_diff_f:.3e} ===")
    assert arb_diff_f < 1e-15 and spot_diff < 1e-10, f"n_gl=64 vs 80 diff too large: {arb_diff_f}"

    def fit_asymp(rows, alpha_f):
        Ls = np.array([r[1] for r in rows if r[0] >= 10 ** 5])
        ys = np.array([r[3] for r in rows if r[0] >= 10 ** 5])
        invS = 1.0 / (Ls + alpha_f)
        X = np.column_stack([np.ones_like(invS), invS, invS ** 2])
        beta, _, _, _ = np.linalg.lstsq(X, ys, rcond=None)
        return beta

    def assemble_rows(kind: str, label: str, alpha_str: str, eta_lim: float, lex: tuple[float, float, float], ngl: int):
        print(f"=== Finite-k sweep: {label} (alpha={alpha_str}, limit eta={eta_lim:.6f}) ===")
        I_v, M_v, N_v = lex
        alpha_f = float(Fraction(alpha_str))
        rows = []
        for k in ks:
            L_f = math.log(k)
            Mk_f = res_map[(kind, k, ngl)]
            diff_L = Mk_f - L_f
            pred_L = (L_f ** 2 * I_v + 2.0 * L_f * M_v + N_v) / ((L_f + alpha_f) * I_v) - L_f
            rows.append((k, L_f, Mk_f, diff_L, pred_L))
            print(
                f"  k=10^{int(round(math.log10(k)))}: M_k={Mk_f:9.6f}, M_k - L = {diff_L:9.6f}, "
                f"L-exact={pred_L:9.6f}, err={diff_L - pred_L:+.2e}, vs limit={diff_L - eta_lim:+.6f}"
            )
        beta = fit_asymp(rows, alpha_f)
        print(f"  Fit (k=10^5..10^8 in 1/(L+a)): -c={beta[0]:.6f} (limit {eta_lim:.6f}, diff {beta[0]-eta_lim:+.2e})\n")
        return rows, beta

    rows_base, beta_base = assemble_rows("baseline", "Baseline (Lewulis)", witnesses["baseline"]["alpha"], witnesses["baseline"]["eta"], lex_base, 64)
    rows_A, beta_A = assemble_rows("A", "Theorem A (Shift + Q)", witnesses["theorem_A"]["alpha"], witnesses["theorem_A"]["eta"], lex_A, 64)
    rows_B, beta_B = assemble_rows("B", "Theorem B (Shift + P2)", witnesses["theorem_B"]["alpha"], witnesses["theorem_B"]["eta"], lex_B, 64)
    rows_C, beta_C = assemble_rows("C", "Theorem C (Shift + Lambda_9)", wC["alpha"], wC["eta"], lex_C, 64)

    print("=== Direct comparison across k = 10^3 .. 10^8 ===")
    for rb, rA, rB, rC in zip(rows_base, rows_A, rows_B, rows_C):
        k = rb[0]
        print(
            f"  k={k:9d}: Base={rb[3]:9.6f}, ThmA={rA[3]:9.6f} (+{rA[3]-rb[3]:.6f}), "
            f"ThmB={rB[3]:9.6f} (+{rB[3]-rb[3]:.6f}), ThmC={rC[3]:9.6f} (+{rC[3]-rb[3]:.6f}, vsB=+{rC[3]-rB[3]:.6f})"
        )
        assert rC[3] > rB[3] > rA[3] > rb[3], f"Strict order C > B > A > Base violated at k={k}!"

    assert abs(rows_C[-1][3] - rows_C[-1][4]) < 1e-6, f"Thm C k=10^8 L-exact error too large"
    assert abs(float(beta_B[0]) - witnesses["theorem_B"]["eta"]) < 0.0005
    assert abs(float(beta_C[0]) - wC["eta"]) < 0.0005, f"Theorem C fit {beta_C[0]} differs from limit {wC['eta']}"

    def fmt_rows(rows, eta_lim):
        return [
            {
                "k": r[0],
                "L": r[1],
                "M_k": r[2],
                "M_k_minus_L": r[3],
                "L_exact_pred": r[4],
                "L_exact_err": r[3] - r[4],
                "diff_vs_limit": r[3] - eta_lim,
            }
            for r in rows
        ]

    out = {
        "k4_validation": k4_res,
        "k5_validation": k5_res,
        "ngl_spot_check": {"k": 10 ** 6, "M_ngl64": spot_64, "M_ngl80": spot_80, "abs_diff": spot_diff, "arb448_abs_diff": arb_diff_f},
        "L_exact_checks": {"eta_limit_C": wC["eta"], "eta_L_exact_C": eta_L_exact_C, "abs_diff": abs(eta_L_exact_C - wC["eta"])},
        "sweeps": {
            "baseline": {"alpha": witnesses["baseline"]["alpha"], "eta_limit": witnesses["baseline"]["eta"], "fit_1_over_sigma": [float(x) for x in beta_base], "rows": fmt_rows(rows_base, witnesses["baseline"]["eta"])},
            "theorem_A": {"alpha": witnesses["theorem_A"]["alpha"], "eta_limit": witnesses["theorem_A"]["eta"], "fit_1_over_sigma": [float(x) for x in beta_A], "rows": fmt_rows(rows_A, witnesses["theorem_A"]["eta"])},
            "theorem_B": {"alpha": witnesses["theorem_B"]["alpha"], "eta_limit": witnesses["theorem_B"]["eta"], "fit_1_over_sigma": [float(x) for x in beta_B], "rows": fmt_rows(rows_B, witnesses["theorem_B"]["eta"])},
            "theorem_C": {"alpha": wC["alpha"], "eta_limit": wC["eta"], "fit_1_over_sigma": [float(x) for x in beta_C], "rows": fmt_rows(rows_C, wC["eta"])},
        },
    }
    return out


def main() -> None:
    t0 = time.time()
    witnesses = get_witnesses()
    if "theorem_C" not in witnesses:
        from cert_asymptotic import extract_theorem_c_witness
        witnesses["theorem_C"] = extract_theorem_c_witness()
    k4_res = validate_k4_direct()
    k5_res = validate_k5_direct_C()
    sweep_out = run_k_sweep(witnesses, k4_res=k4_res, k5_res=k5_res)
    sweep_out["elapsed_sec"] = round(time.time() - t0, 2)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(sweep_out, f, indent=2)
    print(f"\nSaved finite-k results to {RESULTS_PATH} in {sweep_out['elapsed_sec']:.2f}s")


if __name__ == "__main__":
    main()
