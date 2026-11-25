"""Step 1 (Theorem C): Systematic reproducible screen of Families 1, 2, and 3."""

import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np
from flint import arb, arb_mat, ctx
from scipy.optimize import minimize

import os as _os, sys as _sys
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _SCRIPT_DIR not in _sys.path:
    _sys.path.insert(0, _SCRIPT_DIR)

from finite_k_check import compute_U_alpha_fast_gl
from limit_xspace import EULER_GAMMA, XSpaceEvaluator, build_orth_poly_vals


OUT_PATH = os.path.join(os.path.dirname(__file__), "screen_thm_c_results.json")


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
def W_part(L: tuple[int, ...]) -> dict[int, int]:
    """Exact Poisson point process moment polynomial W_L(z) = sum c_r z^{-r} via set partitions."""
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
        for r, c in W_part(tuple(rem)).items():
            acc[r + k_pow] = acc.get(r + k_pow, 0) + k_coef * c
    return acc


def make_blocks(W: int, deg_by_wt: dict[int, int]) -> list[tuple[tuple[int, ...], int]]:
    """Pair each partition in Lambda_W with its polynomial degree from deg_by_wt."""
    return [(p, deg_by_wt[sum(p)]) for p in partitions_ge2(W)]


def solve_hierarchy_arb(
    U: list[arb], alpha_str: str, blocks: list[tuple[tuple[int, ...], int]], prec_bits: int = 512
) -> tuple[float, int, int, float]:
    """Solve complete symmetric-polynomial generalized eigenproblem using C-speed Arb Cholesky + float64 eigvalsh."""
    ctx.prec = prec_bits
    alpha_arb = arb(alpha_str)
    max_u = 0
    for pa, da in blocks:
        for pb, db in blocks:
            L = tuple(sorted(pa + pb, reverse=True))
            max_u = max(max_u, da + db + max(W_part(L).keys()) + 1)
    assert max_u < len(U), f"Need U[{max_u}], have {len(U) - 1}"

    H = [arb(0)]
    for n in range(1, max_u + 5):
        H.append(H[-1] + arb(1) / n)
    fac = [arb(math.factorial(n)) for n in range(max_u + 10)]

    mI: dict[tuple[int, tuple[int, ...]], arb] = {}
    mL: dict[tuple[int, tuple[int, ...]], arb] = {}

    def mom_I(n: int, L: tuple[int, ...]) -> arb:
        ck = (n, L)
        if ck not in mI:
            acc = arb(0)
            for r, c in W_part(L).items():
                acc += c * (fac[n] / fac[n + r]) * U[n + r]
            mI[ck] = acc
        return mI[ck]

    def mom_L(n: int, L: tuple[int, ...]) -> arb:
        ck = (n, L)
        if ck not in mL:
            acc = arb(0)
            for r, c in W_part(L).items():
                acc += c * (fac[n] / fac[n + r]) * ((H[n] + 1 + alpha_arb) * U[n + r] - U[n + r + 1])
            mL[ck] = acc
        return mL[ck]

    offs = [0]
    for _, d in blocks:
        offs.append(offs[-1] + d + 1)
    N = offs[-1]
    I_mat = arb_mat(N, N)
    M_mat = arb_mat(N, N)

    for bi, (pa, da) in enumerate(blocks):
        oi = offs[bi]
        for bj, (pb, db) in enumerate(blocks):
            oj = offs[bj]
            L_ab = tuple(sorted(pa + pb, reverse=True))
            m_pb = len(pb)
            sdict: dict[tuple[int, tuple[int, ...]], int] = {}
            for mask in range(1, 1 << m_pb):
                s = 0
                rem = []
                for idx in range(m_pb):
                    if (mask >> idx) & 1:
                        s += pb[idx]
                    else:
                        rem.append(pb[idx])
                L_sub = tuple(sorted(pa + tuple(rem), reverse=True))
                sdict[(s, L_sub)] = sdict.get((s, L_sub), 0) + 1
            shifts = [(coef, s, L_sub) for (s, L_sub), coef in sdict.items()]
            for i in range(da + 1):
                for j in range(db + 1):
                    n = i + j
                    iv = mom_I(n, L_ab)
                    I_mat[oi + i, oj + j] = iv
                    mv = mom_L(n, L_ab) - H[j] * iv
                    for coef, s, L_sub in shifts:
                        mv += coef * ((fac[j] * fac[s - 1]) / fac[j + s]) * mom_I(n + s, L_sub)
                    M_mat[oi + i, oj + j] = mv

    D_mat = M_mat + M_mat.transpose() - alpha_arb * I_mat

    # Midpoint LDL^T
    L_rows = [[arb(0)] * N for _ in range(N)]
    piv = []
    min_piv = 1e300
    max_piv = 0.0
    for j in range(N):
        s = I_mat[j, j].mid()
        for k in range(j):
            s = (s - L_rows[j][k] * L_rows[j][k] * piv[k]).mid()
        p = s.mid()
        pf = float(p)
        assert pf > 0, f"Non-positive midpoint pivot {pf} at col {j}"
        min_piv = min(min_piv, pf)
        max_piv = max(max_piv, pf)
        piv.append(p)
        L_rows[j][j] = arb(1)
        for i in range(j + 1, N):
            s = I_mat[i, j].mid()
            for k in range(j):
                s = (s - L_rows[i][k] * L_rows[j][k] * piv[k]).mid()
            L_rows[i][j] = (s / p).mid()

    C_mat = arb_mat(N, N)
    for j in range(N):
        sp = piv[j].sqrt().mid()
        for i in range(j, N):
            C_mat[i, j] = (L_rows[i][j] * sp).mid()

    D_mid = arb_mat([[D_mat[r, c].mid() for c in range(N)] for r in range(N)])
    Cinv = C_mat.inv()
    W_arb = Cinv * D_mid * Cinv.transpose()
    W_np = np.array([[float(W_arb[r, c].mid()) for c in range(N)] for r in range(N)], dtype=np.float64)
    W_np = 0.5 * (W_np + W_np.T)
    eigvals = np.linalg.eigvalsh(W_np)
    return float(eigvals[-1]), N, max_u, min_piv / max_piv


def _alpha_sweep_worker(args: tuple[str, list[float], dict[int, int]]) -> dict:
    a_str, panels, deg_map = args
    Ua = compute_U_alpha_fast_gl(42, a_str, panels, n_gl=64, work_dps=200)
    eta, _, _, _ = solve_hierarchy_arb(Ua, a_str, make_blocks(9, deg_map), prec_bits=512)
    return {"alpha": a_str, "eta": eta}


def screen_family_1() -> dict:
    """Screen Family 1: Complete Weight-<=W Symmetric Polynomial Hierarchy."""
    print("=== FAMILY 1: Complete Weight-<=W Symmetric Polynomial Hierarchy ===")
    panels = [0, 0.5, 1.5, 3, 6, 10, 16, 24, 35, 50, 70, 100, 140, 190, 250]

    U_A = compute_U_alpha_fast_gl(42, "-0.3367", panels, n_gl=64, work_dps=200)
    U_B = compute_U_alpha_fast_gl(46, "-0.2596", panels, n_gl=64, work_dps=200)
    U_262 = compute_U_alpha_fast_gl(46, "-0.262", panels, n_gl=64, work_dps=200)
    U_2615 = compute_U_alpha_fast_gl(46, "-0.2615", panels, n_gl=64, work_dps=200)

    configs = [
        ("W=0 (Thm A) deg(20)", "-0.3367", U_A, make_blocks(0, {0: 20}), 512),
        ("W<=2 (Thm B) deg(20,16)", "-0.2596", U_B, make_blocks(2, {0: 20, 2: 16}), 512),
        ("W<=4 (5 blk) deg(20,16,14,12)", "-0.262", U_262, make_blocks(4, {0: 20, 2: 16, 3: 14, 4: 12}), 512),
        ("W<=5 (7 blk) deg(20,16,14,12,10)", "-0.262", U_262, make_blocks(5, {0: 20, 2: 16, 3: 14, 4: 12, 5: 10}), 512),
        ("W<=6 (11 blk) deg(20,16,14,12,10,8)", "-0.262", U_262, make_blocks(6, {0: 20, 2: 16, 3: 14, 4: 12, 5: 10, 6: 8}), 512),
        ("W<=7 (15 blk) deg(20,14,12,8,8,6,6)", "-0.262", U_262, make_blocks(7, {0: 20, 2: 14, 3: 12, 4: 8, 5: 8, 6: 6, 7: 6}), 512),
        ("W<=8 (22 blk) deg(16,14,12,10,8,8,6,5)", "-0.262", U_262, make_blocks(8, {0: 16, 2: 14, 3: 12, 4: 10, 5: 8, 6: 8, 7: 6, 8: 5}), 512),
        ("W<=8 (22 blk) deg(20,14,12,10,8,8,6,6)", "-0.262", U_262, make_blocks(8, {0: 20, 2: 14, 3: 12, 4: 10, 5: 8, 6: 8, 7: 6, 8: 6}), 512),
        ("W<=9 (Thm C) deg(20,14,12,10,8,8,6,6,5)", "-0.2615", U_2615, make_blocks(9, {0: 20, 2: 14, 3: 12, 4: 10, 5: 8, 6: 8, 7: 6, 8: 6, 9: 5}), 512),
        ("W<=10 (42 blk) deg(20,14,12,10,8,8,6,5,4,4)", "-0.2615", U_2615, make_blocks(10, {0: 20, 2: 14, 3: 12, 4: 10, 5: 8, 6: 8, 7: 6, 8: 5, 9: 4, 10: 4}), 512),
    ]

    hierarchy_rows = []
    eta_B_ref = None
    for name, alpha_str, U_arr, blks, pb in configs:
        ts = time.time()
        eta, N, max_u, piv_rat = solve_hierarchy_arb(U_arr, alpha_str, blks, prec_bits=pb)
        if "Thm B" in name:
            eta_B_ref = eta
        row = {
            "name": name,
            "alpha": alpha_str,
            "prec_bits": pb,
            "n_blocks": len(blks),
            "N": N,
            "max_U": max_u,
            "eta": eta,
            "piv_ratio": piv_rat,
            "elapsed_s": round(time.time() - ts, 3),
        }
        hierarchy_rows.append(row)
        print(f"  {name:46s}: blk={len(blks):2d}, N={N:3d}, maxU={max_u:2d}, eta={eta:.8f}, piv_rat={piv_rat:.1e}")

    print("\n  -- Fine alpha sweep for W<=9 (20,14,12,10,8,8,6,6,5) --")
    deg_map9 = {0: 20, 2: 14, 3: 12, 4: 10, 5: 8, 6: 8, 7: 6, 8: 6, 9: 5}
    tasks = [(a_str, panels, deg_map9) for a_str in ("-0.264", "-0.262", "-0.2615", "-0.260")]
    with ProcessPoolExecutor(max_workers=4) as pool:
        alpha_sweep = list(pool.map(_alpha_sweep_worker, tasks))
    for item in alpha_sweep:
        print(f"    alpha={item['alpha']}: eta={item['eta']:.10f}")

    return {
        "eta_B_ref": eta_B_ref,
        "hierarchy": hierarchy_rows,
        "w9_alpha_sweep": alpha_sweep,
    }


def eval_linear_stats(
    ev: XSpaceEvaluator, alpha: float, stats_list: list[tuple[str, float | None]], degs: list[int]
) -> float:
    """Evaluate asymptotic eta for linear statistics {1, stat_1, ..., stat_m} via exact PPP transforms."""
    def psi_fn(st, w):
        kind, th = st
        if kind == "S":
            return w * (1.0 - np.exp(-th * w))
        if kind == "P2":
            return w ** 2
        if kind == "P3":
            return w ** 3
        raise ValueError(st)

    def I1_fn(st, z):
        kind, th = st
        if kind == "S":
            return np.log((z + th) / z)
        if kind == "P2":
            return 1.0 / z
        if kind == "P3":
            return 1.0 / (z ** 2)
        raise ValueError(st)

    def I2_fn(st1, st2, z):
        k1, t1 = st1
        k2, t2 = st2
        if k1 == "S" and k2 == "S":
            return 1.0 / z - 1.0 / (z + t1) - 1.0 / (z + t2) + 1.0 / (z + t1 + t2)
        if k1 == "P2" and k2 == "P2":
            return 2.0 / (z ** 3)
        if k1 == "P3" and k2 == "P3":
            return 24.0 / (z ** 5)
        if (k1 == "P2" and k2 == "P3") or (k1 == "P3" and k2 == "P2"):
            return 6.0 / (z ** 4)
        if k1 == "P2" and k2 == "S":
            return 1.0 / (z ** 2) - 1.0 / ((z + t2) ** 2)
        if k1 == "S" and k2 == "P2":
            return 1.0 / (z ** 2) - 1.0 / ((z + t1) ** 2)
        if k1 == "P3" and k2 == "S":
            return 2.0 / (z ** 3) - 2.0 / ((z + t2) ** 3)
        if k1 == "S" and k2 == "P3":
            return 2.0 / (z ** 3) - 2.0 / ((z + t1) ** 3)
        raise ValueError((st1, st2))

    m = len(stats_list)
    n_x = len(ev.x)
    dtau = 0.04
    nu0 = np.zeros(n_x)
    nu1 = np.zeros((m, n_x))
    nu2 = np.zeros((m, m, n_x))
    for idx, xx in enumerate(ev.x):
        y = xx - alpha
        c = max(1.0, np.exp(y - 1.0 - EULER_GAMMA))
        T = 45.0 + 13.0 * np.sqrt(c)
        n_pts = int(np.ceil(T / dtau))
        tau = (np.arange(n_pts) + 0.5) * dtau
        z = c + 1j * tau
        base = np.exp(z * np.log(z) + EULER_GAMMA * z - z * y)
        fac = dtau / np.pi
        nu0[idx] = float(np.real(base.sum()) * fac)
        i1s = [I1_fn(st, z) for st in stats_list]
        for a in range(m):
            nu1[a, idx] = float(np.real((base * i1s[a]).sum()) * fac)
        for a in range(m):
            for b in range(a, m):
                w_ab = I2_fn(stats_list[a], stats_list[b], z) + i1s[a] * i1s[b]
                v = float(np.real((base * w_ab).sum()) * fac)
                nu2[a, b, idx] = v
                nu2[b, a, idx] = v

    max_d = max(degs)
    w0 = nu0 * ev.w
    Qx, Qsh, _, _ = build_orth_poly_vals(ev.x, ev.WW, w0, max_d)
    AQ = Qx * np.log(ev.x)[None, :] + np.sum(
        ev.WT[None, :, :] * (Qsh - Qx[:, :, None]) / ev.WW[None, :, :], axis=2
    )
    B_list = []
    for st in stats_list:
        pv = psi_fn(st, ev.WW) / ev.WW
        B_list.append(np.sum(ev.WT[None, :, :] * (Qsh * pv[None, :, :]), axis=2))

    nb = 1 + m
    I_rows = [[None] * nb for _ in range(nb)]
    M_rows = [[None] * nb for _ in range(nb)]
    for i in range(nb):
        Qi = Qx[: degs[i] + 1]
        for j in range(nb):
            Qj = Qx[: degs[j] + 1]
            AQj = AQ[: degs[j] + 1]
            if i == 0 and j == 0:
                wij = w0
                I_rows[i][j] = (Qi * wij[None, :]) @ Qj.T
                M_rows[i][j] = (Qi * wij[None, :]) @ AQj.T
            elif i == 0 and j > 0:
                b = j - 1
                wij = nu1[b] * ev.w
                I_rows[i][j] = (Qi * wij[None, :]) @ Qj.T
                M_rows[i][j] = (Qi * wij[None, :]) @ AQj.T + (Qi * w0[None, :]) @ B_list[b][: degs[j] + 1].T
            elif i > 0 and j == 0:
                a = i - 1
                wij = nu1[a] * ev.w
                I_rows[i][j] = (Qi * wij[None, :]) @ Qj.T
                M_rows[i][j] = (Qi * wij[None, :]) @ AQj.T
            else:
                a = i - 1
                b = j - 1
                wij = nu2[a, b] * ev.w
                wa = nu1[a] * ev.w
                I_rows[i][j] = (Qi * wij[None, :]) @ Qj.T
                M_rows[i][j] = (Qi * wij[None, :]) @ AQj.T + (Qi * wa[None, :]) @ B_list[b][: degs[j] + 1].T

    I_blk = np.block(I_rows)
    M_blk = np.block(M_rows)
    D_blk = M_blk + M_blk.T
    ev_i, V_i = np.linalg.eigh(I_blk)
    keep = ev_i > 1e-13 * ev_i.max()
    P = V_i[:, keep] / np.sqrt(ev_i[keep])
    return float(np.max(np.linalg.eigvalsh(P.T @ D_blk @ P)) - alpha)


def screen_family_2() -> dict:
    """Screen Family 2: Damped linear statistics S_theta and hybrids."""
    print("\n=== FAMILY 2: Damped Linear Statistics S_theta & Hybrids (X=9.0, n_x=24, n_w=40) ===")
    ev = XSpaceEvaluator(max_deg=14, X=9.0, n_x=24, n_w=40)
    single_sweep = []
    for th in (0.1, 0.25, 0.5, 0.8, 1.2, 2.0):
        eta = eval_linear_stats(ev, -0.25, [("S", th)], [10, 8])
        single_sweep.append({"theta": th, "alpha": -0.25, "eta": eta})
        print(f"  {{1, S_{th:4.2f}}} (10,8) alpha=-0.25: eta={eta:.8f}")

    res_s1 = minimize(
        lambda p: -eval_linear_stats(ev, float(p[0]), [("S", float(p[1]))], [10, 8])
        if 0.05 < p[1] < 10.0
        else 1.0,
        [-0.22, 0.6],
        method="Nelder-Mead",
        options={"maxfev": 60, "xatol": 1e-3},
    )
    opt_single = {"alpha": float(res_s1.x[0]), "theta": float(res_s1.x[1]), "eta": float(-res_s1.fun)}
    print(f"  OPT {{1, S_theta}} (10,8): alpha={opt_single['alpha']:.4f}, theta={opt_single['theta']:.4f} => eta={opt_single['eta']:.8f}")

    multi_and_hybrids = []
    for name, st, degs, alpha_v in [
        ("{1, S_0.2, S_1.0}", [("S", 0.2), ("S", 1.0)], [10, 8, 8], -0.22),
        ("{1, S_0.2, S_0.8, S_3.0}", [("S", 0.2), ("S", 0.8), ("S", 3.0)], [10, 8, 8, 8], -0.22),
        ("{1, P2, S_1.2}", [("P2", None), ("S", 1.2)], [10, 8, 8], -0.22),
        ("{1, P2, P3, S_1.0}", [("P2", None), ("P3", None), ("S", 1.0)], [10, 8, 8, 8], -0.22),
    ]:
        eta = eval_linear_stats(ev, alpha_v, st, degs)
        multi_and_hybrids.append({"name": name, "degs": degs, "alpha": alpha_v, "eta": eta})
        print(f"  {name:28s} {degs} alpha={alpha_v}: eta={eta:.8f}")

    return {
        "single_sweep": single_sweep,
        "opt_single": opt_single,
        "multi_and_hybrids": multi_and_hybrids,
    }


def eval_phi_powersums(
    ev: XSpaceEvaluator, alpha: float, b: float, theta: float, blocks: list[tuple[int, int, int]], dtau: float = 0.06
) -> float:
    """Evaluate macro kernel phi(w) = 1 + b*(1 - (1 + theta*w)*exp(-theta*w)) paired with {1, P2, P3, P2^2}."""
    def delta_phi(z):
        J1 = z * np.log(z / (z + theta)) + theta
        J2 = z * np.log(z * (z + 2.0 * theta) / ((z + theta) ** 2)) + (theta ** 2) / (z + 2.0 * theta)
        return 2.0 * b * J1 + (b ** 2) * J2

    def phi_fn(w):
        return 1.0 + b * (1.0 - (1.0 + theta * w) * np.exp(-theta * w))

    max_p2 = max(bl[0] for bl in blocks) * 2
    max_p3 = max(bl[1] for bl in blocks) * 2

    def kappa_z(S: int, z):
        m = S - 2
        f0, f1, f2 = math.factorial(m), math.factorial(m + 1), math.factorial(m + 2)
        zp, z2 = z + theta, z + 2.0 * theta
        return (
            (1.0 + b) ** 2 * f0 / (z ** (m + 1))
            - 2.0 * b * (1.0 + b) * (f0 / (zp ** (m + 1)) + theta * f1 / (zp ** (m + 2)))
            + (b ** 2) * (f0 / (z2 ** (m + 1)) + 2.0 * theta * f1 / (z2 ** (m + 2)) + (theta ** 2) * f2 / (z2 ** (m + 3)))
        )

    needed = set()
    for p2a, p3a, _ in blocks:
        for p2b, p3b, _ in blocks:
            for a2 in range(p2b + 1):
                for a3 in range(p3b + 1):
                    needed.add((p2a + p2b - a2, p3a + p3b - a3))

    n_x = len(ev.x)
    nu_map = {k: np.zeros(n_x) for k in needed}
    for idx, xx in enumerate(ev.x):
        y = xx - alpha
        c = max(1.0, np.exp(y - 1.0 - EULER_GAMMA))
        T = 38.0 + 11.0 * np.sqrt(c)
        n_pts = int(np.ceil(T / dtau))
        tau = (np.arange(n_pts) + 0.5) * dtau
        z = c + 1j * tau
        base = np.exp(z * np.log(z) + EULER_GAMMA * z - z * y + delta_phi(z))
        C = {}
        for m2 in range(max_p2 + 1):
            for m3 in range(max_p3 + 1):
                if m2 == 0 and m3 == 0:
                    continue
                C[(m2, m3)] = kappa_z(2 * m2 + 3 * m3, z) / (math.factorial(m2) * math.factorial(m3))
        expC = {(0, 0): np.ones_like(z)}
        term = {(0, 0): np.ones_like(z)}
        for order in range(1, max_p2 + max_p3 + 1):
            nxt = {}
            for (a1, b1), v1 in term.items():
                for (a2, b2), v2 in C.items():
                    if a1 + a2 > max_p2 or b1 + b2 > max_p3:
                        continue
                    kk = (a1 + a2, b1 + b2)
                    nxt[kk] = nxt[kk] + v1 * v2 if kk in nxt else v1 * v2
            term = nxt
            invf = 1.0 / math.factorial(order)
            for kk, v in term.items():
                expC[kk] = expC[kk] + v * invf if kk in expC else v * invf
        fac = dtau / np.pi
        for kk in needed:
            nu_map[kk][idx] = float(
                np.real((base * expC[kk]).sum()) * math.factorial(kk[0]) * math.factorial(kk[1]) * fac
            )

    nu0 = nu_map[(0, 0)]
    max_d = max(bl[2] for bl in blocks)
    w0 = nu0 * ev.w
    Qx, Qsh, _, _ = build_orth_poly_vals(ev.x, ev.WW, w0, max_d)
    ph = phi_fn(ev.WW)
    AQ = Qx * np.log(ev.x)[None, :] + np.sum(
        ev.WT[None, :, :] * (Qsh * ph[None, :, :] - Qx[:, :, None]) / ev.WW[None, :, :], axis=2
    )
    B_cache = {}

    def get_B(s):
        if s not in B_cache:
            B_cache[s] = np.sum(ev.WT[None, :, :] * (Qsh * ((ev.WW ** (s - 1)) * ph)[None, :, :]), axis=2)
        return B_cache[s]

    nb = len(blocks)
    I_rows = [[None] * nb for _ in range(nb)]
    M_rows = [[None] * nb for _ in range(nb)]
    for i, (p2a, p3a, da) in enumerate(blocks):
        Qa = Qx[: da + 1]
        for j, (p2b, p3b, db) in enumerate(blocks):
            Qb = Qx[: db + 1]
            AQb = AQ[: db + 1]
            wij = nu_map[(p2a + p2b, p3a + p3b)] * ev.w
            I_rows[i][j] = (Qa * wij[None, :]) @ Qb.T
            M_ij = (Qa * wij[None, :]) @ AQb.T
            for a2 in range(p2b + 1):
                for a3 in range(p3b + 1):
                    if a2 == 0 and a3 == 0:
                        continue
                    coef = math.comb(p2b, a2) * math.comb(p3b, a3)
                    w_sub = nu_map[(p2a + p2b - a2, p3a + p3b - a3)] * ev.w
                    M_ij += coef * ((Qa * w_sub[None, :]) @ get_B(2 * a2 + 3 * a3)[: db + 1].T)
            M_rows[i][j] = M_ij

    I_blk = np.block(I_rows)
    M_blk = np.block(M_rows)
    D_blk = M_blk + M_blk.T
    ev_i, V_i = np.linalg.eigh(I_blk)
    keep = ev_i > 1e-13 * ev_i.max()
    P = V_i[:, keep] / np.sqrt(ev_i[keep])
    return float(np.max(np.linalg.eigvalsh(P.T @ D_blk @ P)) - alpha)


def screen_family_3() -> list[dict]:
    """Screen Family 3: Decreasing macro kernel phi(w) paired with {1}, {1,P2}, and {1,P2,P3,P2^2}."""
    print("\n=== FAMILY 3: Macro Kernel phi(w) = 1 + b*(1-(1+th*w)e^{-th*w}) (X=8.0, n_x=16, n_w=28) ===")
    ev = XSpaceEvaluator(max_deg=12, X=8.0, n_x=16, n_w=28)
    rows = []
    for name, blks in [
        ("{1} (10)", [(0, 0, 10)]),
        ("{1,P2} (10,8)", [(0, 0, 10), (1, 0, 8)]),
        ("{1,P2,P3,P2^2} (10,8,8,6)", [(0, 0, 10), (1, 0, 8), (0, 1, 8), (2, 0, 6)]),
    ]:
        e0 = eval_phi_powersums(ev, -0.255, 0.0, 1.0, blks)
        grid = []
        print(f"  {name:28s} b=0: eta={e0:.8f}")
        for b in (-0.7, -0.4, -0.2, 0.2):
            for th in (0.5, 1.2, 2.5):
                e = eval_phi_powersums(ev, -0.255, b, th, blks)
                diff = e - e0
                grid.append({"b": b, "theta": th, "eta": e, "diff_vs_b0": diff})
                print(f"    b={b:+.1f}, th={th:3.1f}: eta={e:.8f} (diff={diff:+.6f})")
        rows.append({"name": name, "eta_b0": e0, "grid": grid})
    return rows


def main() -> None:
    t0 = time.time()
    f1 = screen_family_1()
    f2 = screen_family_2()
    f3 = screen_family_3()
    elapsed = round(time.time() - t0, 2)
    out = {
        "family1_power_sums": f1,
        "family2_damped_linear": f2,
        "family3_macro_kernel": f3,
        "elapsed_sec": elapsed,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved screening results to {OUT_PATH} in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
