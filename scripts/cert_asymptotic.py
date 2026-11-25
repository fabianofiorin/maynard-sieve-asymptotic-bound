"""Step 5: Rigorous Arb certification of Theorem A (shift + Q), Theorem B (shift + P2), and Theorem C (shift + Lambda_9)."""

import argparse
import datetime
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, localcontext
from fractions import Fraction
from functools import lru_cache

import numpy as np
from flint import acb, arb, arb_mat, ctx, fmpq


import os as _os, sys as _sys
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _SCRIPT_DIR not in _sys.path:
    _sys.path.insert(0, _SCRIPT_DIR)

CERT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cert_asymptotic.json"))
WITNESS_PATH = os.path.join(os.path.dirname(__file__), "witness_cache.json")

THM_C_ALPHA = "-523/2000"
THM_C_WEIGHT_MAX = 9
THM_C_WEIGHT_DEGS = {0: 20, 2: 14, 3: 12, 4: 10, 5: 8, 6: 8, 7: 6, 8: 6, 9: 5}


def outward(x: arb, rounding, digits: int = 50) -> str:
    """Exact decimal outward rounding of an Arb midpoint/endpoint."""
    man, exp = x.mid().man_exp()
    q = Fraction(int(man)) * Fraction(2) ** int(exp)
    with localcontext() as c:
        c.prec = digits
        c.rounding = rounding
        return str(Decimal(q.numerator) / Decimal(q.denominator))


def exact_frac(s: str) -> arb:
    """Parse exact decimal or 'p/q' rational string in current Arb precision."""
    q = Fraction(s)
    return arb(q.numerator) / arb(q.denominator)


def compute_U_moments_certified(max_n: int, alpha_str: str, panels: list[float], prec_bits: int, tol_exp: int) -> list[arb]:
    """Rigorous Arb enclosure of U_n(alpha) = (n!/pi) Re int_0^inf exp(z(gamma+alpha+log z))/z^{n+1} dtau."""
    ctx.prec = prec_bits
    g = arb.const_euler()
    alpha = exact_frac(alpha_str)
    shift = g + alpha
    pi = arb.pi()
    T = arb(panels[-1])
    tol = arb(10) ** (-tol_exp)

    U = []
    for n in range(max_n + 1):
        def f(t, analytic, n=n):
            z = 1 + acb(0, 1) * t
            return (z * (shift + z.log(analytic=analytic))).exp() / (z ** (n + 1))

        total = acb(0)
        for a, b in zip(panels, panels[1:]):
            total += acb.integral(f, arb(a), arb(b), rel_tol=tol, abs_tol=tol)
        assert total.is_finite(), f"Arb integration failed at n={n}"
        tail_mag = (2 * (shift + 1).exp() * math.factorial(n) * (-pi * T / 2).exp() / (pi ** 2 * (T ** n))).upper()
        tail_ball = arb(0).union(tail_mag).union(-tail_mag)
        U.append(total.real * math.factorial(n) / pi + tail_ball)
    return U


def _integ_U_worker(args: tuple[int, str, list[float], int, int]) -> tuple[int, int, int, int, int]:
    """Compute n!-scaled U_n(alpha) enclosure in a worker process and return exact (lo_p, lo_q, hi_p, hi_q)."""
    n, alpha_str, panels, prec_bits, tol_exp = args
    ctx.prec = prec_bits
    g = arb.const_euler()
    alpha = exact_frac(alpha_str)
    shift = g + alpha
    pi = arb.pi()
    T = arb(panels[-1])
    tol = arb(10) ** (-tol_exp)
    fn = arb(math.factorial(n))

    def f(t, analytic):
        z = 1 + acb(0, 1) * t
        return fn * (z * (shift + z.log(analytic=analytic))).exp() / (z ** (n + 1))

    total = acb(0)
    for a, b in zip(panels, panels[1:]):
        total += acb.integral(f, arb(a), arb(b), rel_tol=tol, abs_tol=tol)
    assert total.is_finite(), f"Arb integration failed at n={n}"
    tail_mag = (2 * (shift + 1).exp() * fn * (-pi * T / 2).exp() / (pi ** 2 * (T ** n))).upper()
    tail_ball = arb(0).union(tail_mag).union(-tail_mag)
    u_ball = total.real / pi + tail_ball
    lo_q = u_ball.lower().fmpq()
    hi_q = u_ball.upper().fmpq()
    return (n, int(lo_q.p), int(lo_q.q), int(hi_q.p), int(hi_q.q))


def compute_U_moments_certified_parallel(
    max_n: int, alpha_str: str, panels: list[float], prec_bits: int, tol_exp: int, max_workers: int = 4
) -> list[arb]:
    """Compute rigorous Arb enclosures of U_0..U_{max_n} in parallel using n!-scaled acb.integral."""
    ctx.prec = prec_bits
    tasks = [(n, alpha_str, panels, prec_bits, tol_exp) for n in range(max_n + 1)]
    results: dict[int, arb] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for n, lp, lq, hp, hq in pool.map(_integ_U_worker, tasks):
            results[n] = arb(fmpq(lp, lq)).union(arb(fmpq(hp, hq)))
    return [results[n] for n in range(max_n + 1)]


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


def is_pos_def_arb(A: arb_mat) -> bool:
    """Prove positive definiteness of symmetric arb_mat via interval LDL^T."""
    n = A.nrows()
    L = [[arb(0)] * n for _ in range(n)]
    piv = []
    for j in range(n):
        p = A[j, j] - sum((L[j][k] ** 2 * piv[k] for k in range(j)), arb(0))
        if not (p > 0):
            return False
        piv.append(p)
        for i in range(j + 1, n):
            L[i][j] = (A[i, j] - sum((L[i][k] * L[j][k] * piv[k] for k in range(j)), arb(0))) / p
    return True


def quad_form(v: list[arb], A: arb_mat) -> arb:
    """Evaluate v^T A v rigorously in Arb using C-speed matrix multiplication."""
    n = len(v)
    v_col = arb_mat(n, 1, v)
    return (v_col.transpose() * A * v_col)[0, 0]


def build_A_forms(U: list[arb], deg: int, alpha: arb) -> tuple[arb_mat, arb_mat]:
    """Build Gram I and net D matrices for Theorem A."""
    H = [sum((arb(1) / j for j in range(1, n + 1)), arb(0)) for n in range(2 * deg + 2)]
    I_mat = arb_mat([[U[i + j] for j in range(deg + 1)] for i in range(deg + 1)])
    D_mat = arb_mat([
        [(2 * H[i + j] + 2 + alpha - H[i] - H[j]) * U[i + j] - 2 * U[i + j + 1] for j in range(deg + 1)]
        for i in range(deg + 1)
    ])
    return I_mat, D_mat


def build_B_forms(U: list[arb], d0: int, d1: int, alpha: arb) -> tuple[arb_mat, arb_mat]:
    """Build block Gram I and net D matrices for Theorem B (shift + P2)."""
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


def build_C_forms(
    U: list[arb], blocks: list[tuple[tuple[int, ...], int]], alpha: arb
) -> tuple[arb_mat, arb_mat]:
    """Build block Gram I and net D matrices for Theorem C (shift + Lambda_W symmetric hierarchy)."""
    max_u = 0
    for pa, da in blocks:
        for pb, db in blocks:
            L_ab = tuple(sorted(pa + pb, reverse=True))
            max_u = max(max_u, da + db + max(W_part(L_ab).keys()) + 1)
    assert max_u < len(U), f"Need U[{max_u}], have {len(U) - 1}"

    H = [arb(0)]
    for m in range(1, max_u + 5):
        H.append(H[-1] + arb(1) / m)
    fac = [arb(math.factorial(m)) for m in range(max_u + 10)]

    mI_cache: dict[tuple[int, tuple[int, ...]], arb] = {}
    mL_cache: dict[tuple[int, tuple[int, ...]], arb] = {}

    def mom_I(n: int, L_tup: tuple[int, ...]) -> arb:
        ck = (n, L_tup)
        if ck not in mI_cache:
            acc = arb(0)
            for r, c in W_part(L_tup).items():
                acc += c * (fac[n] / fac[n + r]) * U[n + r]
            mI_cache[ck] = acc
        return mI_cache[ck]

    def mom_L(n: int, L_tup: tuple[int, ...]) -> arb:
        ck = (n, L_tup)
        if ck not in mL_cache:
            acc = arb(0)
            for r, c in W_part(L_tup).items():
                acc += c * (fac[n] / fac[n + r]) * ((H[n] + 1 + alpha) * U[n + r] - U[n + r + 1])
            mL_cache[ck] = acc
        return mL_cache[ck]

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

    D_mat = arb_mat(N, N)
    for r in range(N):
        for c in range(N):
            D_mat[r, c] = M_mat[r, c] + M_mat[c, r] - alpha * I_mat[r, c]
    return I_mat, D_mat


def extract_theorem_c_witness(U_in: list[arb] | None = None, prec_bits: int = 768) -> dict:
    """Compute 60-digit witness for Theorem C via Arb shifted inverse iteration and save to witness_cache.json."""
    t0 = time.time()
    blocks = [(p, THM_C_WEIGHT_DEGS[sum(p)]) for p in partitions_ge2(THM_C_WEIGHT_MAX)]
    max_u = max(
        da + db + max(W_part(tuple(sorted(pa + pb, reverse=True))).keys()) + 1
        for pa, da in blocks
        for pb, db in blocks
    )
    if U_in is None:
        from finite_k_check import compute_U_alpha_fast_gl
        panels_gl = [0, 0.5, 1.5, 3, 6, 10, 16, 24, 35, 50, 70, 100, 140, 190, 250]
        U_use = compute_U_alpha_fast_gl(max_u, THM_C_ALPHA, panels_gl, n_gl=96, work_dps=250)
    else:
        U_use = U_in

    ctx.prec = prec_bits
    alpha_arb = exact_frac(THM_C_ALPHA)
    U_mid = [u.mid() for u in U_use]
    I_mat, D_mat = build_C_forms(U_mid, blocks, alpha_arb)
    N = I_mat.nrows()

    # Midpoint LDL^T -> Cholesky factor C -> whitened inverse iteration
    Lm = [[arb(0)] * N for _ in range(N)]
    piv = []
    for j in range(N):
        p = I_mat[j, j].mid()
        for k in range(j):
            p = (p - Lm[j][k] * Lm[j][k] * piv[k]).mid()
        assert float(p) > 0.0, f"Witness LDL^T failed at column {j}"
        piv.append(p)
        for i in range(j + 1, N):
            s = I_mat[i, j].mid()
            for k in range(j):
                s = (s - Lm[i][k] * Lm[j][k] * piv[k]).mid()
            Lm[i][j] = (s / p).mid()

    C_mat = arb_mat(N, N)
    for j in range(N):
        sp = piv[j].sqrt().mid()
        C_mat[j, j] = sp
        for i in range(j + 1, N):
            C_mat[i, j] = (Lm[i][j] * sp).mid()

    D_mid = arb_mat([[D_mat[r, c].mid() for c in range(N)] for r in range(N)])
    Ci = C_mat.inv()
    W_arb = Ci * D_mid * Ci.transpose()
    W_np = np.array([[float(W_arb[r, c].mid()) for c in range(N)] for r in range(N)], dtype=np.float64)
    W_np = 0.5 * (W_np + W_np.T)
    vals, vecs = np.linalg.eigh(W_np)
    lam_f = float(vals[-1])

    y_vec = arb_mat(N, 1, [arb(repr(float(x))) for x in vecs[:, -1]])
    Eye = arb_mat(N, N)
    for j in range(N):
        Eye[j, j] = arb(1)
    W_mid = arb_mat([[W_arb[r, c].mid() for c in range(N)] for r in range(N)])
    A_sh = W_mid - arb(repr(lam_f + 1e-12)) * Eye
    lam_cur = arb(repr(lam_f))
    for _ in range(4):
        y_vec = A_sh.solve(y_vec)
        nrm = sum((y_vec[j, 0].mid() ** 2 for j in range(N)), arb(0)).sqrt().mid()
        y_vec = arb_mat(N, 1, [(y_vec[j, 0].mid() / nrm).mid() for j in range(N)])
        Wy = W_mid * y_vec
        lam_cur = sum((y_vec[j, 0].mid() * Wy[j, 0].mid() for j in range(N)), arb(0)).mid()
        A_sh = W_mid - (lam_cur + arb(10) ** (-50)) * Eye

    v_mat = Ci.transpose() * y_vec
    v0 = v_mat[0, 0].mid()
    v_strs = [(v_mat[j, 0].mid() / v0).str(60, radius=False) for j in range(N)]

    blocks_out = []
    off = 0
    for p, d in blocks:
        blocks_out.append({
            "partition": list(p),
            "deg": d,
            "coeffs": v_strs[off : off + d + 1],
        })
        off += d + 1

    w_c = {
        "alpha": THM_C_ALPHA,
        "weight_max": THM_C_WEIGHT_MAX,
        "N": N,
        "eta": float(lam_cur),
        "blocks": blocks_out,
    }
    if os.path.exists(WITNESS_PATH):
        with open(WITNESS_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = {}
    cache["theorem_C"] = w_c
    with open(WITNESS_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    print(f"[witness_C] Generated 60-digit witness (N={N}, eta={float(lam_cur):.8f}) in {time.time() - t0:.2f}s")
    return w_c


def format_enclosures(U: list[arb]) -> list[dict]:
    """Serialize Arb moment enclosures cleanly as outward mid/rad strings."""
    out = []
    for n, u in enumerate(U):
        out.append({
            "n": n,
            "lower": outward(u.lower(), ROUND_FLOOR, 40),
            "upper": outward(u.upper(), ROUND_CEILING, 40),
            "rad": f"{float(u.rad()):.3e}",
        })
    return out


def certify_entry(entry_name: str, witness: dict, prec_bits: int = 600, tol_exp: int = 115) -> dict:
    """Certify entry A or B in Arb and return JSON dictionary."""
    t0 = time.time()
    panels = [0, 1, 4, 12, 35, 90, 200]
    ctx.prec = prec_bits

    if entry_name == "A":
        alpha_exact = "-3367/10000"
        deg = int(witness["deg"])
        U = compute_U_moments_certified(2 * deg + 1, alpha_exact, panels, prec_bits, tol_exp)
        alpha_arb = exact_frac(alpha_exact)
        I_mat, D_mat = build_A_forms(U, deg, alpha_arb)
        assert is_pos_def_arb(I_mat), "Entry A Gram matrix I_mat failed interval LDL^T"
        v_strs = [str(s) for s in witness["c"]]
        v = [exact_frac(s) for s in v_strs]
        coeffs_field = v_strs
        extra_meta = {"family": "shift_poly", "degree": deg}
    elif entry_name == "B":
        alpha_exact = "-649/2500"
        d0 = int(witness["d0"])
        d1 = int(witness["d1"])
        max_n = max(2 * d0 + 1, d0 + d1 + 2, 2 * d1 + 4)
        U = compute_U_moments_certified(max_n, alpha_exact, panels, prec_bits, tol_exp)
        alpha_arb = exact_frac(alpha_exact)
        I_mat, D_mat = build_B_forms(U, d0, d1, alpha_arb)
        assert is_pos_def_arb(I_mat), "Entry B Gram matrix I_mat failed interval LDL^T"
        q0_strs = [str(s) for s in witness["q0"]]
        q1_strs = [str(s) for s in witness["q1"]]
        v = [exact_frac(s) for s in q0_strs] + [exact_frac(s) for s in q1_strs]
        coeffs_field = {"q0": q0_strs, "q1": q1_strs}
        extra_meta = {"family": "shift_P2", "d0": d0, "d1": d1}
    elif entry_name == "C":
        alpha_exact = THM_C_ALPHA
        weight_max = int(witness["weight_max"])
        blocks = [(tuple(b["partition"]), int(b["deg"])) for b in witness["blocks"]]
        max_n = max(
            da + db + max(W_part(tuple(sorted(pa + pb, reverse=True))).keys()) + 1
            for pa, da in blocks
            for pb, db in blocks
        )
        U = compute_U_moments_certified_parallel(max_n, alpha_exact, panels, prec_bits, tol_exp, max_workers=4)
        alpha_arb = exact_frac(alpha_exact)
        I_mat, D_mat = build_C_forms(U, blocks, alpha_arb)
        assert is_pos_def_arb(I_mat), "Entry C Gram matrix I_mat failed interval LDL^T"
        v = []
        coeffs_field = []
        for b in witness["blocks"]:
            c_strs = [str(s) for s in b["coeffs"]]
            v.extend(exact_frac(s) for s in c_strs)
            coeffs_field.append({
                "partition": list(b["partition"]),
                "deg": int(b["deg"]),
                "coeffs": c_strs,
            })
        extra_meta = {
            "family": f"shift_sym_W{weight_max}",
            "weight_max": weight_max,
            "N": len(v),
            "blocks": [{"partition": list(p), "deg": d} for p, d in blocks],
        }
    else:
        raise ValueError(f"Unknown entry {entry_name}")

    I_val = quad_form(v, I_mat)
    D_val = quad_form(v, D_mat)
    assert I_val > 0 and I_val.is_finite() and D_val.is_finite(), "Non-positive or infinite Rayleigh denominator"
    eta_ball = D_val / I_val
    c_star_ball = -eta_ball

    eta_lo = outward(eta_ball.lower(), ROUND_FLOOR, 50)
    eta_hi = outward(eta_ball.upper(), ROUND_CEILING, 50)
    c_star_up = outward(c_star_ball.upper(), ROUND_CEILING, 35)
    elapsed = time.time() - t0
    print(
        f"[{entry_name}] CERTIFIED in {elapsed:.2f}s (prec={prec_bits} bits): "
        f"eta in [{eta_lo[:22]}, {eta_hi[:22]}], c*_upper = {c_star_up[:22]}"
    )

    return {
        **extra_meta,
        "alpha": alpha_exact,
        "coefficients": coeffs_field,
        "T": panels[-1],
        "panels": panels,
        "prec_bits": prec_bits,
        "tol_exp": tol_exp,
        "U_enclosures": format_enclosures(U),
        "I_val": {"lower": outward(I_val.lower(), ROUND_FLOOR, 40), "upper": outward(I_val.upper(), ROUND_CEILING, 40)},
        "D_val": {"lower": outward(D_val.lower(), ROUND_FLOOR, 40), "upper": outward(D_val.upper(), ROUND_CEILING, 40)},
        "eta_lower": eta_lo,
        "eta_upper": eta_hi,
        "c_star_upper": c_star_up,
        "baseline": "0.273572",
        "elapsed_sec": round(elapsed, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", choices=["A", "B", "C", "both", "all"], default="C")
    parser.add_argument("--gen-witness-c", action="store_true", help="Only generate/refresh theorem_C witness cache")
    args = parser.parse_args()

    if args.gen_witness_c:
        extract_theorem_c_witness()
        return

    with open(WITNESS_PATH, "r", encoding="utf-8") as f:
        witnesses = json.load(f)

    if os.path.exists(CERT_PATH):
        with open(CERT_PATH, "r", encoding="utf-8") as f:
            cert = json.load(f)
    else:
        cert = {"baseline": "0.273572", "entries": {}}

    orig_A = json.loads(json.dumps(cert["entries"]["A"])) if "A" in cert.get("entries", {}) else None
    orig_B = json.loads(json.dumps(cert["entries"]["B"])) if "B" in cert.get("entries", {}) else None

    if args.entry == "both":
        entries_to_run = ["A", "B"]
    elif args.entry == "all":
        entries_to_run = ["A", "B", "C"]
    else:
        entries_to_run = [args.entry]

    for name in entries_to_run:
        if name == "C" and "theorem_C" not in witnesses:
            witnesses["theorem_C"] = extract_theorem_c_witness()
        w_key = f"theorem_{name}"
        if name == "C":
            cert["entries"][name] = certify_entry(name, witnesses[w_key], prec_bits=768, tol_exp=145)
        else:
            cert["entries"][name] = certify_entry(name, witnesses[w_key])

    if "A" not in entries_to_run and orig_A is not None:
        assert cert["entries"]["A"] == orig_A, "Entry A was unexpectedly modified!"
    if "B" not in entries_to_run and orig_B is not None:
        assert cert["entries"]["B"] == orig_B, "Entry B was unexpectedly modified!"

    cert["baseline"] = "0.273572"
    cert["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    os.makedirs(os.path.dirname(CERT_PATH), exist_ok=True)
    with open(CERT_PATH, "w", encoding="utf-8") as f:
        json.dump(cert, f, indent=2)
    print(f"Saved certificate to {CERT_PATH}")


if __name__ == "__main__":
    main()
