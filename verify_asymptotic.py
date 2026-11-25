#!/usr/bin/env python3
"""
Independent From-Scratch Adversarial Verifier for Asymptotic Lower Bounds (Theorems A, B & C).
Zero code reuse from work/asymp/. Uses an independent geometric panel grid and rigorous
simultaneous Bernstein-ellipse Gauss-Legendre quadrature in Arb for U_n and direct L_n
(WITHOUT the IBP identity).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from fractions import Fraction
from functools import lru_cache
from typing import Any

import flint

arb = flint.arb
acb = flint.acb
arb_mat = flint.arb_mat
ctx = flint.ctx
fmpq = flint.fmpq


class VerificationError(Exception):
    """Raised when a certificate fails any adversarial mathematical or schema check."""


def parse_exact_fraction(val: Any, name: str) -> Fraction:
    """Parse an integer or exact decimal/rational string into an exact Fraction."""
    if isinstance(val, bool) or val is None or isinstance(val, float):
        raise VerificationError(f"Invalid type for '{name}': {type(val).__name__}")
    if isinstance(val, str):
        s = val.strip()
        if not s or "nan" in s.lower() or "inf" in s.lower():
            raise VerificationError(f"Invalid string for '{name}': '{val}'")
        try:
            return Fraction(s)
        except (ValueError, ZeroDivisionError) as exc:
            raise VerificationError(f"Unparseable fraction for '{name}': '{val}'") from exc
    if isinstance(val, int):
        return Fraction(val, 1)
    raise VerificationError(f"Unsupported type for '{name}': {type(val).__name__}")


def frac_to_arb(fr: Fraction) -> arb:
    """Convert exact Fraction to Arb ball at current ctx.prec."""
    return arb(fmpq(fr.numerator, fr.denominator))


def verify_pos_def_ldl(A: arb_mat) -> bool:
    """Rigorous interval LDL^T test: returns True iff every pivot is strictly > 0."""
    n = A.nrows()
    if n != A.ncols() or n == 0:
        return False
    L = [[arb(0)] * n for _ in range(n)]
    piv = []
    for j in range(n):
        p = A[j, j] - sum((L[j][k] ** 2 * piv[k] for k in range(j)), arb(0))
        if not (p > 0):
            return False
        piv.append(p)
        for i in range(j + 1, n):
            if not (A[i, j] - A[j, i]).contains(arb(0)):
                return False
            L[i][j] = (A[i, j] - sum((L[i][k] * L[j][k] * piv[k] for k in range(j)), arb(0))) / p
    return True


def eval_quad_form(v: list[arb], A: arb_mat) -> arb:
    """Evaluate v^T A v rigorously using C-speed Arb matrix multiplication."""
    n = len(v)
    v_col = arb_mat(n, 1, v)
    return (v_col.transpose() * A * v_col)[0, 0]


def compute_independent_moments(
    max_n: int, alpha_fr: Fraction, panels: list[float], prec_bits: int, m_gl: int = 210, n_arc: int = 32
) -> tuple[list[arb], list[arb]]:
    """
    Independently compute U_n(alpha) and direct log-weighted L_n(alpha) (NO IBP identity!)
    via rigorous simultaneous Bernstein-ellipse Gauss-Legendre quadrature in Arb, using explicit
    filled-ellipse holomorphicity checks, boundary maximum-modulus enclosures on E_rho, and
    the exact Chebyshev error bound |int_a^b f - Q_m| <= h * 8 * M / ((rho - 1) * rho^(2m - 1)).
    """
    ctx.prec = prec_bits
    g = arb.const_euler()
    pi = arb.pi()
    alpha = frac_to_arb(alpha_fr)
    shift = g + alpha
    T = arb(panels[-1])
    if not (T >= 10):
        raise VerificationError(f"Verifier cutoff T={T} must be >= 10")

    nodes = [arb.legendre_p_root(m_gl, i, weight=True) for i in range(m_gl)]
    if not sum((w for _, w in nodes), arb(0)).contains(arb(2)):
        raise VerificationError("Gauss-Legendre weights do not sum to 2")

    facts = [arb(math.factorial(n)) for n in range(max_n + 1)]
    acc_U = [arb(0)] * (max_n + 1)
    acc_L = [arb(0)] * (max_n + 1)
    err_U = [arb(0)] * (max_n + 1)
    err_L = [arb(0)] * (max_n + 1)

    for a_f, b_f in zip(panels, panels[1:]):
        a = arb(a_f)
        b = arb(b_f)
        mu = (a + b) / 2
        h = (b - a) / 2
        hf = float(h.mid())
        muf = float(mu.mid())
        if a_f == 0.0:
            B_t = min(3.0, 0.55 / hf)
        else:
            K_max = (muf - 0.25 * a_f) / hf
            B_from_K = math.sqrt(max(0.01, K_max * K_max - 1.0))
            B_from_exp = 0.45 * m_gl / (hf * max(1.5, math.log(b_f + 2.0)))
            B_t = min(3.0, B_from_K, B_from_exp)
        rho_f = B_t + math.sqrt(B_t * B_t + 1.0)
        if not (rho_f > 1.05):
            raise VerificationError(f"Panel [{a_f}, {b_f}] too wide for Bernstein ellipse (rho={rho_f})")

        rho = arb(repr(round(rho_f, 4)))
        K = (rho + 1 / rho) / 2
        B = (rho - 1 / rho) / 2
        if not ((1 - h * B > 0) or (mu - h * K > 0)):
            raise VerificationError(f"Panel [{a_f}, {b_f}] Bernstein ellipse may touch log branch cut")

        geom_fac = h * 8 / ((rho - 1) * (rho ** (2 * m_gl - 1)))

        for x_gl, w_gl in nodes:
            t = mu + h * x_gl
            wt = h * w_gl
            z = 1 + acb(0, 1) * t
            lz = z.log()
            g_lz = g + lz
            cur = (z * (shift + lz)).exp() / z
            inv_z = 1 / z
            for n in range(max_n + 1):
                acc_U[n] += wt * cur.real
                acc_L[n] += wt * (g_lz * cur).real
                cur *= inv_z

        M_U = [arb(0)] * (max_n + 1)
        M_L = [arb(0)] * (max_n + 1)
        dth = 2 * pi / n_arc
        for k_arc in range(n_arc):
            th_iv = (dth * k_arc).union(dth * (k_arc + 1))
            u_iv = acb(K * th_iv.cos(), B * th_iv.sin())
            z_iv = 1 + acb(0, 1) * (mu + h * u_iv)
            lz_iv = z_iv.log()
            z_lo = abs(z_iv).lower()
            if not (lz_iv.is_finite() and z_lo > 0):
                raise VerificationError(f"Branch cut or singularity touched on panel [{a_f}, {b_f}] arc {k_arc}")
            g_lz_abs = abs(g + lz_iv).upper()
            base_abs = abs((z_iv * (shift + lz_iv)).exp()).upper()
            inv_z_hi = (1 / z_lo).upper()
            zp = inv_z_hi
            for n in range(max_n + 1):
                m_u = (base_abs * zp).upper()
                m_l = (m_u * g_lz_abs).upper()
                if m_u > M_U[n]:
                    M_U[n] = m_u
                if m_l > M_L[n]:
                    M_L[n] = m_l
                zp = (zp * inv_z_hi).upper()

        for n in range(max_n + 1):
            err_U[n] = (err_U[n] + geom_fac * M_U[n]).upper()
            err_L[n] = (err_L[n] + geom_fac * M_L[n]).upper()

    C_T = g + pi / 2 + (1 + T).log()
    exp_pref = (shift + 1).exp()
    exp_tail = (-pi * T / 2).exp()
    U_out: list[arb] = []
    L_out: list[arb] = []
    for n in range(max_n + 1):
        scale = facts[n] / pi
        base_tail = scale * exp_pref * (T ** (-n)) * exp_tail
        qU = (scale * err_U[n] + base_tail * (2 / pi)).upper()
        qL = (scale * err_L[n] + base_tail * (2 * C_T / pi + 4 / (pi ** 2 * T))).upper()
        U_out.append(acc_U[n] * scale + arb(0).union(qU).union(-qU))
        L_out.append(acc_L[n] * scale + arb(0).union(qL).union(-qL))
    return U_out, L_out


def indep_all_partitions_ge2(w_max: int) -> set[tuple[int, ...]]:
    """Independently generate all partitions of weight 0 <= w <= w_max with parts >= 2."""
    out: set[tuple[int, ...]] = {()}

    def rec(rem: int, max_p: int, cur: tuple[int, ...]) -> None:
        if rem == 0:
            out.add(cur)
            return
        for p in range(min(rem, max_p), 1, -1):
            rec(rem - p, p, cur + (p,))

    for w in range(2, w_max + 1):
        rec(w, w, ())
    return out


@lru_cache(None)
def indep_bell_weights(L: tuple[int, ...]) -> dict[int, int]:
    """Independently compute exact set-partition moment coefficients sum c_r z^{-r} for sorted multiset L."""
    if not L:
        return {0: 1}
    first, rest = L[0], L[1:]
    m = len(rest)
    acc: dict[int, int] = {}
    for mask in range(1 << m):
        s = first
        rem = []
        for idx in range(m):
            if (mask >> idx) & 1:
                s += rest[idx]
            else:
                rem.append(rest[idx])
        k_pow = s - 1
        k_coef = math.factorial(s - 2)
        for r, c in indep_bell_weights(tuple(rem)).items():
            acc[r + k_pow] = acc.get(r + k_pow, 0) + k_coef * c
    return acc


def validate_and_parse_C_blocks(entry: dict) -> list[tuple[tuple[int, ...], int, list[arb]]]:
    """Validate Theorem C schema, partition completeness, degrees, and exact coefficients."""
    w_max = entry.get("weight_max")
    if isinstance(w_max, bool) or not isinstance(w_max, int) or w_max < 7:
        raise VerificationError(f"Entry C invalid weight_max={w_max}")
    expected_family = f"shift_sym_W{w_max}"
    if entry.get("family") != expected_family:
        raise VerificationError(f"Entry C has wrong family name (expected '{expected_family}', got '{entry.get('family')}')")
    coeffs = entry.get("coefficients")
    if not isinstance(coeffs, list) or len(coeffs) == 0:
        raise VerificationError("Entry C coefficients must be a non-empty list of blocks")

    expected_parts = indep_all_partitions_ge2(w_max)
    seen_parts: set[tuple[int, ...]] = set()
    parsed: list[tuple[tuple[int, ...], int, list[arb]]] = []

    for b_idx, blk in enumerate(coeffs):
        if not isinstance(blk, dict):
            raise VerificationError(f"Entry C block[{b_idx}] is not a dict")
        raw_p = blk.get("partition")
        deg = blk.get("deg")
        raw_c = blk.get("coeffs")
        if not isinstance(raw_p, list) or isinstance(deg, bool) or not isinstance(deg, int) or deg < 0:
            raise VerificationError(f"Entry C block[{b_idx}] invalid partition or degree")
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 2 for x in raw_p):
            raise VerificationError(f"Entry C block[{b_idx}] partition parts must be ints >= 2")
        p_tup = tuple(sorted(raw_p, reverse=True))
        if list(raw_p) != list(p_tup):
            raise VerificationError(f"Entry C block[{b_idx}] partition {raw_p} is not sorted descending")
        if sum(p_tup) > w_max:
            raise VerificationError(f"Entry C block[{b_idx}] partition {p_tup} exceeds weight_max={w_max}")
        if p_tup in seen_parts:
            raise VerificationError(f"Entry C duplicate partition {p_tup}")
        seen_parts.add(p_tup)
        if not isinstance(raw_c, list) or len(raw_c) != deg + 1:
            raise VerificationError(f"Entry C block[{b_idx}] deg={deg} vs len(coeffs)={len(raw_c) if isinstance(raw_c, list) else None}")
        v_blk = [frac_to_arb(parse_exact_fraction(s, f"C.blk[{b_idx}].coeffs[{i}]")) for i, s in enumerate(raw_c)]
        parsed.append((p_tup, deg, v_blk))

    if seen_parts != expected_parts:
        missing = sorted(expected_parts - seen_parts)
        extra = sorted(seen_parts - expected_parts)
        raise VerificationError(f"Entry C incomplete partition set: missing={missing}, extra={extra}")

    if "blocks" in entry:
        meta_blocks = entry["blocks"]
        if not isinstance(meta_blocks, list) or len(meta_blocks) != len(parsed):
            raise VerificationError("Entry C blocks metadata length mismatch")
        for idx, (mb, (p_tup, deg, _)) in enumerate(zip(meta_blocks, parsed)):
            if tuple(mb.get("partition", [])) != p_tup or mb.get("deg") != deg:
                raise VerificationError(f"Entry C blocks[{idx}] metadata disagrees with coefficients")

    return parsed


def build_forms_from_UL(
    entry_name: str, entry: dict, U: list[arb], L_mom: list[arb], alpha: arb
) -> tuple[arb_mat, arb_mat, list[arb]]:
    """Construct Gram I and net D directly from U_n and L_n (no IBP identity)."""
    if entry_name == "A":
        if entry.get("family") != "shift_poly":
            raise VerificationError("Entry A has wrong family name")
        deg = int(entry["degree"])
        if deg < 0 or len(entry["coefficients"]) != deg + 1:
            raise VerificationError("Entry A degree / coefficients mismatch")
        v = [frac_to_arb(parse_exact_fraction(s, f"A.c[{i}]")) for i, s in enumerate(entry["coefficients"])]
        H = [sum((arb(1) / j for j in range(1, n + 1)), arb(0)) for n in range(2 * deg + 1)]
        I_mat = arb_mat([[U[i + j] for j in range(deg + 1)] for i in range(deg + 1)])
        D_mat = arb_mat([
            [(2 * H[i + j] - H[i] - H[j] - alpha) * U[i + j] - 2 * L_mom[i + j] for j in range(deg + 1)]
            for i in range(deg + 1)
        ])
        return I_mat, D_mat, v

    elif entry_name == "B":
        if entry.get("family") != "shift_P2":
            raise VerificationError("Entry B has wrong family name")
        d0 = int(entry["d0"])
        d1 = int(entry["d1"])
        coeffs = entry["coefficients"]
        if len(coeffs["q0"]) != d0 + 1 or len(coeffs["q1"]) != d1 + 1:
            raise VerificationError("Entry B degree / coefficients mismatch")
        v0 = [frac_to_arb(parse_exact_fraction(s, f"B.q0[{i}]")) for i, s in enumerate(coeffs["q0"])]
        v1 = [frac_to_arb(parse_exact_fraction(s, f"B.q1[{i}]")) for i, s in enumerate(coeffs["q1"])]
        v = v0 + v1

        max_m = max(2 * d0, d0 + d1 + 2, 2 * d1 + 3)
        H = [sum((arb(1) / j for j in range(1, m + 1)), arb(0)) for m in range(max_m + 5)]

        def inv_poch(n: int, s: int) -> arb:
            val = arb(1)
            for t in range(n + 1, n + s + 1):
                val /= t
            return val

        def J_ns(n: int, s: int) -> arb:
            return inv_poch(n, s) * (H[n] * U[n + s] - L_mom[n + s])

        N = d0 + 1 + d1 + 1
        I_mat = arb_mat(N, N)
        M_mat = arb_mat(N, N)
        o = d0 + 1
        for i in range(d0 + 1):
            for j in range(d0 + 1):
                n = i + j
                I_mat[i, j] = U[n]
                M_mat[i, j] = J_ns(n, 0) - H[j] * U[n]
        for i in range(d0 + 1):
            for j in range(d1 + 1):
                n = i + j
                I_val = inv_poch(n, 1) * U[n + 1]
                I_mat[i, o + j] = I_val
                I_mat[o + j, i] = I_val
                M_mat[i, o + j] = J_ns(n, 1) - H[j] * I_val + U[n + 2] / ((j + 1) * (j + 2))
                M_mat[o + j, i] = J_ns(n, 1) - H[i] * I_val
        for i in range(d1 + 1):
            for j in range(d1 + 1):
                n = i + j
                I_val = inv_poch(n, 2) * U[n + 2] + 2 * inv_poch(n, 3) * U[n + 3]
                I_mat[o + i, o + j] = I_val
                M_mat[o + i, o + j] = (J_ns(n, 2) + 2 * J_ns(n, 3)) - H[j] * I_val + U[n + 3] / ((j + 1) * (j + 2) * (n + 3))

        D_mat = M_mat + M_mat.transpose() - alpha * I_mat
        return I_mat, D_mat, v

    elif entry_name == "C":
        parsed_blocks = validate_and_parse_C_blocks(entry)
        blocks = [(p, d) for p, d, _ in parsed_blocks]
        v = [c for _, _, vc in parsed_blocks for c in vc]

        max_m = max(
            da + db + max(indep_bell_weights(tuple(sorted(pa + pb, reverse=True))).keys())
            for pa, da in blocks
            for pb, db in blocks
        )
        if max_m >= len(U) or max_m >= len(L_mom):
            raise VerificationError(f"Entry C needs moments up to {max_m}, got {len(U) - 1}")

        H = [sum((arb(1) / j for j in range(1, m + 1)), arb(0)) for m in range(max_m + 5)]
        fac = [arb(math.factorial(m)) for m in range(max_m + 10)]

        def inv_poch(n: int, r: int) -> arb:
            val = arb(1)
            for t in range(n + 1, n + r + 1):
                val /= t
            return val

        mI_cache: dict[tuple[int, tuple[int, ...]], arb] = {}
        mJ_cache: dict[tuple[int, tuple[int, ...]], arb] = {}

        def mom_I(n: int, L_tup: tuple[int, ...]) -> arb:
            ck = (n, L_tup)
            if ck not in mI_cache:
                acc = arb(0)
                for r, c in indep_bell_weights(L_tup).items():
                    acc += c * inv_poch(n, r) * U[n + r]
                mI_cache[ck] = acc
            return mI_cache[ck]

        def mom_J(n: int, L_tup: tuple[int, ...]) -> arb:
            """Direct log-weighted moment (NO IBP identity!)."""
            ck = (n, L_tup)
            if ck not in mJ_cache:
                acc = arb(0)
                for r, c in indep_bell_weights(L_tup).items():
                    acc += c * inv_poch(n, r) * (H[n] * U[n + r] - L_mom[n + r])
                mJ_cache[ck] = acc
            return mJ_cache[ck]

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
                        mv = mom_J(n, L_ab) - H[j] * iv
                        for coef, s, L_sub in shifts:
                            mv += coef * ((fac[j] * fac[s - 1]) / fac[j + s]) * mom_I(n + s, L_sub)
                        M_mat[oi + i, oj + j] = mv

        D_mat = M_mat + M_mat.transpose() - alpha * I_mat
        return I_mat, D_mat, v

    raise VerificationError(f"Unknown entry '{entry_name}'")


def verify_entry_with_moments(
    entry_name: str,
    entry: dict,
    U: list[arb],
    L_mom: list[arb],
    prec_bits: int,
    check_enclosures: bool = True,
) -> dict:
    """Verify certificate entry using supplied Arb moment enclosures (U, L_mom)."""
    ctx.prec = prec_bits
    alpha_fr = parse_exact_fraction(entry["alpha"], f"{entry_name}.alpha")
    alpha = frac_to_arb(alpha_fr)

    # Check stored U_enclosures overlap our independent moments if requested
    if check_enclosures:
        stored_U = entry.get("U_enclosures", [])
        if len(stored_U) < len(U):
            raise VerificationError(f"{entry_name}: insufficient U_enclosures in certificate")
        for n, u_ind in enumerate(U):
            lo_s = frac_to_arb(parse_exact_fraction(stored_U[n]["lower"], f"{entry_name}.U[{n}].lower"))
            hi_s = frac_to_arb(parse_exact_fraction(stored_U[n]["upper"], f"{entry_name}.U[{n}].upper"))
            if not (lo_s <= hi_s):
                raise VerificationError(f"{entry_name}: stored U[{n}] has inverted interval lower > upper")
            if not u_ind.overlaps(lo_s.union(hi_s)):
                raise VerificationError(f"{entry_name}: U[{n}] independent integral disagrees with stored enclosure")

    I_mat, D_mat, v = build_forms_from_UL(entry_name, entry, U, L_mom, alpha)
    if not verify_pos_def_ldl(I_mat):
        raise VerificationError(f"{entry_name}: Gram matrix I failed interval LDL^T positive definiteness")

    I_val = eval_quad_form(v, I_mat)
    D_val = eval_quad_form(v, D_mat)
    if not (I_val > 0 and I_val.is_finite() and D_val.is_finite()):
        raise VerificationError(f"{entry_name}: witness denominator I_val is not strictly positive or finite")

    eta_ball = D_val / I_val
    c_star_claimed = frac_to_arb(parse_exact_fraction(entry["c_star_upper"], f"{entry_name}.c_star_upper"))
    eta_lo_claimed = frac_to_arb(parse_exact_fraction(entry["eta_lower"], f"{entry_name}.eta_lower"))
    eta_hi_claimed = frac_to_arb(parse_exact_fraction(entry["eta_upper"], f"{entry_name}.eta_upper"))

    if not eta_ball.overlaps(eta_lo_claimed.union(eta_hi_claimed)):
        raise VerificationError(f"{entry_name}: eta_ball {eta_ball.str(25)} does not overlap claimed bracket")
    if not ((-eta_ball.lower()) <= c_star_claimed):
        raise VerificationError(
            f"{entry_name}: claimed c_star_upper={c_star_claimed.str(30)} is below proven upper bound {(-eta_ball.lower()).str(30)}"
        )
    baseline = frac_to_arb(parse_exact_fraction(entry.get("baseline", "0.273572"), "baseline"))
    if not (c_star_claimed < baseline):
        raise VerificationError(f"{entry_name}: c_star_upper does not improve baseline")

    return {"eta_ball": eta_ball, "I_val": I_val, "D_val": D_val}


def run_adversarial_tests(entry_name: str, entry: dict, U: list[arb], L_mom: list[arb], prec_bits: int) -> None:
    """Run >= 5 adversarial rejection tests per entry (must all raise VerificationError)."""
    ctx.prec = prec_bits
    alpha_fr = parse_exact_fraction(entry["alpha"], "alpha")

    def expect_reject(label: str, mod_entry: dict, mod_U=U, mod_L=L_mom, check_enc=False):
        try:
            verify_entry_with_moments(entry_name, mod_entry, mod_U, mod_L, prec_bits, check_enclosures=check_enc)
        except VerificationError:
            print(f"    [adversarial PASS] {entry_name}/{label}: correctly rejected")
            return
        raise AssertionError(f"Adversarial test {entry_name}/{label} unexpectedly PASSED!")

    # 1. Tampered optimistic c_star_upper claim (shrink c* by 1e-6)
    e1 = copy.deepcopy(entry)
    e1["c_star_upper"] = str(parse_exact_fraction(entry["c_star_upper"], "c") - Fraction(1, 10 ** 6))
    expect_reject("tampered_c_star_optimistic", e1)

    # 2. Wrong sign on c_star_upper (negative c*)
    e2 = copy.deepcopy(entry)
    e2["c_star_upper"] = "-" + entry["c_star_upper"].lstrip("+-")
    expect_reject("wrong_sign_c_star", e2)

    # 3. Wrong alpha (perturbs alpha by +0.05 without updating moments/claim)
    e3 = copy.deepcopy(entry)
    e3["alpha"] = str(alpha_fr + Fraction(1, 20))
    expect_reject("wrong_alpha", e3)

    # 4. Perturbed polynomial coefficient (violates claimed eta bracket / c_star_upper)
    e4 = copy.deepcopy(entry)
    if entry_name == "A":
        c0_new = parse_exact_fraction(e4["coefficients"][0], "c0") + Fraction(1, 100)
        e4["coefficients"][0] = str(c0_new)
    elif entry_name == "B":
        q0_new = parse_exact_fraction(e4["coefficients"]["q0"][0], "q0") + Fraction(1, 100)
        e4["coefficients"]["q0"][0] = str(q0_new)
    else:
        c0_new = parse_exact_fraction(e4["coefficients"][0]["coeffs"][0], "C.c0") + Fraction(1, 100)
        e4["coefficients"][0]["coeffs"][0] = str(c0_new)
    expect_reject("perturbed_coefficient", e4)

    # 5. Corrupted tail enclosure mismatch in stored U_enclosures
    e5 = copy.deepcopy(entry)
    e5["U_enclosures"][0]["lower"] = str(parse_exact_fraction(e5["U_enclosures"][0]["lower"], "u0l") - Fraction(1, 1000))
    e5["U_enclosures"][0]["upper"] = str(parse_exact_fraction(e5["U_enclosures"][0]["upper"], "u0u") - Fraction(1, 1000))
    expect_reject("corrupted_U_enclosure_tail", e5, check_enc=True)

    # 5b. Inverted U_enclosure interval (upper < lower)
    e5b = copy.deepcopy(entry)
    e5b["U_enclosures"][0]["upper"] = str(parse_exact_fraction(e5b["U_enclosures"][0]["lower"], "u0l") - Fraction(1, 1000))
    expect_reject("inverted_U_enclosure", e5b, check_enc=True)

    # 6. Float64 uncertified parameter injection (must be rejected by parse_exact_fraction)
    e6 = copy.deepcopy(entry)
    e6["alpha"] = -0.25
    expect_reject("uncertified_float_alpha", e6)

    # 7. Severely truncated T < 10 in independent moment integration check
    try:
        compute_independent_moments(1, alpha_fr, [0.0, 5.0], prec_bits, m_gl=64)
        raise AssertionError("Truncated T < 10 was not rejected!")
    except VerificationError:
        print(f"    [adversarial PASS] {entry_name}/truncated_T_cutoff: correctly rejected")

    # 8. Entry C partition schema tampering (duplicate / missing partition or deg mismatch)
    if entry_name == "C":
        e9 = copy.deepcopy(entry)
        e9["coefficients"][1]["partition"] = [99]
        expect_reject("tampered_partition_schema", e9)


def verify_single_entry(cert: dict, entry_name: str, prec_bits: int | None = None, m_gl: int = 210) -> dict:
    """Run full independent verification + adversarial suite on one certificate entry."""
    t0 = time.time()
    entry = cert["entries"][entry_name]
    alpha_fr = parse_exact_fraction(entry["alpha"], f"{entry_name}.alpha")
    if entry_name == "A":
        max_n = 2 * int(entry["degree"])
    elif entry_name == "B":
        d0, d1 = int(entry["d0"]), int(entry["d1"])
        max_n = max(2 * d0, d0 + d1 + 2, 2 * d1 + 3)
    elif entry_name == "C":
        blocks_parsed = validate_and_parse_C_blocks(entry)
        blocks = [(p, d) for p, d, _ in blocks_parsed]
        max_n = max(
            da + db + max(indep_bell_weights(tuple(sorted(pa + pb, reverse=True))).keys())
            for pa, da in blocks
            for pb, db in blocks
        )
    else:
        raise VerificationError(f"Unknown entry '{entry_name}'")

    raw_pb = entry.get("prec_bits", 640)
    if isinstance(raw_pb, bool) or not isinstance(raw_pb, int):
        raise VerificationError(f"{entry_name}: invalid prec_bits={raw_pb}")
    pb = max(640, raw_pb) if prec_bits is None else prec_bits
    eff_m_gl = 240 if (pb >= 768 and m_gl == 210) else m_gl

    # Independent geometric panel grid ending at T = 225 (distinct from certifier's T = 200)
    ver_panels = [0.0, 0.35, 0.9, 2.0, 4.5, 10.0, 22.0, 45.0, 85.0, 145.0, 225.0]
    print(f"[{entry_name}] Computing independent U[0..{max_n}] & direct L[0..{max_n}] via Bernstein-GL (m_gl={eff_m_gl}, prec={pb}, T=225)...")
    U, L_mom = compute_independent_moments(max_n, alpha_fr, ver_panels, pb, m_gl=eff_m_gl)
    res = verify_entry_with_moments(entry_name, entry, U, L_mom, pb, check_enclosures=True)

    # Strict improvement check vs Theorem B when verifying Entry C alongside Entry B
    if entry_name == "C" and "B" in cert.get("entries", {}):
        ctx.prec = pb
        c_B = frac_to_arb(parse_exact_fraction(cert["entries"]["B"]["c_star_upper"], "B.c_star_upper"))
        c_C = frac_to_arb(parse_exact_fraction(entry["c_star_upper"], "C.c_star_upper"))
        if not ((-res["eta_ball"].lower()) < c_B and c_C < c_B):
            raise VerificationError("Entry C does not strictly improve Theorem B c_star_upper")

    print(
        f"[{entry_name}] Independent eta_ball = [{res['eta_ball'].lower().str(38)}, {res['eta_ball'].upper().str(38)}] "
        f"<= -c*_claimed (-{entry['c_star_upper']})"
    )
    n_adv = 9 if entry_name == "C" else 8
    print(f"[{entry_name}] Running adversarial rejection suite ({n_adv} tests)...")
    run_adversarial_tests(entry_name, entry, U, L_mom, pb)
    elapsed = time.time() - t0
    print(f"[{entry_name}] PASS in {elapsed:.2f}s")
    return {"eta_ball": res["eta_ball"], "elapsed_sec": elapsed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cert_path", nargs="?", default=os.path.join("sieve", "cert_asymptotic.json"))
    parser.add_argument("--entry", choices=["A", "B", "C", "both", "all"], default="all")
    args = parser.parse_args()

    with open(args.cert_path, "r", encoding="utf-8") as f:
        cert = json.loads(f.read(), parse_float=str)

    if args.entry == "both":
        names = ["A", "B"]
    elif args.entry == "all":
        names = ["A", "B", "C"]
    else:
        names = [args.entry]
    for name in names:
        verify_single_entry(cert, name)
    print("\nOVERALL VERDICT: PASS")


if __name__ == "__main__":
    main()
