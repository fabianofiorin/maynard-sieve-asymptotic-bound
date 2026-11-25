"""Step 3': Screen power-sum basis blocks R in {1, P2, P3, P2^2, P2*P3} using Bell polynomials."""

import math
import time
import numpy as np
from scipy.optimize import minimize_scalar

import os as _os, sys as _sys
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _SCRIPT_DIR not in _sys.path:
    _sys.path.insert(0, _SCRIPT_DIR)

from limit_xspace import XSpaceEvaluator, eval_nu, build_orth_poly_vals, EULER_GAMMA


def bell_weights_coeffs(max_p: int, max_q: int) -> dict[tuple[int, int], list[tuple[int, float]]]:
    """Return dictionary (p, q) -> list of (r, coeff) such that W_{p,q}(z) = sum coeff * z^{-r}."""
    # Build bivariate Taylor polynomial of exp(sum kappa_{m,n} u^m v^n / (m! n!))
    # Represent series as dict (p, q) -> dict {power_of_inv_z: coeff}
    def mul_series(A, B, max_p, max_q):
        out = {}
        for (p1, q1), d1 in A.items():
            for (p2, q2), d2 in B.items():
                if p1 + p2 > max_p or q1 + q2 > max_q:
                    continue
                target = out.setdefault((p1 + p2, q1 + q2), {})
                for r1, c1 in d1.items():
                    for r2, c2 in d2.items():
                        target[r1 + r2] = target.get(r1 + r2, 0.0) + c1 * c2
        return out

    # C = sum_{(m,n) != (0,0)} kappa_{m,n}/(m! n!) u^m v^n
    C = {}
    for m in range(max_p + 1):
        for n in range(max_q + 1):
            if m == 0 and n == 0:
                continue
            r = 2 * m + 3 * n - 1
            coeff = float(math.factorial(2 * m + 3 * n - 2)) / (math.factorial(m) * math.factorial(n))
            C[(m, n)] = {r: coeff}

    # exp(C) = 1 + C + C^2/2! + ... up to order max_p + max_q
    max_ord = max_p + max_q
    expC = {(0, 0): {0: 1.0}}
    term = {(0, 0): {0: 1.0}}
    for k in range(1, max_ord + 1):
        term = mul_series(term, C, max_p, max_q)
        inv_fact = 1.0 / math.factorial(k)
        for pq, d in term.items():
            target = expC.setdefault(pq, {})
            for r, c in d.items():
                target[r] = target.get(r, 0.0) + c * inv_fact

    # Multiply (p, q) entry by p! * q! to convert Taylor coeff to moments
    result = {}
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            mult = float(math.factorial(p) * math.factorial(q))
            d = expC.get((p, q), {})
            result[(p, q)] = sorted((r, mult * c) for r, c in d.items() if c != 0.0)
    return result


def compute_nu_pq_grid(
    x: np.ndarray, alpha: float, max_p: int, max_q: int, w_coeffs: dict, dtau: float = 0.04
) -> dict[tuple[int, int], np.ndarray]:
    """Compute nu_{p,q}(x) for all needed (p, q) in one pass over Bromwich nodes."""
    max_r = max((r for lst in w_coeffs.values() for r, _ in lst), default=0)
    # Precompute I_r(x) = (1/pi) Re int_0^inf e^{z(gamma+alpha+log z) - z x} z^{-r} dtau for r = 0..max_r
    n_x = len(x)
    I_r = np.zeros((max_r + 1, n_x), dtype=np.float64)
    for idx, xx in enumerate(x):
        y = xx - alpha
        c = max(1.0, np.exp(y - 1.0 - EULER_GAMMA))
        T = 45.0 + 13.0 * np.sqrt(c)
        n_pts = int(np.ceil(T / dtau))
        tau = (np.arange(n_pts) + 0.5) * dtau
        z = c + 1j * tau
        base = np.exp(z * np.log(z) + EULER_GAMMA * z - z * y)
        inv_z = 1.0 / z
        cur = base
        for r in range(max_r + 1):
            I_r[r, idx] = float(np.real(cur.sum()) * dtau / np.pi)
            cur = cur * inv_z

    nu_pq = {}
    for (p, q), lst in w_coeffs.items():
        arr = np.zeros(n_x, dtype=np.float64)
        for r, coeff in lst:
            arr += coeff * I_r[r]
        nu_pq[(p, q)] = arr
    return nu_pq


def solve_blocks(
    ev: XSpaceEvaluator,
    blocks: list[tuple[int, int, int]],  # (p, q, deg_R)
    alpha: float,
    nu_pq: dict[tuple[int, int], np.ndarray],
) -> tuple[float, float]:
    """Assemble and solve block generalized eigenproblem. Returns (eta, min_eig_I)."""
    nu0 = nu_pq[(0, 0)]
    max_d = max(d for _, _, d in blocks)
    # Use common nu0-orthonormal polynomial basis on [0, X]
    w0 = nu0 * ev.w
    Qx, Qsh, _, _ = build_orth_poly_vals(ev.x, ev.WW, w0, max_d)
    AQ = Qx * np.log(ev.x)[None, :] + np.sum(
        ev.WT[None, :, :] * (Qsh - Qx[:, :, None]) / ev.WW[None, :, :],
        axis=2,
    )

    # Precompute B_{a,b} Q(x) = int_0^x Q(x - w) w^{2a + 3b - 1} dw for needed (a, b)
    B_cache = {}
    needed_ab = set()
    for _, pb, qb in [(0, pb, qb) for pb, qb, _ in blocks]:
        for a in range(pb + 1):
            for b in range(qb + 1):
                if a == 0 and b == 0:
                    continue
                needed_ab.add((a, b))
    for a, b in needed_ab:
        pow_w = ev.WW ** (2 * a + 3 * b - 1)
        B_cache[(a, b)] = np.sum(ev.WT[None, :, :] * (Qsh * pow_w[None, :, :]), axis=2)

    # Build block I and M
    n_blk = len(blocks)
    I_rows = [[None] * n_blk for _ in range(n_blk)]
    M_rows = [[None] * n_blk for _ in range(n_blk)]

    for i, (pa, qa, da) in enumerate(blocks):
        Qa = Qx[: da + 1]
        for j, (pb, qb, db) in enumerate(blocks):
            Qb = Qx[: db + 1]
            AQb = AQ[: db + 1]
            wij = nu_pq[(pa + pb, qa + qb)] * ev.w
            I_rows[i][j] = (Qa * wij[None, :]) @ Qb.T
            M_ij = (Qa * wij[None, :]) @ AQb.T
            for a in range(pb + 1):
                for b in range(qb + 1):
                    if a == 0 and b == 0:
                        continue
                    coef = math.comb(pb, a) * math.comb(qb, b)
                    w_sub = nu_pq[(pa + pb - a, qa + qb - b)] * ev.w
                    BQb = B_cache[(a, b)][: db + 1]
                    M_ij += coef * ((Qa * w_sub[None, :]) @ BQb.T)
            M_rows[i][j] = M_ij

    I_blk = np.block(I_rows)
    M_blk = np.block(M_rows)
    D_blk = M_blk + M_blk.T

    ev_i, V_i = np.linalg.eigh(I_blk)
    min_eig = float(ev_i.min())
    assert min_eig > 0.0, f"I_blk not positive definite! min_eig={min_eig}"
    keep = ev_i > 1e-12 * ev_i.max()
    P = V_i[:, keep] / np.sqrt(ev_i[keep])
    eigvals = np.linalg.eigvalsh(P.T @ D_blk @ P)
    return float(np.max(eigvals) - alpha), min_eig


def main() -> None:
    t0 = time.time()
    # Evaluate on X=8.0 and X=10.0 to check truncation sensitivity
    ev8 = XSpaceEvaluator(max_deg=14, X=8.0, n_x=24, n_w=36)
    ev10 = XSpaceEvaluator(max_deg=14, X=10.0, n_x=24, n_w=36)

    max_p, max_q = 4, 2
    w_coeffs = bell_weights_coeffs(max_p, max_q)

    configs = [
        ("shift-only (deg 10)", [(0, 0, 10)]),
        ("+ P2 (10, 8)", [(0, 0, 10), (1, 0, 8)]),
        ("+ P3 (10, 8)", [(0, 0, 10), (0, 1, 8)]),
        ("+ P2 + P3 (10, 8, 8)", [(0, 0, 10), (1, 0, 8), (0, 1, 8)]),
        ("+ P2 + P2^2 (10, 8, 6)", [(0, 0, 10), (1, 0, 8), (2, 0, 6)]),
        ("all {1,P2,P3,P2^2,P2P3} (10,8,8,6,6)", [(0, 0, 10), (1, 0, 8), (0, 1, 8), (2, 0, 6), (1, 1, 6)]),
    ]

    # Check fixed alpha = -0.337 on X=8.0 vs X=10.0
    print("=== X-truncation sensitivity check at alpha=-0.337 ===")
    nu8 = compute_nu_pq_grid(ev8.x, -0.337, max_p, max_q, w_coeffs)
    nu10 = compute_nu_pq_grid(ev10.x, -0.337, max_p, max_q, w_coeffs)
    base_eta = None
    for name, blks in configs:
        eta8, min8 = solve_blocks(ev8, blks, -0.337, nu8)
        eta10, min10 = solve_blocks(ev10, blks, -0.337, nu10)
        if base_eta is None:
            base_eta = eta8
        print(
            f"  {name:38s}: X=8 eta={eta8:.8f} (min_ev={min8:.2e}), "
            f"X=10 eta={eta10:.8f}, diff={abs(eta8-eta10):.2e}, gain={eta8 - base_eta:+.6f}"
        )

    # Jointly optimize alpha for each config on X=8.0
    print("\n=== Joint alpha optimization ===")
    base_opt = None
    for name, blks in configs:
        def obj(a):
            nu_a = compute_nu_pq_grid(ev8.x, a, max_p, max_q, w_coeffs)
            eta_val, _ = solve_blocks(ev8, blks, a, nu_a)
            return -eta_val

        res = minimize_scalar(obj, bounds=(-0.55, -0.15), method="bounded", options={"xatol": 2e-4})
        a_opt = float(res.x)
        eta_opt = float(-res.fun)
        if base_opt is None:
            base_opt = eta_opt
        print(f"  {name:38s}: alpha*={a_opt:.6f}, eta*={eta_opt:.8f}, gain vs shift={eta_opt - base_opt:+.6f}")

    # Also test higher degree (12, 10) for + P2 and + P2 + P3 at optimal alpha
    print("\n=== Higher degree check (12, 10) ===")
    for name, blks in [
        ("shift-only (deg 12)", [(0, 0, 12)]),
        ("+ P2 (12, 10)", [(0, 0, 12), (1, 0, 10)]),
        ("+ P2 + P3 (12, 10, 10)", [(0, 0, 12), (1, 0, 10), (0, 1, 10)]),
        ("+ P2 + P2^2 (12, 10, 8)", [(0, 0, 12), (1, 0, 10), (2, 0, 8)]),
    ]:
        nu_a = compute_nu_pq_grid(ev8.x, -0.337, max_p, max_q, w_coeffs)
        eta_val, min_ev = solve_blocks(ev8, blks, -0.337, nu_a)
        print(f"  {name:28s}: alpha=-0.337, eta={eta_val:.8f}, min_ev={min_ev:.2e}")

    print(f"Total runtime: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
