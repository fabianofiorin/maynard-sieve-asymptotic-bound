"""Fast x-space floating-point evaluator for asymptotic variational limits."""

import numpy as np
from numpy.polynomial.legendre import leggauss

EULER_GAMMA = np.euler_gamma


def make_x_nodes(X: float = 7.5, n_per_panel: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Build piecewise Gauss-Legendre quadrature nodes and weights on [0, X]."""
    pts = [0.0, 1e-4, 1e-3, 1e-2, 0.05, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, X]
    pts = sorted(set([p for p in pts if p < X] + [X]))
    t, w = leggauss(n_per_panel)
    xs, ws = [], []
    for a, b in zip(pts, pts[1:]):
        xs.append(0.5 * (b - a) * t + 0.5 * (a + b))
        ws.append(0.5 * (b - a) * w)
    return np.concatenate(xs), np.concatenate(ws)


def eval_nu(
    x: np.ndarray,
    extra_fn=None,
    alpha: float = 0.0,
    dtau: float = 0.05,
) -> np.ndarray:
    """Evaluate limiting density nu(x) via adaptive saddle-point Bromwich midpoint rule on [0, T]."""
    if extra_fn is None:
        extra_fn = lambda z: np.zeros_like(z)
    out = np.empty_like(x, dtype=np.float64)
    for idx, xx in enumerate(x):
        y = xx - alpha
        c = max(1.0, np.exp(y - 1.0 - EULER_GAMMA))
        T = 40.0 + 12.0 * np.sqrt(c)
        n_pts = int(np.ceil(T / dtau))
        tau = (np.arange(n_pts) + 0.5) * dtau
        z = c + 1j * tau
        integrand = np.exp(z * np.log(z) + EULER_GAMMA * z - z * y + extra_fn(z))
        out[idx] = max(0.0, float(np.real(integrand.sum()) * dtau / np.pi))
    return out


def build_orth_poly_vals(
    x: np.ndarray, w_grid: np.ndarray, weights: np.ndarray, max_deg: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct weighted orthonormal polynomials on x and evaluate on x and x - WW via 3-term recurrence."""
    n_x = len(x)
    x_sh = x[:, None] - w_grid
    Qx = np.zeros((max_deg + 1, n_x))
    Qsh = np.zeros((max_deg + 1, n_x, w_grid.shape[1]))
    alphas = np.zeros(max_deg)
    betas = np.zeros(max_deg + 1)

    norm0 = np.sqrt(np.sum(weights))
    Qx[0] = 1.0 / norm0
    Qsh[0] = 1.0 / norm0
    betas[0] = norm0

    for j in range(max_deg):
        v_x = x * Qx[j]
        v_sh = x_sh * Qsh[j]
        a_j = np.sum(weights * Qx[j] * v_x)
        alphas[j] = a_j
        v_x = v_x - a_j * Qx[j] - (betas[j] * Qx[j - 1] if j > 0 else 0.0)
        v_sh = v_sh - a_j * Qsh[j] - (betas[j] * Qsh[j - 1] if j > 0 else 0.0)
        # Re-orthogonalize once against previous polynomials for numerical stability
        for m in range(j + 1):
            proj = np.sum(weights * Qx[m] * v_x)
            v_x = v_x - proj * Qx[m]
            v_sh = v_sh - proj * Qsh[m]
        b_next = np.sqrt(np.sum(weights * v_x * v_x))
        betas[j + 1] = b_next
        Qx[j + 1] = v_x / b_next
        Qsh[j + 1] = v_sh / b_next

    return Qx, Qsh, alphas, betas


class XSpaceEvaluator:
    """Precomputed x-space quadrature grid with nu-orthonormal polynomial basis."""

    def __init__(self, max_deg: int = 20, X: float = 7.5, n_x: int = 32, n_w: int = 48):
        self.max_deg = max_deg
        self.X = X
        self.x, self.w = make_x_nodes(X=X, n_per_panel=n_x)
        t_w, w_w = leggauss(n_w)
        self.WW = self.x[:, None] * 0.5 * (t_w[None, :] + 1.0)
        self.WT = self.x[:, None] * 0.5 * w_w[None, :]
        self._cache = {}

    def _get_basis(self, deg: int, alpha: float, extra_key, extra_fn, nu_vals: np.ndarray | None):
        key = (deg, alpha, extra_key)
        if nu_vals is None and key in self._cache:
            return self._cache[key]
        if nu_vals is None:
            nu_vals = eval_nu(self.x, extra_fn=extra_fn, alpha=alpha)
        weights = nu_vals * self.w
        Qx, Qsh, _, _ = build_orth_poly_vals(self.x, self.WW, weights, deg)
        res = (weights, Qx, Qsh)
        if extra_key is not None or extra_fn is None:
            self._cache[key] = res
        return res

    def build_matrices(
        self,
        deg: int,
        alpha: float = 0.0,
        extra_fn=None,
        phi_fn=None,
        nu_vals: np.ndarray | None = None,
        extra_key=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build orthonormal-basis Gram matrix I (~Identity) and symmetric form D (without -alpha shift)."""
        assert deg <= self.max_deg
        weights, Qx, Qsh = self._get_basis(deg, alpha, extra_key, extra_fn, nu_vals)
        ph = np.ones_like(self.WW) if phi_fn is None else phi_fn(self.WW)
        AQ = Qx * np.log(self.x)[None, :] + np.sum(
            self.WT[None, :, :] * (Qsh * ph[None, :, :] - Qx[:, :, None]) / self.WW[None, :, :],
            axis=2,
        )
        I_mat = (Qx * weights[None, :]) @ Qx.T
        M = (Qx * weights[None, :]) @ AQ.T
        D_mat = M + M.T
        return I_mat, D_mat

    def solve_eta(
        self,
        deg: int,
        alpha: float = 0.0,
        extra_fn=None,
        phi_fn=None,
        nu_vals: np.ndarray | None = None,
        extra_key=None,
    ) -> tuple[float, np.ndarray]:
        """Solve max_{Q} (2<Q, AQ> / <Q, Q> - alpha) in the nu-orthonormal basis."""
        I_mat, D_mat = self.build_matrices(
            deg, alpha=alpha, extra_fn=extra_fn, phi_fn=phi_fn, nu_vals=nu_vals, extra_key=extra_key
        )
        ev, V = np.linalg.eigh(I_mat)
        keep = ev > 1e-13 * ev.max()
        P = V[:, keep] / np.sqrt(ev[keep])
        eigvals, eigvecs = np.linalg.eigh(P.T @ D_mat @ P)
        top_idx = int(np.argmax(eigvals))
        c_orth = P @ eigvecs[:, top_idx]
        return float(eigvals[top_idx] - alpha), c_orth


def main() -> None:
    ev = XSpaceEvaluator(max_deg=20, X=7.5, n_x=32, n_w=48)
    nu0 = eval_nu(ev.x, alpha=0.0)
    print("x-space baseline check (alpha=0, phi=1):")
    for d in (0, 2, 4, 10, 15, 20):
        eta_val, _ = ev.solve_eta(d, alpha=0.0, nu_vals=nu0)
        print(f"  deg {d:2d}: eta = {eta_val:.10f}")


if __name__ == "__main__":
    main()
