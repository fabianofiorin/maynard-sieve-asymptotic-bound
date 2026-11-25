# Asymptotic Lower Bounds on Maynard's Sieve Functional and Microscale Uniqueness

**Author:** Fabiano Fiorin (`fabianofiorin06@gmail.com`)

Companion manuscripts (English and Italian), Arb-certified rational witnesses, and independent verification software for *"Asymptotic lower bounds on Maynard's sieve functional and microscale uniqueness"*.

---

## Overview of Main Results

Let $M_k := \sup_F R_k(F)$ ($R_k(F) := k J(F)/I(F)$) denote Maynard's $k$-dimensional sieve variational functional on the unit simplex $\mathcal{R}_k$. This repository accompanies two main theorems and their prime-gap corollaries:

1. **Microscale Uniqueness (Theorem 1):** Among regular reciprocal-type microscale kernels $g(u)$ normalized to $m_0(g) = \int_0^\infty g(u)^2\,\mathrm{d}u = 1$ and $g(u) \sim u^{-1}$ as $u \to \infty$, the Cauchy kernel $h(u) = (1+u)^{-1}$ is uniquely optimal within the product-kernel and radial-polynomial class, with every normalized kernel $g \neq h$ (a.e.) incurring the strict penalty
   $$-\int_0^\infty (1+u)\bigl(g(u) - h(u)\bigr)^2\,\mathrm{d}u < 0$$
   (hence, among unnormalized kernels, $h$ is unique up to multiplicative scaling and positive argument dilation, $g(u) = C/(1+\theta u)$ a.e.).
2. **Certified Asymptotic Lower Bound on $M_k$ (Theorem 2 & Corollary 3):** Combining an $O(k)$ shift of the reciprocal scale $a = k(\log k + \alpha)$ at $\alpha = -523/2000$ with the complete weight-graded hierarchy $\Lambda_9$ of symmetric power-sum products over all $30$ integer partitions of weight at most $9$ with parts at least $2$ ($N = 250$ polynomial parameters), we improve Lewulis's bound $M_k \ge \log k - 0.273572 + o(1)$ (`arXiv:2610.41113`) unconditionally as $k \to \infty$ to
   $$M_k \ge \log k - c^* + o(1), \qquad c^* \le 0.26098863656469156542.$$

| Method / Result | Bound on $\log k - M_k$ | Pure BV ($4e^{c^*}$) | EH ($2e^{c^*}$) |
|---|---|---|---|
| Lewulis (2026) ($\alpha = 0$, $\deg 20$) | $\le 0.273572 + o(1)$ | $5.2586081$ | $2.6293041$ |
| **Theorem 2 ($\alpha = -523/2000$, $\Lambda_9$, $N = 250$)** | **$\le 0.26098863656469156542 + o(1)$** | **$5.1928517$** | **$2.5964259$** |

Restricting to the Bombieri–Vinogradov level ($\theta = 1/2$) and conditionally assuming the Elliott–Halberstam conjecture ($\theta = 1$), Corollary 3 yields $H_m \le (5.1928517 + o(1))\,m\,e^{4m}$ and $H_m \le (2.5964259 + o(1))\,m\,e^{2m}$ as $m \to \infty$ (while unconditionally as $m \to \infty$, prime equidistribution beyond $\theta = 1/2$ gives smaller exponential growth rates down to $H_m = O(e^{3.8075 m})$).

---

## Repository Structure

```text
.
├── README.md                  # Overview, installation, verification commands, SHA-256 digests, and citation
├── LICENSE                    # MIT License (Copyright (c) 2026 Fabiano Fiorin)
├── requirements.txt           # Exact pinned Python package versions
├── cert_asymptotic.json       # Arb-certified rational witnesses (Entries A, B, C) and 50-digit enclosures
├── verify_asymptotic.py       # Independent 768-bit Arb Bernstein-ellipse Gauss-Legendre verifier
├── verify_all.py              # Master verification script (Entries A, B, C, SHA-256, and scripts/ check)
├── scripts/
│   ├── cert_asymptotic.py     # Generates and certifies cert_asymptotic.json in 768-bit Arb arithmetic
│   ├── finite_k_check.py      # k = 5 5D Duffy simplex cross-check and k = 10^3..10^8 Arb Laplace sweep
│   ├── limit_xspace.py        # Macroscopic x-space quadrature helper
│   ├── negative_checks.py     # Numerical illustrations of microscale uniqueness and power-law tails
│   ├── powersum_screen.py     # Bell-polynomial symmetric power-sum screening helper
│   ├── screen_thm_c.py        # Weight-graded partition hierarchy screening script
│   └── shifted_analysis.py    # Shifted radial degree sweep and Arb closed-form cross-checks
└── paper/
    ├── main.pdf               # Compiled English manuscript (PDF)
    ├── main.tex               # English LaTeX source (amsart)
    ├── main_it.pdf            # Compiled Italian manuscript (PDF)
    └── main_it.tex            # Italian LaTeX source (amsart + babel)
```

---

## Installation and Verification

Tested on Windows 11 with Python `3.13.17`, `python-flint 0.9.0`, `numpy 2.5.4`, `scipy 1.18.1`, and `mpmath 1.4.1`. Only `python-flint` is required to run the deterministic Arb verifier `verify_asymptotic.py`.

Install the pinned dependencies from the repository root:

```bash
pip install -r requirements.txt
```

To independently verify Theorem 2 (Entry C in `cert_asymptotic.json`) from scratch in $768$-bit Arb ball arithmetic (~24.5 s, peak RSS ~78 MB):

```bash
python verify_asymptotic.py cert_asymptotic.json --entry C
```

To run the complete verification suite across all archived entries (`A`, `B`, `C`), check `scripts/` relative imports, and validate canonical SHA-256 digests (~18–36 s on at most 4 CPU threads):

```bash
python verify_all.py
```

Unlike `scripts/cert_asymptotic.py` (which uses adaptive `acb.integral` for $U_0,\dots,U_{41}$ and integration by parts $L_n = U_{n+1} - (1+\alpha)U_n$), `verify_asymptotic.py` shares no code with the generator: it computes $U_0,\dots,U_{40}$ and direct log-weighted moments $L_0,\dots,L_{40}$ independently via $240$-node `arb.legendre_p_root` Gauss–Legendre quadrature with rigorous Bernstein-ellipse error enclosures and $[225, \infty)$ analytic tail bounds, certifies $I(\alpha) \succ 0$ via interval $LDL^\top$, checks the Rayleigh quotient lower bound, and executes $9/9$ adversarial tamper-rejection tests for Entry C.

---

## Canonical SHA-256 Digests

| Artifact | SHA-256 Digest |
|---|---|
| `cert_asymptotic.json` | `541e5d4fd45aff5905b38aa30dd5b77cc4d9f11a360823946498f26e80e02921` |
| `verify_asymptotic.py` | `e3721c23fdeaec99c95d9a6d34ae83ee9516410f6ce035d6cf53efaf53970a78` |

---

## Citation

```bibtex
@misc{fiorin2026asymptotic,
  author       = {Fiorin, Fabiano},
  title        = {Asymptotic lower bounds on {Maynard}'s sieve functional and microscale uniqueness},
  year         = {2026},
  howpublished = {Preprint and companion verification archive, https://doi.org/10.5281/zenodo.24528079}
}
```

## License

Released under the [MIT License](LICENSE).
