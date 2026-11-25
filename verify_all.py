#!/usr/bin/env python3
"""Master reproducibility verifier for the Arb-certified asymptotic M_k archive."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
import os
import re
import subprocess
import time


ROOT = os.path.dirname(os.path.abspath(__file__))

EXPECTED_SHA256 = {
    "cert_asymptotic.json": "541e5d4fd45aff5905b38aa30dd5b77cc4d9f11a360823946498f26e80e02921",
    "verify_asymptotic.py": "e3721c23fdeaec99c95d9a6d34ae83ee9516410f6ce035d6cf53efaf53970a78",
}

SUPPORT_SCRIPTS = (
    "cert_asymptotic.py",
    "finite_k_check.py",
    "limit_xspace.py",
    "negative_checks.py",
    "powersum_screen.py",
    "screen_thm_c.py",
    "shifted_analysis.py",
)


def sha256_file(rel_path: str) -> str:
    """Compute SHA-256 hex digest for a repository-relative file path."""
    full_path = os.path.join(ROOT, rel_path)
    h = hashlib.sha256()
    with open(full_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(cmd: list[str], env: dict[str, str], timeout_s: int = 180) -> tuple[int, str, float]:
    """Run a child command and return (returncode, combined_stdout_stderr, elapsed_sec)."""
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
    )
    elapsed = time.perf_counter() - t0
    return proc.returncode, proc.stdout, elapsed


def check_support_scripts(env: dict[str, str]) -> tuple[bool, str, float]:
    """Verify from ROOT that all relative paths and sibling imports under scripts/ resolve cleanly."""
    script_rels = [os.path.join("scripts", name) for name in SUPPORT_SCRIPTS]
    for rel in script_rels:
        if not os.path.isfile(os.path.join(ROOT, rel)):
            return False, f"Missing support script: {rel}", 0.0

    code = (
        "import importlib.util, os, sys; "
        "sys.dont_write_bytecode = True; "
        f"names = {list(SUPPORT_SCRIPTS)!r}; "
        "sdir = os.path.abspath('scripts'); "
        "sys.path.insert(0, sdir); "
        "[importlib.util.spec_from_file_location(n[:-3], os.path.join(sdir, n)).loader.exec_module("
        "importlib.util.module_from_spec(importlib.util.spec_from_file_location(n[:-3], os.path.join(sdir, n)))) "
        "for n in names]; "
        "import cert_asymptotic; "
        "assert os.path.isfile(cert_asymptotic.CERT_PATH), cert_asymptotic.CERT_PATH; "
        "print('SCRIPTS_IMPORT_OK')"
    )
    rc, out, dt = run_cmd([sys.executable, "-B", "-c", code], env=env, timeout_s=60)
    ok = (rc == 0) and ("SCRIPTS_IMPORT_OK" in out)
    return ok, out, dt


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify certified asymptotic bounds and repository integrity.")
    parser.add_argument(
        "--run-supplementary",
        action="store_true",
        help="Also execute supplementary numerical scripts (scripts/negative_checks.py).",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["OMP_NUM_THREADS"] = "4"
    env["OPENBLAS_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"

    t_start = time.perf_counter()
    results: list[dict] = []
    all_passed = True

    cert_rel = "cert_asymptotic.json"
    verifier_rel = "verify_asymptotic.py"

    with open(os.path.join(ROOT, cert_rel), "r", encoding="utf-8") as f:
        cert = json.loads(f.read(), parse_float=str)

    rc, out, dt_asymp = run_cmd(
        [sys.executable, "-B", verifier_rel, cert_rel, "--entry", "all"],
        env=env,
        timeout_s=180,
    )
    asymp_ok = rc == 0 and ("OVERALL VERDICT: PASS" in out)
    if not asymp_ok:
        all_passed = False
        print(out)

    for entry_name in ("A", "B", "C"):
        m_pass = re.search(rf"\[{entry_name}\]\s+PASS\s+in\s+([0-9.]+)s", out)
        entry_dt = float(m_pass.group(1)) if m_pass else dt_asymp
        entry_ok = asymp_ok and (m_pass is not None)
        if not entry_ok:
            all_passed = False
        c_up = cert["entries"][entry_name]["c_star_upper"]
        c_up_20 = str(Decimal(c_up).quantize(Decimal("1e-20"), rounding=ROUND_CEILING))
        results.append({
            "artifact": f"{cert_rel} [Entry {entry_name}]",
            "verdict": "PASS" if entry_ok else "FAIL",
            "bound": f"c* <= {c_up_20}",
            "runtime_s": entry_dt,
            "sha256_path": cert_rel,
        })

    scripts_ok, scripts_out, dt_scripts = check_support_scripts(env)
    if not scripts_ok:
        all_passed = False
        print(scripts_out)
    results.append({
        "artifact": "scripts/*.py (7 modules)",
        "verdict": "PASS" if scripts_ok else "FAIL",
        "bound": "relative paths & imports verified",
        "runtime_s": dt_scripts,
        "sha256_path": verifier_rel,
    })

    if args.run_supplementary:
        rc_s, out_s, dt_s = run_cmd(
            [sys.executable, "-B", os.path.join("scripts", "negative_checks.py")],
            env=env,
            timeout_s=180,
        )
        ok_s = rc_s == 0
        if not ok_s:
            all_passed = False
            print(out_s)
        results.append({
            "artifact": "scripts/negative_checks.py",
            "verdict": "PASS" if ok_s else "FAIL",
            "bound": "supplementary microscale checks",
            "runtime_s": dt_s,
            "sha256_path": verifier_rel,
        })

    tracked_files = [cert_rel, verifier_rel]
    sha_map = {p: sha256_file(p) for p in tracked_files}
    for path_key, exp_hex in EXPECTED_SHA256.items():
        if sha_map[path_key] != exp_hex:
            all_passed = False
            print(f"SHA-256 MISMATCH for {path_key}: got {sha_map[path_key]}, expected {exp_hex}")

    print("=" * 108)
    print(f"{'Artifact':34s} | {'Verdict':7s} | {'Certified Bound / Check':34s} | {'Time':6s} | {'SHA-256'}")
    print("-" * 108)
    for row in results:
        art = row["artifact"]
        sha_p = sha_map[row["sha256_path"]]
        print(f"{art:34s} | {row['verdict']:7s} | {row['bound']:34s} | {row['runtime_s']:5.2f}s | {sha_p[:12]}...")
    print("=" * 108)
    print("\nCanonical SHA-256 Checksums:")
    for path_key, digest in sha_map.items():
        status = "MATCH" if digest == EXPECTED_SHA256[path_key] else "MISMATCH"
        print(f"  {digest}  {path_key}  [{status}]")

    total_dt = time.perf_counter() - t_start
    overall = "PASS" if all_passed else "FAIL"
    print(f"\nMASTER VERIFICATION: {overall} ({len(results)} checks in {total_dt:.2f}s)")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
