#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate synthetic circular data on [0, 2*pi) using a JSON config file.

Components:
- Two Gaussian peaks
- One extended Gaussian plateau (wide Gaussian)
- Constant background level

Samples are drawn from a probability distribution proportional to
bg + g1 + g2 + g_plateau, then saved to CSV.

Usage
-----
python examples/gen_sample.py --config examples/sample.config

Optionally override output path:
python examples/gen_sample.py --config examples/sample.config --out examples/sample.csv
"""

import argparse
import json
import math
import sys
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd

TWOPI = 2.0 * math.pi


def wrap_0_2pi(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.mod(x, TWOPI)
    return x


def periodic_gaussian(t: np.ndarray, mu: float, sigma: float, kwrap: int = 3) -> np.ndarray:
    """Circular Gaussian by summing normal densities over wrap copies.

    Parameters
    ----------
    t : np.ndarray
        Evaluation grid in [0, 2pi)
    mu : float
        Mean angle in [0, 2pi)
    sigma : float
        Standard deviation (radians)
    kwrap : int
        Number of wrap images on each side to approximate periodicity
    """
    # Sum over images mu + m*2pi
    x = np.zeros_like(t, dtype=float)
    for m in range(-kwrap, kwrap + 1):
        d = t - (mu + m * TWOPI)
        x += np.exp(-0.5 * (d / sigma) ** 2)
    # Normalize constant dropped (only relative shape matters)
    return x


def periodic_generalized_normal(t: np.ndarray, mu: float, scale: float, beta: float, kwrap: int = 3) -> np.ndarray:
    """Circular generalized normal (exponential power): exp(-(|t-mu|/scale)^beta).

    - beta < 2 yields flatter (plateau-like) tops; beta > 2 sharp peaks.
    - scale > 0 is a spread parameter analogous to sigma.
    """
    x = np.zeros_like(t, dtype=float)
    b = max(1e-6, float(beta))
    s = max(1e-6, float(scale))
    for m in range(-kwrap, kwrap + 1):
        d = np.abs(t - (mu + m * TWOPI))
        x += np.exp(- (d / s) ** b)
    return x


def _load_json_with_comments(path: str) -> dict:
    """Load JSON allowing // and /* */ comments.

    This stripper preserves comment-like sequences inside string literals.
    """
    with open(path, "r", encoding="utf-8") as f:
        s = f.read()

    out = []
    i = 0
    n = len(s)
    in_str = False
    esc = False
    while i < n:
        ch = s[i]
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue

        # not in string
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue

        if ch == '/' and i + 1 < n:
            nxt = s[i + 1]
            # line comment
            if nxt == '/':
                i += 2
                while i < n and s[i] not in ('\n', '\r'):
                    i += 1
                continue
            # block comment
            if nxt == '*':
                i += 2
                while i + 1 < n and not (s[i] == '*' and s[i + 1] == '/'):
                    i += 1
                i = min(n, i + 2)
                continue

        out.append(ch)
        i += 1

    return json.loads(''.join(out))


def build_intensity(
    grid: np.ndarray,
    components: List[Dict[str, Any]],
    bg: float,
) -> np.ndarray:
    """Build intensity from a list of component dicts with keys a, mu, sigma."""
    I = np.full_like(grid, float(bg), dtype=float)
    for comp in components:
        a = float(comp.get("a", 0.0))
        if a == 0.0:
            continue
        mu = float(comp.get("mu", 0.0)) % TWOPI
        sigma = max(1e-6, float(comp.get("sigma", 0.1)))
        ctype = str(comp.get("type", "gauss")).lower()
        if ctype in ("gauss", "gaussian", "normal"):
            I += a * periodic_gaussian(grid, mu, sigma)
        elif ctype in ("generalized", "generalized_normal", "plateau"):
            beta = float(comp.get("beta", 1.0))
            I += a * periodic_generalized_normal(grid, mu, sigma, beta)
        else:
            # Fallback to gaussian if unknown type
            I += a * periodic_gaussian(grid, mu, sigma)
    return I


def sample_from_intensity(intensity: np.ndarray, grid: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n values in [0, 2pi) from discrete intensity defined on grid.

    We interpret intensity on uniform grid as a categorical distribution over bins.
    For each chosen bin, we add a small uniform jitter within the bin width for continuity.
    """
    I = np.asarray(intensity, dtype=float)
    I = np.maximum(I, 0.0)
    if not np.any(I > 0):
        raise ValueError("Intensity is all zeros; cannot sample.")
    p = I / I.sum()
    K = grid.size
    # assume uniform grid
    w = float(TWOPI / K)
    # centers at grid points; sample bin indices
    idx = rng.choice(K, size=int(n), replace=True, p=p)
    # jitter within bin
    jitter = rng.uniform(-0.5 * w, 0.5 * w, size=int(n))
    x = grid[idx] + jitter
    return wrap_0_2pi(x)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate synthetic circular mixture samples in [0, 2pi) from JSON config")
    parser.add_argument("-c", "--config", type=str, default="sample.config", help="Path to JSON config file")
    parser.add_argument("-o", "--out", type=str, default=None, help="Optional override for output CSV path")
    args = parser.parse_args(argv)

    # Load config (supports // and /* */ comments)
    cfg = _load_json_with_comments(args.config)

    n = int(cfg.get("n", 20000))
    out_path = args.out if args.out is not None else cfg.get("out", "sample.csv")
    seed = int(cfg.get("seed", 0))
    K = int(cfg.get("grid", 4096))
    comps = cfg.get("components", [])
    bg = float(cfg.get("bg", 0.2))

    # Build grid and intensity
    if K <= 8:
        raise ValueError("Grid must be > 8")
    grid = np.linspace(0.0, TWOPI, K, endpoint=False)

    intensity = build_intensity(grid, components=comps, bg=max(0.0, bg))

    rng = np.random.default_rng(seed)
    x = sample_from_intensity(intensity, grid, n=n, rng=rng)

    df = pd.DataFrame({"x": x})
    df.to_csv(out_path, index=False)
    print(f"Saved {x.size} samples to {out_path}")


if __name__ == "__main__":
    sys.exit(main())
