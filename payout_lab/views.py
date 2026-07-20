"""Subjective ("real-world") terminal-price densities.

The thesis being expressed for semiconductor names: realized return
distributions have historically sat to the right of what option prices
implied. ``historical_view`` encodes that directly — a block bootstrap of
actual h-day compounded returns, kernel-smoothed into a density over S_T.
``geometric_blend`` shrinks the view toward the market density, which is
exactly fractional Kelly on the resulting payoff.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def historical_view(closes: pd.Series, horizon_days: int, S0: float,
                    grid: np.ndarray, n_boot: int = 20_000, block: int = 5,
                    drift_override: float | None = None,
                    seed: int = 0) -> np.ndarray:
    """Density of S_T from a block bootstrap of historical h-day log returns.

    drift_override (annualized) recenters the bootstrap mean — use it to
    haircut the historical drift if you think the past overstates the edge.
    """
    rng = np.random.default_rng(seed)
    r = np.log(closes).diff().dropna().to_numpy()
    n_blocks = int(np.ceil(horizon_days / block))
    starts = rng.integers(0, len(r) - block, size=(n_boot, n_blocks))
    idx = starts[:, :, None] + np.arange(block)[None, None, :]
    R = r[idx].reshape(n_boot, -1)[:, :horizon_days].sum(axis=1)

    if drift_override is not None:
        R = R - R.mean() + drift_override * horizon_days / 252

    kde = gaussian_kde(S0 * np.exp(R))
    pdf = kde(grid)
    return pdf / np.trapezoid(pdf, grid)


def geometric_blend(p: np.ndarray, q: np.ndarray, grid: np.ndarray,
                    f: float = 1.0) -> np.ndarray:
    """p_f ∝ p^f · q^(1-f): confidence-weighted view.

    f=1 is full conviction (pure p), f=0 collapses to the market (q) and the
    optimal payoff degenerates to holding cash/bonds. Using p_f in the Kelly
    payoff is equivalent to fractional-Kelly sizing, since
    (p_f / q) = (p/q)^f.
    """
    eps = 1e-300
    blended = np.exp(f * np.log(p + eps) + (1 - f) * np.log(q + eps))
    return blended / np.trapezoid(blended, grid)
