"""Market-implied (risk-neutral) distribution from an option chain.

Pipeline (the robust version of the Breeden-Litzenberger differencing in
``Option Price and Probability Duality.ipynb``):

1. mid quotes -> implied vols (provider IVs when present, else inverted)
2. smooth the vol curve in log-moneyness (spread-weighted spline), because
   differencing raw prices twice amplifies quote noise into negative
   densities
3. rebuild call prices on a dense strike grid from the smoothed vols
4. q(K) = C''(K) / Z  (Breeden-Litzenberger), clip tiny negatives,
   renormalize, and report diagnostics
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq
from scipy.stats import norm

from .providers import ChainSnapshot


def bs_call_price(S, K, T, r, vol):
    K, vol = np.asarray(K, float), np.asarray(vol, float)
    d1 = (np.log(S / K) + (r + vol**2 / 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_vol(price, S, K, T, r) -> float:
    intrinsic = max(S - K * np.exp(-r * T), 0.0)
    if price <= intrinsic + 1e-10 or price >= S:
        return np.nan
    try:
        return brentq(lambda v: bs_call_price(S, K, T, r, v) - price, 1e-4, 5.0)
    except ValueError:
        return np.nan


@dataclass
class ImpliedDistribution:
    grid: np.ndarray          # terminal prices S_T
    pdf: np.ndarray           # risk-neutral density q(S_T)
    cdf: np.ndarray
    clipped_mass: float       # negative density removed (should be ~0)
    raw_mass: float           # integral before renormalizing (should be ~1)

    def prob_between(self, a: float, b: float) -> float:
        return float(np.interp(b, self.grid, self.cdf) - np.interp(a, self.grid, self.cdf))


def stitched_quotes(chain: ChainSnapshot) -> pd.DataFrame:
    """OTM-stitched call-equivalent quotes: puts below the forward, calls above.

    Deep-ITM calls trade wide and stale; the liquidity below the forward
    lives in OTM puts. Put-call parity (C = P + S - K·Z, no dividends)
    converts put mids into synthetic call mids, so the whole strike axis is
    quoted off the tighter side of the book. Falls back to calls-only when
    the snapshot has no put side.
    """
    calls = chain.calls.assign(source="call", mid_call=chain.calls["mid"])
    if chain.puts is None or chain.puts.empty:
        return calls
    F = chain.forward
    puts = chain.puts.assign(source="put")
    puts["mid_call"] = puts["mid"] + chain.spot - puts["strike"] * chain.Z
    out = pd.concat([puts[puts["strike"] <= F], calls[calls["strike"] > F]])
    return out[out["mid_call"] > 0].sort_values("strike").reset_index(drop=True)


def implied_distribution(chain: ChainSnapshot, n_grid: int = 601,
                         smooth: float | None = None) -> ImpliedDistribution:
    quotes = stitched_quotes(chain)
    S0, T, r, Z = chain.spot, chain.T, chain.r, chain.Z
    F = S0 * np.exp(r * T)

    if "iv" not in quotes or quotes["iv"].isna().all():
        quotes["iv"] = [implied_vol(p, S0, k, T, r)
                        for p, k in zip(quotes["mid_call"], quotes["strike"])]
    quotes = quotes.dropna(subset=["iv"])
    quotes = quotes[(quotes["iv"] > 0.01) & (quotes["iv"] < 4.0)]

    k = np.log(quotes["strike"].to_numpy() / F)
    iv = quotes["iv"].to_numpy()
    w = 1.0 / np.maximum((quotes["ask"] - quotes["bid"]).to_numpy(), 0.01)

    if smooth is None:
        smooth = len(k) * np.var(iv) * 0.05          # light default smoothing
    spline = UnivariateSpline(k, iv, w=w / w.mean(), s=smooth, k=3)

    Ks = np.linspace(quotes["strike"].min(), quotes["strike"].max(), n_grid)
    C = bs_call_price(S0, Ks, T, r, spline(np.log(Ks / F)))

    dK = Ks[1] - Ks[0]
    pdf = (C[:-2] - 2 * C[1:-1] + C[2:]) / (Z * dK**2)
    grid = Ks[1:-1]

    raw_mass = float(np.trapezoid(np.abs(pdf), grid))
    clipped = float(np.trapezoid(np.abs(np.minimum(pdf, 0)), grid))
    pdf = np.maximum(pdf, 0)
    pdf = pdf / np.trapezoid(pdf, grid)
    cdf = np.concatenate([[0], np.cumsum((pdf[1:] + pdf[:-1]) / 2 * np.diff(grid))])
    return ImpliedDistribution(grid, pdf, np.clip(cdf, 0, 1), clipped, raw_mass)
