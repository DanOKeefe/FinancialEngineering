"""The inverse problem: from a probability disagreement to an order ticket.

Maximize expected log terminal wealth E_p[log g(S_T)] subject to the budget
that the payoff's market price equals initial wealth W:

    Z * E_q[g(S_T)] = W          (FTAP: price = discounted q-expectation)

Lagrange gives the growth-optimal (Kelly) payoff

    g*(S) = (W / Z) * p(S) / q(S)

and the resulting expected log growth is

    E_p[log(g*/W)] = KL(p || q) - log Z

— your edge, in nats, is literally the KL divergence between your density
and the market's. The payoff is then made tradable by clipping it to the
listed-strike range and replicating the piecewise-linear interpolant
exactly with bonds + calls (slope-change decomposition, the discrete form
of the spanning integral  g(S) = g(0) + g'(0)S + ∫ (S-K)^+ g''(K) dK ).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .providers import ChainSnapshot


def log_optimal_payout(grid: np.ndarray, p: np.ndarray, q: np.ndarray,
                       wealth: float = 1.0, Z: float = 1.0,
                       ratio_cap: float = 20.0) -> np.ndarray:
    """g*(S) = (W/Z) p/q, with the ratio capped in the far tails.

    Where q -> 0 faster than p the raw ratio explodes; those payouts are
    unbuyable (no strikes trade there) and dominate the budget, so cap the
    ratio and renormalize to respect the budget exactly on the grid.
    """
    ratio = np.minimum(p / np.maximum(q, 1e-12), ratio_cap)
    g = ratio * wealth / Z
    cost = Z * np.trapezoid(q * g, grid)
    return g * wealth / cost


def expected_log_growth(grid: np.ndarray, p: np.ndarray, g: np.ndarray,
                        wealth: float = 1.0) -> float:
    """E_p[log(g/W)] in nats over the horizon (not annualized)."""
    return float(np.trapezoid(p * np.log(np.maximum(g, 1e-12) / wealth), grid))


@dataclass
class Ticket:
    """A tradable replication: bonds + stock + option legs at listed strikes.

    With ``use_puts`` the strikes below spot are held as OTM puts instead of
    deep-ITM calls, via parity (S-K)^+ = (K-S)^+ + S - K: same payoff, but
    quoted off the liquid side of the book. The parity conversion moves
    K worth of face value into (possibly negative = borrowed) bonds and one
    share per contract into the stock line.
    """
    bonds: float                       # face value of zero-coupon bonds held
    stock: float                       # shares of the underlying held
    legs: pd.DataFrame                 # type (C/P), strike, qty, mid, cost
    cost: float                        # total cost: bonds at Z + stock + legs
    payoff_strikes: np.ndarray = field(repr=False)
    payoff_values: np.ndarray = field(repr=False)

    def payoff(self, S: np.ndarray) -> np.ndarray:
        """Exact payoff of the replicating portfolio at expiry."""
        S = np.asarray(S, float)
        out = self.bonds + self.stock * S
        for _, leg in self.legs.iterrows():
            if leg["type"] == "C":
                out = out + leg["qty"] * np.maximum(S - leg["strike"], 0)
            else:
                out = out + leg["qty"] * np.maximum(leg["strike"] - S, 0)
        return out


def replicate(grid: np.ndarray, g: np.ndarray, chain: ChainSnapshot,
              max_legs: int | None = None, min_mid: float = 0.10,
              smooth_sigma: float | None = None, use_puts: bool = True) -> Ticket:
    """Replicate g on the chain's listed strikes with bonds + calls.

    The piecewise-linear interpolant of g at strikes K_0..K_n, held flat
    outside [K_0, K_n] (bounded liability in the untraded tails), is
    replicated *exactly* by:

        g(K_0) bonds  +  sum_i (slope_{i+1} - slope_i) calls at K_i

    with slope_0 = 0 below K_0 and slope_{n+1} = 0 above K_n. This is the
    finite-strike version of holding g''(K)dK calls at every strike.

    Practicalities: strikes whose mid quote is below ``min_mid`` are
    dropped (junk quotes make junk legs), and g is pre-smoothed with a
    Gaussian kernel (default: one strike spacing) so estimation wiggle in
    p/q doesn't masquerade as huge offsetting call positions.
    """
    liquid = chain.calls[chain.calls["mid"] >= min_mid]
    strikes = liquid["strike"].to_numpy()
    strikes = strikes[(strikes >= grid[0]) & (strikes <= grid[-1])]
    if max_legs and len(strikes) > max_legs:
        keep = np.linspace(0, len(strikes) - 1, max_legs).round().astype(int)
        strikes = strikes[np.unique(keep)]

    from scipy.ndimage import gaussian_filter1d
    dgrid = grid[1] - grid[0]
    if smooth_sigma is None:
        smooth_sigma = float(np.median(np.diff(strikes))) / dgrid
    g_sm = gaussian_filter1d(g, smooth_sigma, mode="nearest")

    gk = np.interp(strikes, grid, g_sm)
    slopes = np.concatenate([[0.0], np.diff(gk) / np.diff(strikes), [0.0]])
    qty = np.diff(slopes)                        # call quantity at each strike

    mids = chain.calls.set_index("strike")["mid"].reindex(strikes).to_numpy()
    legs = pd.DataFrame({"type": "C", "strike": strikes, "qty": qty, "mid": mids})
    legs = legs[np.abs(legs["qty"]) > 1e-6].reset_index(drop=True)

    bonds, stock = float(gk[0]), 0.0
    if use_puts and chain.puts is not None and not chain.puts.empty:
        put_mid = chain.puts.set_index("strike")["mid"]
        swap = (legs["strike"] < chain.forward) & legs["strike"].isin(put_mid.index)
        # (S-K)^+  =  (K-S)^+ + S - K   at expiry
        stock = float(legs.loc[swap, "qty"].sum())
        bonds -= float((legs.loc[swap, "qty"] * legs.loc[swap, "strike"]).sum())
        legs.loc[swap, "type"] = "P"
        legs.loc[swap, "mid"] = put_mid.reindex(legs.loc[swap, "strike"]).to_numpy()

    legs["cost"] = legs["qty"] * legs["mid"]
    cost = bonds * chain.Z + stock * chain.spot + float(legs["cost"].sum())
    return Ticket(bonds=bonds, stock=stock, legs=legs, cost=cost,
                  payoff_strikes=strikes, payoff_values=gk)


def summarize(grid, p, q, g, ticket: Ticket, chain: ChainSnapshot,
              wealth: float = 1.0) -> pd.Series:
    """Headline stats for a proposed trade."""
    h = ticket.payoff(grid)
    growth_ideal = expected_log_growth(grid, p, g, wealth)
    growth_repl = expected_log_growth(grid, p, np.maximum(h, 1e-9), ticket.cost)
    p_loss = float(np.trapezoid(p * (h < ticket.cost), grid))
    q_loss = float(np.trapezoid(q * (h < ticket.cost), grid))
    return pd.Series({
        "budget W": wealth,
        "replication cost": ticket.cost,
        "expected log growth (ideal)": growth_ideal,
        "expected log growth (replicated)": growth_repl,
        "annualized (replicated)": growth_repl / chain.T,
        "P(lose money) under your view": p_loss,
        "P(lose money) market-implied": q_loss,
        "option legs": len(ticket.legs),
    })
