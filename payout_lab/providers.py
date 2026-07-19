"""Market-data providers for the payout lab.

Two implementations of the same interface:

- ``MassiveProvider`` — live data from Massive (formerly Polygon.io).
  Needs ``MASSIVE_API_KEY`` (or legacy ``POLYGON_API_KEY``) in the
  environment. Endpoints used:
    * option chain snapshot:  GET /v3/snapshot/options/{underlying}
    * daily history:          GET /v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}
- ``SyntheticProvider`` — offline stand-in generating a realistic chain
  (SVI vol smile) and a fat-tailed, high-drift daily return history that
  mimics a large-cap semiconductor name. Lets the whole pipeline run and
  be tested without a key; numbers are illustrative only.

Both return the same two objects:

- ``ChainSnapshot``: spot, expiry, year-fraction T, rate r, and a calls
  DataFrame with columns ``strike, bid, ask, mid, iv``.
- ``history()``: a pd.Series of daily closes indexed by date.
"""
from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ChainSnapshot:
    ticker: str
    spot: float
    expiry: dt.date
    T: float                 # year fraction to expiry
    r: float                 # risk-free rate used for discounting
    calls: pd.DataFrame      # columns: strike, bid, ask, mid, iv
    puts: pd.DataFrame | None = None   # same columns; enables parity stitching

    @property
    def Z(self) -> float:
        """Zero-coupon bond price Z(t, T) = e^{-rT}."""
        return float(np.exp(-self.r * self.T))

    @property
    def forward(self) -> float:
        return float(self.spot * np.exp(self.r * self.T))


def get_provider(prefer_live: bool = True, **kwargs):
    """Return MassiveProvider when an API key is configured, else Synthetic."""
    if prefer_live and (os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY")):
        return MassiveProvider(**kwargs)
    return SyntheticProvider(**kwargs)


# --------------------------------------------------------------------------- #
#  Massive (formerly Polygon.io)                                              #
# --------------------------------------------------------------------------- #

class MassiveProvider:
    """Thin REST client for the Massive options + aggregates endpoints."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 r: float = 0.04):
        self.base_url = base_url or os.getenv("MASSIVE_BASE_URL", "https://api.massive.com")
        self.api_key = api_key or os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set MASSIVE_API_KEY (or POLYGON_API_KEY) to use live data.")
        self.r = r

    def _get(self, path_or_url: str, **params) -> dict:
        import requests
        url = path_or_url if path_or_url.startswith("http") else self.base_url + path_or_url
        params["apiKey"] = self.api_key
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def expirations(self, ticker: str) -> list[dt.date]:
        out, url, seen = [], "/v3/reference/options/contracts", set()
        data = self._get(url, underlying_ticker=ticker, contract_type="call",
                         limit=1000, sort="expiration_date")
        for row in data.get("results", []):
            d = dt.date.fromisoformat(row["expiration_date"])
            if d not in seen:
                seen.add(d)
                out.append(d)
        return out

    def _side(self, ticker: str, expiry: dt.date, contract_type: str):
        rows, url, params = [], f"/v3/snapshot/options/{ticker}", dict(
            expiration_date=expiry.isoformat(), contract_type=contract_type, limit=250)
        spot = None
        while url:
            data = self._get(url, **params)
            params = {}                                   # next_url embeds the query
            for res in data.get("results", []):
                det, quote = res.get("details", {}), res.get("last_quote", {}) or {}
                spot = spot or (res.get("underlying_asset", {}) or {}).get("price")
                rows.append({
                    "strike": det.get("strike_price"),
                    "bid": quote.get("bid"),
                    "ask": quote.get("ask"),
                    "iv": res.get("implied_volatility"),
                })
            url = data.get("next_url")
        df = pd.DataFrame(rows).dropna(subset=["strike"]).sort_values("strike")
        df = df[(df["bid"] > 0) & (df["ask"] > 0)].reset_index(drop=True)
        df["mid"] = (df["bid"] + df["ask"]) / 2
        return df, spot

    def chain(self, ticker: str, expiry: dt.date | None = None,
              dte_target: int = 45) -> ChainSnapshot:
        if expiry is None:
            today = dt.date.today()
            exps = [e for e in self.expirations(ticker) if e > today]
            expiry = min(exps, key=lambda e: abs((e - today).days - dte_target))

        calls, spot = self._side(ticker, expiry, "call")
        puts, spot2 = self._side(ticker, expiry, "put")
        spot = spot or spot2
        if spot is None:
            spot = float(self._get(f"/v2/aggs/ticker/{ticker}/prev")["results"][0]["c"])
        T = max((expiry - dt.date.today()).days, 1) / 365.0
        return ChainSnapshot(ticker, float(spot), expiry, T, self.r, calls, puts)

    def history(self, ticker: str, years: int = 8) -> pd.Series:
        end = dt.date.today()
        start = end - dt.timedelta(days=int(years * 365.25))
        data = self._get(f"/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}",
                         adjusted="true", sort="asc", limit=50000)
        rows = data.get("results", [])
        idx = pd.to_datetime([r["t"] for r in rows], unit="ms")
        return pd.Series([r["c"] for r in rows], index=idx, name=ticker)


# --------------------------------------------------------------------------- #
#  Synthetic offline stand-in                                                 #
# --------------------------------------------------------------------------- #

class SyntheticProvider:
    """Generates a plausible semiconductor-like chain and price history.

    The chain prices come from an SVI-parameterized vol smile (put skew plus
    a mildly lifted call wing, typical of semi names); quotes get a
    vol-dependent bid/ask. History is drawn from a fat-tailed mixture with
    high drift, echoing the sector's realized behavior. Purely illustrative.
    """

    def __init__(self, spot: float = 245.0, atm_vol: float = 0.33, r: float = 0.04,
                 drift: float = 0.22, hist_vol: float = 0.38, seed: int = 7):
        self.spot, self.atm_vol, self.r = spot, atm_vol, r
        self.drift, self.hist_vol, self.seed = drift, hist_vol, seed

    def _svi_vol(self, k: np.ndarray, T: float) -> np.ndarray:
        b, rho, m, sig = 0.10, -0.35, 0.02, 0.25
        a = self.atm_vol**2 * T - b * (rho * (0 - m) + np.sqrt((0 - m) ** 2 + sig**2))
        w = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sig**2))
        return np.sqrt(np.maximum(w, 1e-8) / T)

    def chain(self, ticker: str = "SOXX", expiry: dt.date | None = None,
              dte_target: int = 45) -> ChainSnapshot:
        from .implied import bs_call_price
        rng = np.random.default_rng(self.seed)
        expiry = expiry or dt.date.today() + dt.timedelta(days=dte_target)
        T = max((expiry - dt.date.today()).days, 1) / 365.0
        F = self.spot * np.exp(self.r * T)

        step = 5.0 if self.spot > 100 else 2.5
        lo = step * np.floor(self.spot * 0.55 / step)
        hi = step * np.ceil(self.spot * 1.65 / step)
        strikes = np.arange(lo, hi + step, step)

        iv = self._svi_vol(np.log(strikes / F), T)
        call_mid = bs_call_price(self.spot, strikes, T, self.r, iv)
        put_mid = call_mid - self.spot + strikes * np.exp(-self.r * T)  # parity

        def quote(mid):
            # spread widens away from the money, floored at a nickel; OTM side
            # of each instrument (small mids) keeps the tighter quotes
            spread = np.maximum(0.05, mid * (0.006 + 0.10 * np.abs(np.log(strikes / F))))
            noise = rng.normal(0, spread / 10)            # quote jitter
            df = pd.DataFrame({
                "strike": strikes,
                "bid": np.maximum(mid - spread / 2 + noise, 0.01),
                "ask": mid + spread / 2 + noise,
                "iv": iv,
            })
            df["mid"] = (df["bid"] + df["ask"]) / 2
            return df

        return ChainSnapshot(ticker, self.spot, expiry, T, self.r,
                             quote(call_mid), quote(np.maximum(put_mid, 0.0)))

    def history(self, ticker: str = "SOXX", years: int = 8) -> pd.Series:
        rng = np.random.default_rng(self.seed + 1)
        n = int(years * 252)
        daily_mu = self.drift / 252
        base_sig = self.hist_vol / np.sqrt(252)
        # 90% calm regime / 10% stress regime -> fat tails and vol clustering
        regime = rng.random(n) < 0.10
        sig = np.where(regime, base_sig * 2.2, base_sig * 0.83)
        rets = rng.normal(daily_mu, sig) + rng.standard_t(4, n) * base_sig * 0.15
        idx = pd.bdate_range(end=dt.date.today(), periods=n)
        start = self.spot / float(np.exp(rets.sum()))
        return pd.Series(start * np.exp(np.cumsum(rets)), index=idx, name=ticker)
