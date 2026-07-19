# Payout Lab — app design

A small application that turns a probability view on an underlying (e.g. "the market
persistently underprices semiconductor upside") into a priced, tradable option order
ticket, fed by live data from **Massive** (formerly Polygon.io).

The math and a working end-to-end run live in
`Inverse Problem - Log-Optimal Semiconductor Payouts.ipynb`; the reusable engine is the
`payout_lab/` package; `payout_app.py` is the UI. This document is the plan around them.

## The core loop

```
Massive REST ──► ChainSnapshot ──► implied density q(S_T)     (Breeden–Litzenberger,
      │                                                        vol-space smoothing)
      └────────► daily history ──► subjective density p(S_T)  (block bootstrap,
                                                               drift haircut, Kelly fraction f)
                       g*(S) = (W/Z) · (p/q)^f  ──►  replicate on listed strikes
                                                 ──►  order ticket + edge metrics
```

Key identities the app is built on (derived in the two notebooks):

- price = discounted risk-neutral expectation (FTAP), so butterflies reveal `q`
- log-optimal payoff under budget: `g* = (W/Z)·p/q`
- expected log growth of `g*` = `KL(p‖q) − log Z` — **the edge is the KL divergence**
- any European payoff = bonds + stock + strip of calls (spanning), so `g*` is executable

## Data layer (Massive / ex-Polygon.io)

Implemented in `payout_lab/providers.py::MassiveProvider`.

| Need | Endpoint | Notes |
|---|---|---|
| option chain snapshot | `GET /v3/snapshot/options/{underlying}` | one pass per `contract_type` (call, put) with `expiration_date=`; paginate `next_url`; gives bid/ask/IV/greeks per contract |
| expirations | `GET /v3/reference/options/contracts` | pick nearest to target DTE |
| spot + daily history | `GET /v2/aggs/ticker/{t}/range/1/day/{from}/{to}` | adjusted closes for the bootstrap view |
| (later) historical chains | flat files / options aggregates | needed for the thesis backtest, see below |

Config: `MASSIVE_API_KEY` (legacy `POLYGON_API_KEY` also honored), optional
`MASSIVE_BASE_URL`. Auth via `apiKey` query param.

Practical handling already in the code, worth keeping in any rewrite:

- drop quotes with `bid == 0` or mid < $0.10 (junk quotes make junk legs)
- smooth IVs (spread-weighted spline in log-moneyness) **before** differencing — second
  differences of raw mids produce negative densities
- clip/report negative density mass; renormalize; surface `raw mass ≈ 1` as a health check
- cache chain snapshots ~5 min (`st.cache_data(ttl=300)`); history ~1 day

## UI (Streamlit, `payout_app.py`)

Inputs: ticker, target DTE, budget, **Kelly fraction** (conviction dial), **drift haircut**
(how much of the historical drift you refuse to extrapolate), years of history.
Outputs: density comparison, ideal-vs-replicated payoff, order ticket table, and four
headline metrics (spot, annualized edge-if-right, P(lose) under the view, cost).

Deliberate design choices:

- **Conviction is continuous, not boolean.** `f` and the drift haircut both shrink the
  trade toward "hold bonds"; full Kelly is the aggressive end, not the default.
- **Show P(lose money) prominently.** Density-ratio payoffs lose often and win big;
  hiding that would misrepresent the strategy.
- **Report ideal vs replicated growth.** The gap (ratio cap, finite strikes, smoothing)
  is the implementation-efficiency metric to watch.

## Validating "the market undervalued semis" (before betting on it)

The thesis deserves measurement, not assumption. With Massive's historical options data:

1. For each month-end since ~2016 and each name in the basket (NVDA, AMD, TSM, AVGO,
   MU, ASML, SOXX/SMH), reconstruct `q_t` from that day's chain at ~45 DTE.
2. Build `p_t` from data available *at that time* (rolling bootstrap — no lookahead).
3. When the option expires, record realized `S_T`; score both densities:
   `edge_t = log p_t(S_T) − log q_t(S_T)`.
4. The running mean of `edge_t` is the realized, out-of-sample KL edge — exactly the
   quantity the Kelly payoff monetizes. Positive and stable ⇒ thesis holds; also shows
   *where* it comes from (right tail vs body) and whether it decayed after 2023.
5. Paper-trade the ticket generator over the same history for a P&L-based check
   (includes spread costs, which log scores ignore).

## Feature plan

Four screens, built in this order. Each phase is shippable on its own; effort assumes the
`payout_lab` engine keeps doing the math and the app stays a thin layer over it.

### Screen 1 — Trade Builder (phases 1–2)

The notebook as an interactive page: pick underlying/expiry, shape a view, get a ticket.

| # | Feature | Notes | Status / effort |
|---|---------|-------|--------|
| 1.1 | Live chains + history via Massive | `MassiveProvider`, both sides of book | ✅ done (needs key) |
| 1.2 | Put–call parity stitching | OTM puts quote the left side of `q`; ticket executes left wing in puts | ✅ done |
| 1.3 | Kelly fraction + drift haircut dials | conviction is continuous | ✅ done |
| 1.4 | Contract realism | round to 100-share contracts, scale ticket to exact budget, min-size warning, cap-leg surfaced explicitly | S |
| 1.5 | Spread-aware pricing toggle | mid vs. cross-the-spread (the duality notebook's bid/ask butterfly analysis, generalized); shows edge net of execution | M |
| 1.6 | View builder beyond bootstrap | mixtures: bootstrap ⊕ analyst-target lognormal ⊕ event scenario (bimodal for single names); saved named views | M |
| 1.7 | Scenario save/load | a view + ticket + rationale, serialized to JSON; the unit the backtester replays | S |

### Screen 2 — Sector Scanner (phase 2)

Where "the market undervalues semis" becomes a ranked list instead of an assumption.

| # | Feature | Notes | Effort |
|---|---------|-------|--------|
| 2.1 | Basket table: SOXX, SMH, XSX?, NVDA, AMD, TSM, AVGO, MU, ASML | per name: KL(p‖q), annualized edge, liquidity score (spread/OI), earnings-inside-expiry flag | M |
| 2.2 | Vehicle chooser | same thesis priced on SOXX vs SMH vs top single names; recommends by edge × liquidity (SMH usually wins on liquidity) | S after 2.1 |
| 2.3 | Expiry term structure | KL by expiry — is the disagreement in the front month or the back? | M |

### Screen 3 — Thesis Backtest (phase 3, the credibility screen)

The § above ("Validating…") as a page: reconstruct `q_t` monthly from historical chains
(this is the feature that requires the paid Massive options-history tier), score
`log p_t(S_T) − log q_t(S_T)` out-of-sample, plot the cumulative edge per ticker with
regime shading (pre/post 2023), and paper-trade the ticket generator with spread costs.
Effort: L — but it's the difference between a toy and an instrument.

### Screen 4 — Positions & Monitoring (phase 4)

| # | Feature | Notes | Effort |
|---|---------|-------|--------|
| 4.1 | Saved positions, mark-to-market | reprice ticket legs off live quotes | M |
| 4.2 | View drift alerts | recompute KL(p‖q) daily; alert when the market moves toward (edge gone — take profit?) or away from (re-examine or add) your view | M |
| 4.3 | Roll assistant | at expiry−N days, re-run the builder on the next expiry and diff the tickets | M |
| 4.4 | Export | ticket → CSV / broker multi-leg format | S |

### Deliberately out of scope (for now)

- Auto-execution / broker API — this is a research instrument; keep a human on the ticket.
- Multi-underlying joint Kelly (correlated payoffs) — real math project, revisit after
  the backtest proves single-name edges exist.
- Intraday/websocket streaming — daily-close granularity matches the horizon of the thesis.

## Honest limitations

- The bootstrap view assumes the future resembles the sampled past; the drift haircut is
  the guardrail, not a solution. The backtest (v2) is the real answer.
- American-style single-name options carry early-exercise premium the European math
  ignores (small for OTM/short-dated; SPX/XSP or European-style index options avoid it).
- Mid-price fills are optimistic; the butterfly-spread cost analysis in the duality
  notebook is the template for a real slippage model.
- Nothing here is investment advice; it is a research instrument for making a
  disagreement with the market precise, priced, and falsifiable.
