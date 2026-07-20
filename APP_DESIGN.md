# Payout Lab — design

An interactive **financial-engineering laboratory**: the app teaches the math by letting
you touch it. Every module pairs a derivation (stated in the notation quants actually
use) with a live visualization whose controls map one-to-one onto the symbols in the
formula. Trading-tool features (tickets, budgets) appear only where they *illustrate* the
theory — the deliverable is understanding, benchmarked against quant industry standards.

The engine is `payout_lab/`; the UI is `payout_app.py` (Streamlit,
`streamlit run payout_app.py`); the long-form treatments live in the two notebooks
(*Option Price and Probability Duality*, *Inverse Problem — Log-Optimal Semiconductor
Payouts*). Modules deep-link via `?m=0..4`.

## Design principles

1. **Math first, sliders second.** Each page opens with the identity it teaches, in
   LaTeX, then gives you its parameters as controls. Moving a slider is moving a symbol.
2. **Industry conventions throughout.** Log-forward-moneyness $k=\ln(K/F)$, total
   implied variance $w=\sigma^2T$, SVI parameters, $\mathbb{P}$ vs $\mathbb{Q}$ measure
   language, discount factor $Z(t,T)$ — the vocabulary of a vol desk, not of a textbook
   simplification.
3. **Break things on purpose.** The fastest way to understand a no-arbitrage condition
   is to violate it and watch the diagnostic fire (negative density = butterfly
   arbitrage; Lee wing bound; mass ≠ 1).
4. **Real-world anchors.** Each abstract result lands on something in production use:
   digitals priced as spread limits, surface arb checks, the Cboe VIX formula, variance
   swaps, Kelly sizing.
5. **Runs with zero setup.** A synthetic provider (SVI chain + fat-tailed history) makes
   every module fully interactive offline; `MASSIVE_API_KEY` swaps in live Massive
   (ex-Polygon.io) data with no code change.

## The five modules (implemented)

| # | Module | The identity it teaches | Interactive element | Industry anchor |
|---|--------|------------------------|---------------------|-----------------|
| 1 | **Payoff Algebra** | payoffs form a vector space over $\{(S{-}K)^+,(K{-}S)^+,S,1\}$; payout ≠ P&L | preset structures (spreads, straddle, condor, risk reversal, tight-spread digital), vol slider, P&L toggle | how dealers quote digitals: $-\partial C/\partial K = Z\,\mathbb{Q}(S_T>K)$ |
| 2 | **Smile ↔ Density** | Breeden–Litzenberger: $q = \partial^2C/\partial K^2/Z$ | the five SVI sliders drive smile and density simultaneously; negative-density regions light up red | Gatheral's SVI; butterfly-arbitrage surface checks; Lee's moment bound on wings |
| 3 | **Spanning & the VIX** | Carr–Madan: $g(S_T)=g(\kappa)+g'(\kappa)(S_T{-}\kappa)+\int g''(K)\,\mathrm{opt}(K)\,dK$ | choose target payoff, coarsen the strike grid, watch replication error; strip weights plotted against $\Delta K/K^2$ | the Cboe VIX **is** this formula applied to the log contract; variance swaps |
| 4 | **The Inverse Problem** | $g^*=\frac{W}{Z}\frac{p}{q}$, growth $=\mathrm{KL}(p\|q)$ | Kelly-fraction and horizon dials; densities→ratio→payoff→tradable stepper; 20k-path outcome simulator | Kelly/fractional-Kelly sizing; $\mathbb{P}$-vs-$\mathbb{Q}$ premia; density-ratio trades |
| 5 | **Greeks** | $\Delta,\Gamma,\Theta$ as derivatives of $C(S)$; $\Theta + rS\Delta + \tfrac12\sigma^2S^2\Gamma = rC$ | tangent-line stepper (slope → curvature → rent); $\Gamma(S,t)$ pin-risk heatmap; per-day $\Theta$ vs convexity carry narration | homogeneity ties $\Gamma$ to the module-2 density ($S^2C_{SS}=K^2C_{KK}$); dealer gamma maps & pinning; daily hedged breakeven $\sigma S\sqrt{1/252}$ |

## Module roadmap (each = one page, same recipe: identity → controls → anchor)

| Priority | Module | Teaches | Notes |
|---|--------|---------|-------|
| next | **The vol surface in 3D** | term structure + skew as one object; calendar no-arb ($w$ increasing in $T$) | extend SVI to a slice family; calendar-arb diagnostic like module 2's butterfly check |
| soon | **Measure change, visually** | Girsanov as a tilt: $\frac{d\mathbb{Q}}{d\mathbb{P}} \propto e^{-\lambda W_T}$; risk premium = drift wedge between the module-4 densities | animate the reweighting of paths |
| soon | **Monte Carlo vs closed form** | GBM paths → payoff averaging converging to BS; variance reduction (antithetic, control variate on the stock) | the standard first quant-dev exercise, done honestly |
| later | **American exercise** | binomial lattice; early-exercise boundary; why the module-4 ETF caveat exists | ties to SOXX/SMH options being American |
| later | **Realized vs implied** | variance-swap P&L decomposition $\sum \Gamma S^2(\sigma_{real}^2-\sigma_{imp}^2)dt$ | the cleanest "density disagreement" trade; needs historical data (Massive) |
| later | **Backtesting the thesis** | out-of-sample log scores $\log p_t(S_T) - \log q_t(S_T)$ per ticker | the empirical companion to module 4; needs Massive historical chains |

## Data layer

Unchanged from the engine: `MassiveProvider` (chain snapshots both sides of the book,
daily aggregates; `MASSIVE_API_KEY`) with `SyntheticProvider` fallback. Only module 4 and
the two "later" empirical modules touch data at all — the theory modules are self-
contained mathematics, deliberately, so the app is useful with no key and no network.

## Honest limitations

- European-exercise math on American-style listed options (flagged in module 4; an
  American-exercise module is on the roadmap precisely to teach the gap).
- Synthetic data is calibrated to be plausible, not to match any date's market.
- The educational ticket uses mid fills and ignores fees/margin — by design, with the
  caveat printed next to it. Nothing here is investment advice.
