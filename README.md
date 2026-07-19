Notebooks of financial engineering notes and demos.

## Payout Lab (options → probability → payouts)

- `Option Price and Probability Duality.ipynb` — the duality: butterflies reveal the market's density; calls span all European payoffs.
- `Inverse Problem - Log-Optimal Semiconductor Payouts.ipynb` — the inverse: given your density vs the market's, the log-optimal payoff is `g ∝ p/q`; builds a priced order ticket.
- `payout_lab/` — reusable engine (Massive/ex-Polygon.io provider + synthetic offline provider, implied-density extraction, historical views, Kelly replication).
- `payout_app.py` — Streamlit UI (`MASSIVE_API_KEY=... streamlit run payout_app.py`).
- `APP_DESIGN.md` — architecture, thesis-validation backtest design, roadmap.
- `options-payout-review.html` — review of the options notebooks with verified math.
