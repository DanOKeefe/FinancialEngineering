"""Streamlit front-end for the inverse problem: probability view -> order ticket.

Run:  MASSIVE_API_KEY=... streamlit run payout_app.py
Without a key it runs on the synthetic provider so the UI is explorable offline.
See APP_DESIGN.md for the architecture and roadmap.
"""
import numpy as np
import pandas as pd
import streamlit as st

import payout_lab as pl

st.set_page_config(page_title="Payout Lab", page_icon="📈", layout="wide")
st.title("Payout Lab — bet your density, not a direction")

with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Ticker", "NVDA")
    dte = st.slider("Target days to expiry", 14, 180, 45)
    wealth = st.number_input("Budget ($)", 1_000, 1_000_000, 10_000, step=1_000)
    fraction = st.slider("Kelly fraction (conviction)", 0.0, 1.0, 0.5, 0.05)
    drift_haircut = st.slider("Drift haircut on history (annualized)", 0.0, 0.30, 0.0, 0.01,
                              help="Subtract this from the historical drift before "
                                   "building your view — the past may overstate the edge.")
    years = st.slider("Years of history for the view", 2, 15, 8)


@st.cache_data(ttl=300, show_spinner="Fetching chain…")
def load(ticker: str, dte: int, years: int):
    provider = pl.get_provider()
    chain = provider.chain(ticker, dte_target=dte)
    hist = provider.history(ticker, years=years)
    return chain, hist, type(provider).__name__


chain, hist, provider_name = load(ticker, dte, years)
if provider_name == "SyntheticProvider":
    st.warning("No MASSIVE_API_KEY found — running on synthetic demo data.")

q = pl.implied_distribution(chain)
horizon = max(int(chain.T * 252), 1)
mu_hist = float(np.log(hist).diff().mean() * 252)
p_full = pl.historical_view(hist, horizon, chain.spot, q.grid,
                            drift_override=mu_hist - drift_haircut if drift_haircut else None)
p_view = pl.geometric_blend(p_full, q.pdf, q.grid, f=fraction)
g = pl.log_optimal_payout(q.grid, p_view, q.pdf, wealth=wealth, Z=chain.Z, ratio_cap=8)
ticket = pl.replicate(q.grid, g, chain)
summary = pl.summarize(q.grid, p_view, q.pdf, g, ticket, chain, wealth=wealth)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Spot", f"${chain.spot:,.2f}", help=f"expiry {chain.expiry}")
c2.metric("Edge if right (annualized)", f"{summary['annualized (replicated)']:.1%}")
c3.metric("P(lose) under your view", f"{summary['P(lose money) under your view']:.0%}")
c4.metric("Replication cost", f"${ticket.cost:,.0f}")

left, right = st.columns(2)
with left:
    st.subheader("Densities")
    df = pd.DataFrame({"market-implied q": q.pdf, "your view p_f": p_view}, index=q.grid)
    st.line_chart(df, x_label="S_T", y_label="density")
with right:
    st.subheader("Payoff at expiry")
    df = pd.DataFrame({"ideal g*": g, "replication": ticket.payoff(q.grid),
                       "break even": np.full_like(q.grid, wealth)}, index=q.grid)
    st.line_chart(df, x_label="S_T", y_label="terminal wealth ($)")

st.subheader("Order ticket")
st.caption(f"Plus {ticket.stock:,.1f} shares and ${ticket.bonds:,.0f} bond face value "
           f"(negative = borrow); OTM puts replace deep-ITM calls via parity.")
legs = ticket.legs.assign(side=np.where(ticket.legs["qty"] >= 0, "BUY", "SELL"))
st.dataframe(legs[["type", "strike", "side", "qty", "mid", "cost"]].round(2),
             use_container_width=True, hide_index=True)

st.caption("Research prototype — mid-price fills, no fees/margin modeled. Not investment advice.")
