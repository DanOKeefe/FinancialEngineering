"""Payout Lab — an interactive financial-engineering laboratory.

Four modules, each pairing a derivation with a live visualization:

  1. Payoff Algebra        — compose options; payout vs P&L; limits to digitals
  2. Smile <-> Density     — SVI vol surface conventions and Breeden-Litzenberger
  3. Spanning & the VIX    — replicate any payoff with a call strip; the log contract
  4. The Inverse Problem   — measure disagreement, Kelly payoffs, replication

Run:  streamlit run payout_app.py
Live data (module 4) uses Massive (ex-Polygon.io) when MASSIVE_API_KEY is set;
otherwise a synthetic provider keeps every module fully interactive offline.
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import payout_lab as pl

# ----------------------------------------------------------------- chrome
st.set_page_config(page_title="Payout Lab", page_icon="📐", layout="wide")

INK, MUT, GRIDC = "#0b0b0b", "#898781", "#e1e0d9"
BLUE, GREEN, MAGENTA, YELLOW, RED = "#2a78d6", "#008300", "#e87ba4", "#eda100", "#e34948"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.grid": True, "grid.color": GRIDC, "axes.edgecolor": "#c3c2b7",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUT, "ytick.color": MUT, "axes.labelcolor": "#52514e",
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "legend.frameon": False, "lines.linewidth": 2.0,
})

MODULES = ["1 · Payoff Algebra", "2 · Smile ↔ Density", "3 · Spanning & the VIX",
           "4 · The Inverse Problem"]
try:
    _default = int(st.query_params.get("m", 0))
except (TypeError, ValueError):
    _default = 0

with st.sidebar:
    st.title("📐 Payout Lab")
    module = st.radio("Module", MODULES, index=min(_default, 3))
    st.caption("A financial-engineering laboratory: every page pairs the derivation "
               "with a live visualization. Notation follows industry convention — "
               "forward moneyness, total implied variance, risk-neutral measure $\\mathbb{Q}$.")


def fig_axes(n=1, height=3.4):
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, height))
    return fig, (axes if n > 1 else [axes])


# ======================================================================= 1
def module_payoff_algebra():
    st.header("Payoff algebra: options as a vector space")
    st.markdown(
        "European payoffs add: a portfolio's payout is the weighted sum of its legs' "
        "payouts. Every named structure below is just a point in the vector space "
        "spanned by $\\{(S-K)^+,\\ (K-S)^+,\\ S,\\ 1\\}$. The distinction that trips "
        "everyone up once: **payout** (value at expiry) vs **P&L** (payout minus what "
        "you paid, compounded). Toggle it below.")

    presets = {
        "bull call spread": [("C", 95, 1), ("C", 105, -1)],
        "straddle": [("C", 100, 1), ("P", 100, 1)],
        "strangle": [("C", 110, 1), ("P", 90, 1)],
        "butterfly": [("C", 90, 1), ("C", 100, -2), ("C", 110, 1)],
        "iron condor": [("P", 80, 1), ("P", 90, -1), ("C", 110, -1), ("C", 120, 1)],
        "risk reversal": [("P", 90, -1), ("C", 110, 1)],
        "digital ≈ tight call spread": [("C", 100, 10), ("C", 100.1, -10)],
    }
    c1, c2, c3 = st.columns([2, 1, 1])
    name = c1.selectbox("Structure", list(presets))
    vol = c2.slider("Implied vol for pricing", 0.10, 0.90, 0.30, 0.05)
    show_pnl = c3.toggle("Show P&L (net of premium)", value=True)

    S0, r, T = 100.0, 0.04, 0.25
    S = np.linspace(50, 150, 1201)
    Z = np.exp(-r * T)

    legs = pd.DataFrame(presets[name], columns=["type", "K", "qty"])
    payout = np.zeros_like(S)
    premium = 0.0
    for _, leg in legs.iterrows():
        c = pl.bs_call_price(S0, leg.K, T, r, vol)
        if leg.type == "C":
            payout += leg.qty * np.maximum(S - leg.K, 0)
            premium += leg.qty * c
        else:                                  # parity: P = C - S + K Z
            payout += leg.qty * np.maximum(leg.K - S, 0)
            premium += leg.qty * (c - S0 + leg.K * Z)

    fig, (ax,) = fig_axes()
    ax.plot(S, payout, color=INK, label="payout at $T$")
    if show_pnl:
        ax.plot(S, payout - premium / Z, color=BLUE, ls="--",
                label=f"P&L (premium {premium:+.2f} paid at $t$)")
        ax.axhline(0, color=MUT, lw=1)
    ax.axvline(S0, color=MUT, lw=1, ls=":")
    ax.set_xlabel("$S_T$"); ax.set_ylabel("value at expiry")
    ax.set_title(name); ax.legend(fontsize=9)
    st.pyplot(fig)

    st.dataframe(legs, hide_index=True, width=280)
    with st.expander("Why the digital is a limit, not a product"):
        st.latex(r"\lambda\left(C_K - C_{K+1/\lambda}\right)\;\xrightarrow{\lambda\to\infty}\;"
                 r"-\frac{\partial C_K}{\partial K} \;=\; Z(t,T)\,\mathbb{Q}(S_T>K)")
        st.markdown(
            "The tight call spread in the preset list has payout ramping 0→1 over 10 cents — "
            "already indistinguishable from the indicator $I\\{S_T>K\\}$. Its price is "
            "therefore the **discounted risk-neutral probability** of finishing above $K$. "
            "Dealers price digitals exactly this way, and quote them inside the "
            "over/under-replicating spread pair ($[K-\\varepsilon,K]$ vs $[K,K+\\varepsilon]$).")


# ======================================================================= 2
def module_smile_density():
    st.header("The smile and the density are the same object")
    st.markdown(
        "Industry quotes volatility, not price: the axis is **log-forward-moneyness** "
        "$k=\\ln(K/F)$ and the object is **total implied variance** $w(k)=\\sigma^2(k)\\,T$. "
        "The standard parameterization is Gatheral's **SVI**:")
    st.latex(r"w(k) \;=\; a + b\left(\rho\,(k-m) + \sqrt{(k-m)^2+\sigma^2}\right)")
    st.markdown(
        "Breeden–Litzenberger turns any smile into the risk-neutral density "
        "$q(K) = \\partial^2 C/\\partial K^2 \\,/\\, Z$. Push the sliders to extremes and "
        "watch the density go **negative** — that is *butterfly arbitrage*, and desks run "
        "exactly this check before publishing a surface. Lee's moment formula caps the wing "
        "slopes: $b(1+|\\rho|) \\le 2/T \\cdot T = 2$ (in total-variance terms).")

    c = st.columns(5)
    a_ = c[0].slider("a (base var)", 0.001, 0.10, 0.012, 0.001, format="%.3f")
    b_ = c[1].slider("b (wing slope)", 0.01, 1.50, 0.10, 0.01)
    rho = c[2].slider("ρ (skew)", -0.99, 0.99, -0.35, 0.01)
    m_ = c[3].slider("m (shift)", -0.5, 0.5, 0.02, 0.01)
    sg = c[4].slider("σ (smoothing)", 0.01, 1.0, 0.25, 0.01)

    S0, r, T = 100.0, 0.04, 0.25
    F, Z = S0 * np.exp(r * T), np.exp(-r * T)
    K = np.linspace(40, 220, 901)
    k = np.log(K / F)
    w = a_ + b_ * (rho * (k - m_) + np.sqrt((k - m_) ** 2 + sg**2))
    iv = np.sqrt(np.maximum(w, 1e-8) / T)
    C = pl.bs_call_price(S0, K, T, r, iv)
    dK = K[1] - K[0]
    q = (C[:-2] - 2 * C[1:-1] + C[2:]) / (Z * dK**2)
    Kq = K[1:-1]

    fig, axes = fig_axes(2)
    ax = axes[0]
    ax.plot(k, iv, color=BLUE)
    ax.set_xlabel("log-moneyness  $k=\\ln(K/F)$"); ax.set_ylabel("implied vol")
    ax.set_title("SVI smile (the quoted object)")
    ax = axes[1]
    ax.plot(Kq, q, color=INK)
    neg = q < 0
    if neg.any():
        ax.fill_between(Kq, q, 0, where=neg, color=RED, alpha=0.6, label="q < 0: butterfly arbitrage")
        ax.legend(fontsize=9)
    ax.axhline(0, color=MUT, lw=1)
    ax.set_xlabel("$K$"); ax.set_ylabel("density")
    ax.set_title("Implied density  $q=\\partial^2C/\\partial K^2/Z$  (the same object)")
    st.pyplot(fig)

    mass = float(np.trapezoid(np.maximum(q, 0), Kq))
    negmass = max(0.0, float(-np.trapezoid(np.minimum(q, 0), Kq)))
    lee = b_ * (1 + abs(rho))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("∫q dK", f"{mass:.3f}", help="should be ≈ 1 (some mass lives outside the strike range)")
    c2.metric("negative mass", f"{negmass:.4f}",
              delta="arbitrage!" if negmass > 1e-4 else "arb-free", delta_color="inverse")
    c3.metric("Lee wing bound b(1+|ρ|)", f"{lee:.2f} / 2.00",
              delta="violated" if lee > 2 else "ok", delta_color="inverse")
    c4.metric("ATM vol", f"{np.sqrt((a_ + b_*(rho*(0-m_)+np.sqrt(m_**2+sg**2)))/T):.1%}")

    with st.expander("Derivation: from butterfly prices to the density"):
        st.latex(r"B_{K,\lambda}=\lambda\!\left(C_{K-\frac1\lambda}-2C_K+C_{K+\frac1\lambda}\right)"
                 r"\quad\Rightarrow\quad \lambda B_{K,\lambda}\xrightarrow{\lambda\to\infty}"
                 r"\frac{\partial^2 C}{\partial K^2}=Z\,q(K)")
        st.markdown(
            "A butterfly is a unit-height tent of width $2/\\lambda$: its price is the "
            "(discounted) probability-weighted area under the tent, so scaled butterfly "
            "prices *are* the density. This is why negative density ⇔ a butterfly you'd be "
            "paid to own — free money unless your quotes are stale. In production the "
            "differencing is done on a **smoothed vol curve** (as here), never raw quotes.")


# ======================================================================= 3
def module_spanning():
    st.header("Spanning: calls are a basis — and the VIX is an application")
    st.markdown("Carr–Madan: any twice-differentiable European payoff decomposes exactly as")
    st.latex(r"g(S_T)=g(\kappa)+g'(\kappa)\,(S_T-\kappa)"
             r"+\int_0^{\kappa}\! g''(K)(K-S_T)^+dK+\int_{\kappa}^{\infty}\! g''(K)(S_T-K)^+dK")
    st.markdown(
        "so pricing $g$ needs nothing but the option strip — no model. With finitely many "
        "strikes you hold $g''(K_i)\\Delta K$ per strike; below, the exact piecewise-linear "
        "replication and its error as the strike grid coarsens.")

    S0, r, T = 100.0, 0.04, 0.25
    F = S0 * np.exp(r * T)
    S = np.linspace(40, 200, 1601)

    targets = {
        "log contract  −ln(S/F)  (the VIX payoff)": lambda s: -np.log(s / F),
        "Gaussian bump (bet on a range)": lambda s: np.exp(-0.5 * ((s - 110) / 12) ** 2),
        "squared deviation  (S/F − 1)²": lambda s: (s / F - 1) ** 2,
    }
    c1, c2 = st.columns([2, 1])
    tname = c1.selectbox("Target payoff g(S_T)", list(targets))
    dK = c2.select_slider("Strike spacing ΔK", [2.5, 5.0, 10.0, 20.0], value=5.0)
    g = targets[tname]

    strikes = np.arange(45, 200, dK)
    gk = g(strikes)
    seg = np.diff(gk) / np.diff(strikes)      # segment slopes
    w = np.diff(seg)                          # curvature weights, interior strikes
    Ki = strikes[1:-1]
    bonds, s0 = gk[0], seg[0]                 # the g(κ) and g'(κ) linear part
    repl = (bonds + s0 * (S - strikes[0])
            + np.sum(w[:, None] * np.maximum(S[None, :] - Ki[:, None], 0), axis=0))

    fig, axes = fig_axes(2)
    ax = axes[0]
    ax.plot(S, g(S), color=INK, label="target $g$")
    ax.plot(S, repl, color=BLUE, ls="--", label=f"call strip, ΔK={dK}")
    ax.set_xlabel("$S_T$"); ax.set_title("Static replication")
    ax.legend(fontsize=9)
    ax = axes[1]
    ax.bar(Ki, w, width=0.6 * dK,
           color=[GREEN if x >= 0 else MAGENTA for x in w])
    if "log contract" in tname:
        ax.plot(Ki, dK / Ki**2, color=YELLOW, lw=2,
                label="$\\Delta K/K^2$ — the VIX weights")
        ax.legend(fontsize=9)
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.set_xlabel("strike"); ax.set_title("Strip weights  $w_i \\approx g''(K_i)\\,\\Delta K$")
    st.pyplot(fig)
    st.caption(f"Plus the linear part carried by {bonds:+.3f} bonds and {s0:+.4f} stock — "
               "the $g(\\kappa)$ and $g'(\\kappa)$ terms of the decomposition.")

    err = np.max(np.abs(g(S) - repl)[(S > strikes[0]) & (S < strikes[-1])])
    st.metric("max replication error inside the strike range", f"{err:.4f}",
              help="halves roughly with ΔK for smooth payoffs — second-order accuracy")

    with st.expander("Why this is the VIX formula"):
        st.latex(r"\mathbb{E}^{\mathbb{Q}}\!\left[-2\ln\tfrac{S_T}{F}\right]"
                 r"\;\approx\;\sigma^2 T \qquad\Rightarrow\qquad "
                 r"\mathrm{VIX}^2=\frac{2}{T}\sum_i \frac{\Delta K_i}{K_i^2}\,e^{rT}Q(K_i)"
                 r"-\frac{1}{T}\left(\frac{F}{K_0}-1\right)^2")
        st.markdown(
            "The log contract's curvature is $g''(K)=1/K^2$, so its replicating strip holds "
            "$\\Delta K/K^2$ of each option — the yellow curve, which the computed weights sit "
            "on exactly. Cboe's VIX is nothing but this strip priced off the SPX chain: the "
            "flagship 'fear index' is a Carr–Madan spanning integral in production. Variance "
            "swaps are dealt the same way, which is why realized-vs-implied variance is the "
            "cleanest 'density disagreement' trade of all.")


# ======================================================================= 4
def module_inverse():
    st.header("The inverse problem: from disagreement to payoff")
    st.markdown(
        "Modules 1–3 are the market's side: what prices imply ($\\mathbb{Q}$) and how to build "
        "any payoff. This module adds *your* side — a real-world density $p$ ($\\mathbb{P}$-measure, "
        "here a block bootstrap of history) — and asks for the payoff maximizing expected log "
        "wealth subject to the FTAP budget $Z\\,\\mathbf{E}_q[g]=W$:")
    st.latex(r"g^*(S)=\frac{W}{Z}\,\frac{p(S)}{q(S)},\qquad"
             r"\mathbf{E}_p\!\left[\log\tfrac{g^*}{W}\right]=\mathrm{KL}(p\,\|\,q)-\log Z")
    st.markdown("**Your edge is the KL divergence.** The dials below shrink it honestly.")

    c = st.columns(4)
    ticker = c[0].text_input("Ticker", "SOXX")
    dte = c[1].slider("Days to expiry", 14, 180, 45)
    fraction = c[2].slider("Kelly fraction f", 0.0, 1.0, 0.5, 0.05,
                           help="p_f ∝ p^f q^(1−f); since (p_f/q)=(p/q)^f this *is* fractional Kelly")
    wealth = c[3].number_input("Budget ($)", 1_000, 1_000_000, 10_000, step=1_000)

    @st.cache_data(ttl=300, show_spinner="Fetching chain…")
    def load(ticker, dte):
        prov = pl.get_provider()
        return prov.chain(ticker, dte_target=dte), prov.history(ticker, years=8), type(prov).__name__

    chain, hist, prov_name = load(ticker, dte)
    if prov_name == "SyntheticProvider":
        st.info("No MASSIVE_API_KEY — synthetic data (SVI smile + fat-tailed history). "
                "Set the key and this page runs on live Massive quotes unchanged.")

    q = pl.implied_distribution(chain)
    horizon = max(int(chain.T * 252), 1)
    p_full = pl.historical_view(hist, horizon, chain.spot, q.grid, seed=1)
    p_view = pl.geometric_blend(p_full, q.pdf, q.grid, f=fraction)
    g = pl.log_optimal_payout(q.grid, p_view, q.pdf, wealth=wealth, Z=chain.Z, ratio_cap=8)
    ticket = pl.replicate(q.grid, g, chain)
    summary = pl.summarize(q.grid, p_view, q.pdf, g, ticket, chain, wealth=wealth)

    kl = float(np.trapezoid(p_view * np.log(np.maximum(p_view, 1e-300) /
                                            np.maximum(q.pdf, 1e-300)), q.grid))
    m = st.columns(4)
    m[0].metric("KL(p‖q)", f"{kl:.4f} nats", help="over the option's life")
    m[1].metric("annualized edge (replicated)", f"{summary['annualized (replicated)']:.1%}")
    m[2].metric("P(lose) under your view", f"{summary['P(lose money) under your view']:.0%}",
                help="log-optimal payoffs lose often, win big — by design")
    m[3].metric("ticket cost", f"${ticket.cost:,.0f}")

    fig, axes = fig_axes(2)
    ax = axes[0]
    ax.plot(q.grid, q.pdf, color=INK, label="market $q$")
    ax.plot(q.grid, p_view, color=BLUE, label=f"your $p_f$ (f={fraction})")
    ax.axvline(chain.spot, color=MUT, lw=1, ls=":")
    ax.set_xlabel("$S_T$"); ax.set_title("The disagreement"); ax.legend(fontsize=9)
    ax = axes[1]
    ax.plot(q.grid, g, color=INK, label="ideal $g^*$")
    ax.plot(q.grid, ticket.payoff(q.grid), color=BLUE, ls="--", label="replication")
    ax.axhline(wealth, color=MUT, lw=1, ls=":")
    ax.set_xlabel("$S_T$"); ax.set_title("The payoff that monetizes it"); ax.legend(fontsize=9)
    st.pyplot(fig)

    with st.expander("Order ticket (educational — mid fills, no fees; not investment advice)"):
        st.caption(f"{ticket.stock:+,.1f} shares, ${ticket.bonds:,.0f} bond face "
                   f"(negative = borrow). Left wing in OTM puts via parity.")
        legs = ticket.legs.assign(side=np.where(ticket.legs["qty"] >= 0, "BUY", "SELL"))
        st.dataframe(legs[["type", "strike", "side", "qty", "mid", "cost"]].round(2),
                     hide_index=True)


[module_payoff_algebra, module_smile_density,
 module_spanning, module_inverse][MODULES.index(module)]()
