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


# ------------------------------------------------------------ pedagogy kit
# Teaching happens through interaction, not paragraphs. The scaffold:
#   learn()      - 3 compact goals at the top of a module
#   narrate()    - ONE computed sentence restating the current control state
#                  in plain English; recomputed on every rerun
#   stepper()    - build-it-up stage selector for progressive disclosure
#   quiz()       - predict-before-reveal; feedback carries the explanation
#   experiment() - a numbered "try this" with the answer behind a toggle
# Prose outside these (and outside tabs/expanders) stays under ~50 words
# per block: the charts, the narration, and the learner do the explaining.

def learn(*points):
    st.caption("**You'll be able to:**  " + "  ·  ".join(points))


def narrate(text: str):
    st.info(text, icon="🧭")


def stepper(key: str, stages: list[str]) -> int:
    choice = st.radio("Build it up:", stages, key=key, horizontal=True)
    return stages.index(choice)


def quiz(key: str, question: str, options: list[str], correct: int, explain: str):
    with st.container(border=True):
        st.markdown(f"**🤔 Predict before you touch anything:** {question}")
        choice = st.radio("quiz", options, key=key, horizontal=True,
                          label_visibility="collapsed", index=None)
        if choice is not None:
            if options.index(choice) == correct:
                st.success(explain, icon="✅")
            else:
                st.error("Not quite — " + explain, icon="❌")


def experiment(key: str, title: str, prompt: str, reveal: str):
    with st.expander(f"🧪 Experiment — {title}"):
        st.markdown(prompt)
        if st.toggle("Show what should happen and why", key=key):
            st.markdown(reveal)


# ======================================================================= 1
def module_payoff_algebra():
    from scipy.stats import norm

    st.header("Payoff algebra: options as a vector space")
    learn("read a payoff diagram's kinks as strikes and its slope changes as quantities",
          "convert a payout into P&L by subtracting the compounded premium",
          "price a digital as the limit of a tight call spread and read it as Z·N(d₂)")

    presets = {
        "bull call spread": [("C", 95, 1), ("C", 105, -1)],
        "straddle": [("C", 100, 1), ("P", 100, 1)],
        "strangle": [("C", 110, 1), ("P", 90, 1)],
        "butterfly": [("C", 90, 1), ("C", 100, -2), ("C", 110, 1)],
        "iron condor": [("P", 80, 1), ("P", 90, -1), ("C", 110, -1), ("C", 120, 1)],
        "risk reversal": [("P", 90, -1), ("C", 110, 1)],
        "digital ≈ tight call spread": [("C", 100, 10), ("C", 100.1, -10)],
    }
    S0, r, T = 100.0, 0.04, 0.25
    Z = np.exp(-r * T)
    F = S0 * np.exp(r * T)

    name = st.selectbox("Structure", list(presets), key="m1_structure")

    quiz("m1_quiz",
         "If implied vol rises, the price of the **straddle**…",
         ["rises", "falls", "stays the same"], 0,
         "Rises — both legs are long convexity, so fatter tails raise the value of the call "
         "and the put together: long options are long vol. Check it: pick the straddle, sweep "
         "the vol slider below, and watch the premium column in the table climb.")

    c1, c2 = st.columns(2)
    vol = c1.slider("Implied vol for pricing", 0.10, 0.90, 0.30, 0.05, key="m1_vol")
    show_pnl = c2.toggle("Show P&L (net of premium)", value=True, key="m1_pnl")

    def leg_desc(t_, K_, q_):
        return f"{'long' if q_ > 0 else 'short'} {abs(q_):g}× {t_}@{K_:g}"

    legs_all = presets[name]
    descs = [leg_desc(*lg) for lg in legs_all]
    stages = [descs[0]] + [f"+ {d}" for d in descs[1:]]
    nshow = stepper(f"m1_step_{name}", stages) + 1
    vis = legs_all[:nshow]

    S = np.linspace(50, 150, 1201)
    Sx = np.concatenate([[0.0], np.geomspace(1.0, 2000.0, 12001)])

    def payout_of(Sv, subset):
        out = np.zeros_like(Sv, dtype=float)
        for t_, K_, q_ in subset:
            out += q_ * (np.maximum(Sv - K_, 0) if t_ == "C" else np.maximum(K_ - Sv, 0))
        return out

    def leg_premium(t_, K_, q_):
        cpx = float(pl.bs_call_price(S0, K_, T, r, vol))
        return q_ * cpx if t_ == "C" else q_ * (cpx - S0 + K_ * Z)   # parity: P = C - S0 + KZ

    prems = [leg_premium(*lg) for lg in vis]
    premium = float(sum(prems))
    total = payout_of(S, vis)
    pnl = total - premium / Z            # both legs of the comparison valued at T

    # Risk-neutral P(P&L>0): exact lognormal CDF between the P&L sign changes.
    # P&L is piecewise linear with last kink at max strike << Sx[-1], so the sign
    # pattern found on the extended grid is the sign pattern on all of (0, inf).
    pnl_x = payout_of(Sx, vis) - premium / Z

    def rn_cdf(x):                       # Q(S_T <= x), lognormal at the chosen vol
        x = max(float(x), 1e-12)
        return float(norm.cdf((np.log(x / S0) - (r - vol**2 / 2) * T) / (vol * np.sqrt(T))))

    def pnl_at(x):
        return float(payout_of(np.array([x]), vis)[0]) - premium / Z

    flips = np.where(np.diff((pnl_x > 0).astype(int)) != 0)[0]
    roots = []
    for i in flips:                      # bisect the exact P&L inside the bracketing cell
        a_, b_, fa = float(Sx[i]), float(Sx[i + 1]), float(pnl_x[i])
        for _ in range(40):
            m_ = 0.5 * (a_ + b_)
            if (pnl_at(m_) > 0) == (fa > 0):
                a_ = m_
            else:
                b_ = m_
        roots.append(0.5 * (a_ + b_))
    edges = [0.0] + roots + [np.inf]
    cdfs = [0.0] + [rn_cdf(x) for x in roots] + [1.0]
    prob = 0.0
    for j in range(len(edges) - 1):
        hi_e = edges[j + 1]
        test = 0.5 * (edges[j] + hi_e) if np.isfinite(hi_e) else edges[j] * 1.5 + 50.0
        if float(np.interp(test, Sx, pnl_x)) > 0:
            prob += cdfs[j + 1] - cdfs[j]
    # P&L is piecewise linear with kinks only at the strikes, so (barring a short
    # upper wing, absent from all presets) its extremes sit at a kink or at S=0.
    kink_pts = np.array([0.0] + [float(K_) for _, K_, _ in vis])
    pnl_kinks = payout_of(kink_pts, vis) - premium / Z
    worst = float(pnl_kinks.min())
    slope_up = sum(q_ for t_, K_, q_ in vis if t_ == "C")   # payout slope above top strike

    fig, (ax,) = fig_axes(height=3.8)
    leg_cols = [GREEN, YELLOW, MAGENTA, RED]
    for j, lg in enumerate(vis):
        ax.plot(S, payout_of(S, [lg]), color=leg_cols[j % 4], ls="--", lw=1.1,
                alpha=0.9, label=leg_desc(*lg))
    ax.plot(S, total, color=INK, lw=2.4, label="sum: payout at $T$")
    if show_pnl:
        ax.plot(S, pnl, color=BLUE, ls="--", label=f"P&L (premium {premium:+.2f} at $t$)")
        ax.axhline(0, color=MUT, lw=1)
    ax.axvline(S0, color=MUT, lw=1, ls=":")
    offs = [(-100, 28), (-100, -36), (18, -36), (18, 28)]
    for j, (t_, K_, q_) in enumerate(vis):
        yk = float(np.interp(K_, S, total))
        ax.annotate(f"kink: {leg_desc(t_, K_, q_)}", xy=(K_, yk),
                    xytext=offs[j % 4], textcoords="offset points",
                    fontsize=8, color="#52514e",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#fcfcfb", ec="none", alpha=0.9),
                    arrowprops=dict(arrowstyle="->", color=MUT, lw=1))
    ybase = np.concatenate([total, pnl]) if show_pnl else total
    ylo, yhi = float(ybase.min()), float(ybase.max())
    span = max(yhi - ylo, 1.0)
    ax.set_ylim(ylo - 0.20 * span, yhi + 0.34 * span)
    ax.set_xlabel("$S_T$"); ax.set_ylabel("value at expiry")
    ax.set_title(f"{name} — {nshow}/{len(legs_all)} legs shown")
    ax.legend(fontsize=8)
    st.pyplot(fig)

    def money(x):
        return ("−\\$" if x < 0 else "\\$") + f"{abs(x):,.2f}"

    bev_txt = ((" (P&L breakevens at " + ", ".join(f"{x:.2f}" for x in roots) + ")")
               if roots else "")
    # net-short calls above the top strike => the visible subset loses without
    # bound as S_T rises: never report a finite kink minimum in that state
    downside = (f"UNLIMITED loss above {max(K_ for _, K_, _ in vis):g}"
                if slope_up < -1e-9 else f"worst case {money(worst)}")
    upside = ("uncapped upside" if slope_up > 1e-9
              else f"best case {money(float(pnl_kinks.max()))}")
    narrate(f"Showing {nshow} of {len(legs_all)} legs of the {name}: this position "
            f"{'collects' if premium < 0 else 'costs'} {money(abs(premium))} up front and, "
            f"under the {vol:.0%}-vol risk-neutral lognormal, ends with positive P&L "
            f"{prob:.0%} of the time{bev_txt} — {downside}, {upside}.")

    st.dataframe(pd.DataFrame(vis, columns=["type", "K", "qty"])
                   .assign(premium=np.round(prems, 3)),
                 hide_index=True, width=380)

    # experiment (a): solve for the vol putting the 95/105 spread's breakeven at 100,
    # i.e. premium/Z = 5; spread vega is negative here (105 is nearer F in log-moneyness),
    # so premium is decreasing in vol and the bisection bracket [0.05, 2] is valid.
    prem_bs = float(pl.bs_call_price(S0, 95.0, T, r, vol)
                    - pl.bs_call_price(S0, 105.0, T, r, vol))
    be = 95.0 + prem_bs / Z
    lo_v, hi_v = 0.05, 2.0
    for _ in range(60):
        mid_v = 0.5 * (lo_v + hi_v)
        pm = float(pl.bs_call_price(S0, 95.0, T, r, mid_v)
                   - pl.bs_call_price(S0, 105.0, T, r, mid_v))
        if pm / Z > 5.0:
            lo_v = mid_v
        else:
            hi_v = mid_v
    vstar = 0.5 * (lo_v + hi_v)
    experiment(
        "m1_exp_a", "park the bull spread's breakeven exactly at 100",
        "Select the **bull call spread**, reveal both legs, keep P&L on, and sweep the vol "
        "slider. Watch where the blue P&L line crosses zero. Can you park that crossing "
        "exactly at $S_T = 100$? And why does it move at all — the payout never changes?",
        f"Only the premium moves. At {vol:.0%} vol the 95/105 spread costs {money(prem_bs)}, "
        f"so P&L crosses zero at 95 + {prem_bs:.2f}/{Z:.4f} = **{be:.2f}**. Raising vol "
        f"cheapens the spread — the short 105 call sits nearer the forward F = {F:.2f} in "
        f"log-moneyness, so it gains vega faster than the long 95 call — pulling the "
        f"breakeven down. It lands exactly on 100 at vol ≈ **{vstar:.1%}**.")

    dprem = 10.0 * float(pl.bs_call_price(S0, 100.0, T, r, vol)
                         - pl.bs_call_price(S0, 100.1, T, r, vol))
    d2 = (np.log(S0 / 100.0) + (r - vol**2 / 2) * T) / (vol * np.sqrt(T))
    nd2 = float(norm.cdf(d2))
    experiment(
        "m1_exp_b", "a premium that is a probability",
        "Select **digital ≈ tight call spread**, reveal both legs, and read the total premium "
        "off the table. The structure pays \\$1 whenever $S_T > 100$ (with a 10-cent ramp). "
        "Divide premium by the \\$1 width — what number is the market quoting you?",
        f"At {vol:.0%} vol the tight spread costs {money(dprem)} per \\$1 of payout, and the "
        f"Black–Scholes digital is worth Z·N(d₂) = {Z:.4f} × {nd2:.4f} = **{Z * nd2:.4f}** — "
        f"agreement to \\${abs(dprem - Z * nd2):.4f} (the residual is the 10-cent ramp). "
        f"Undiscounted, that says risk-neutral P(S_T > 100) = **{nd2:.1%}**: digital premium "
        f"÷ width *is* a probability, which is the statement −∂C/∂K = Z·Q(S_T > K).")

    t1, t2, t3 = st.tabs(["Intuition", "The formula", "Deeper"])
    with t1:
        st.markdown(
            "**Payoffs are Lego.** Four bricks — the bond $1$, the stock $S_T$, calls "
            "$(S_T-K)^+$ and puts $(K-S_T)^+$ — snap together by addition. Every kink in a "
            "diagram sits at a strike, and the slope change across it equals the signed "
            "quantity struck there. To reverse-engineer any hockey-stick picture, walk left "
            "to right: each time the slope jumps by $q$, write down \"$q$ calls at that "
            "strike.\"")
    with t2:
        st.latex(r"g(S_T)\;=\;\alpha\cdot 1\;+\;\delta\cdot S_T\;+\;"
                 r"\sum_i q_i\,(S_T-K_i)^+\;+\;\sum_j p_j\,(K_j-S_T)^+")
        st.markdown(
            "- $\\alpha$: bond face (price $Z$ each) — sets the level\n"
            "- $\\delta$: shares (price $S_0$ each) — sets the baseline slope\n"
            "- $q_i,\\ p_j$: option quantities at strikes $K_i, K_j$ — each adds one kink\n"
            "- puts are redundant given calls, by **put–call parity**:")
        st.latex(r"(K-S_T)^+=(S_T-K)^+-S_T+K"
                 r"\quad\Longrightarrow\quad P=C-S_0+K\,Z")
        st.caption("The payoff identity holds path by path, so it must hold in price too — "
                   "that parity is exactly how every put leg above is priced off the call "
                   "pricer.")
    with t3:
        st.markdown(
            "**The payout-vs-P&L trap.** A structure that \"never pays less than zero\" is "
            "not free money: P&L subtracts the compounded premium, and at fair prices the "
            "risk-neutral expected P&L of *any* package is exactly zero. The line that "
            "matters for risk is the blue one, not the black one.")
        st.latex(r"\lambda\left(C_K - C_{K+1/\lambda}\right)"
                 r"\xrightarrow{\;\lambda\to\infty\;}"
                 r"-\frac{\partial C}{\partial K}\;=\;Z\,\mathbb{Q}(S_T>K)")
        st.markdown(
            "**How dealers quote digitals.** Nobody trades the limit. The seller hedges with "
            "the over-replicating call spread on $[K-\\varepsilon, K]$, the buyer is shown "
            "the under-replicating one on $[K, K+\\varepsilon]$, and the client crosses the "
            "gap between the two. That is why digital markets widen near the strike into "
            "expiry — the spread hedge concentrates all its gamma there.")


# ======================================================================= 2
def module_smile_density():
    from scipy.stats import norm

    st.header("The smile and the density are the same object")
    learn("read a smile's left wing as the market price of crash risk",
          "turn any smile into its risk-neutral density with a second difference "
          "(Breeden–Litzenberger)",
          "run a desk's butterfly-arbitrage check: density ≥ 0 at every strike")

    # -- presets seed the sliders; a guard lets manual slider moves survive reruns
    PRESETS = {
        "flat (Black–Scholes)":      dict(a=0.020, b=0.01, rho=0.00,  m=0.00, sg=0.25),
        "equity-index skew":         dict(a=0.012, b=0.10, rho=-0.35, m=0.02, sg=0.25),
        "symmetric smile (FX-like)": dict(a=0.010, b=0.25, rho=0.00,  m=0.00, sg=0.15),
        "steep crash fear":          dict(a=0.012, b=0.30, rho=-0.80, m=0.00, sg=0.25),
    }
    preset = st.radio("Start from a market regime:", list(PRESETS),
                      key="m2_preset", horizontal=True, index=1)
    if st.session_state.get("m2_last_preset") != preset:
        st.session_state["m2_last_preset"] = preset
        for pk, v in PRESETS[preset].items():
            st.session_state["m2_" + pk] = v

    quiz("m2_quiz_rho",
         "Drag ρ from −0.35 up to +0.35. Which **far** tail of the density gets fatter?",
         ["the left tail", "the right tail", "both equally"], 1,
         "ρ tilts the smile: the wings grow with slopes $b(1-\\rho)$ (left) and "
         "$b(1+\\rho)$ (right), so positive ρ loads variance onto high strikes and "
         "fattens the far **right** tail. Equity-index smiles trade with ρ < 0 — crash "
         "fear lives in the left wing. (Near the money the effect can flip: the smile's "
         "local *slope* also bends the CDF.)")

    c = st.columns(5)
    a_ = c[0].slider("a (base var)", 0.001, 0.10, step=0.001, format="%.3f", key="m2_a")
    b_ = c[1].slider("b (wing slope)", 0.01, 1.50, step=0.01, key="m2_b")
    rho = c[2].slider("ρ (skew)", -0.99, 0.99, step=0.01, key="m2_rho")
    m_ = c[3].slider("m (shift)", -0.5, 0.5, step=0.01, key="m2_m")
    sg = c[4].slider("σ (smoothing)", 0.01, 1.0, step=0.01, key="m2_sg")

    # -- same core computation as before: SVI -> prices -> BL second difference
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
    qp = np.maximum(q, 0)

    # tail probabilities: SVI vs flat BS at the same ATM vol.
    # SVI side computed EXACTLY from the call-price slope, P(S_T<K) = 1 + e^{rT} dC/dK,
    # not by integrating the plotted grid (which truncates mass below K=40).
    # Flat lognormal: P(S_T < xF) = Phi((ln x + w0/2)/sqrt(w0)) with w0 = sigma_atm^2 T.
    w0 = a_ + b_ * (rho * (0 - m_) + np.sqrt(m_**2 + sg**2))
    atm = float(np.sqrt(w0 / T))
    iv15 = float(np.interp(-0.15, k, iv))
    Kthr, Kdeep = 0.85 * F, 0.70 * F

    def svi_call(Kx):
        kln = np.log(Kx / F)
        wv = a_ + b_ * (rho * (kln - m_) + np.sqrt((kln - m_) ** 2 + sg**2))
        return float(pl.bs_call_price(S0, Kx, T, r, np.sqrt(max(wv, 1e-8) / T)))

    def svi_cdf(Kx, h=0.25):
        return float(np.clip(1 + np.exp(r * T) * (svi_call(Kx + h) - svi_call(Kx - h)) / (2 * h), 0, 1))

    tail, deep = svi_cdf(Kthr), svi_cdf(Kdeep)
    flat_tail = lambda x: float(norm.cdf((np.log(x) + 0.5 * w0) / np.sqrt(w0)))

    fig, axes = fig_axes(2, height=3.6)
    ax = axes[0]
    ax.plot(k, iv, color=BLUE)
    kthr = float(np.log(0.85))
    ax.axvspan(k[0], kthr, color=MAGENTA, alpha=0.18, lw=0)
    ax.annotate("same information →", xy=(kthr, float(np.interp(kthr, k, iv))),
                xytext=(14, 22), textcoords="offset points", fontsize=8, color=MAGENTA,
                arrowprops=dict(arrowstyle="->", color=MAGENTA, lw=1.2))
    ax.margins(y=0.15)
    ax.plot([0.0], [atm], marker="o", ms=5, color=BLUE)
    ax.annotate(f"ATM {atm:.0%}", xy=(0, atm), xytext=(12, 12),
                textcoords="offset points", fontsize=8, color=MUT,
                bbox=dict(boxstyle="round,pad=0.2", fc="#fcfcfb", ec="none", alpha=0.9),
                arrowprops=dict(arrowstyle="-", color=MUT, lw=0.8))
    ax.set_xlabel("log-moneyness  $k=\\ln(K/F)$"); ax.set_ylabel("implied vol")
    ax.set_title("SVI smile (the quoted object)")
    ax = axes[1]
    ax.plot(Kq, q, color=INK)
    lmask = Kq <= Kthr
    ax.fill_between(Kq, qp, 0, where=lmask, color=MAGENTA, alpha=0.35, lw=0)
    ax.annotate(f"← same information\n$P(S_T<{Kthr:.0f})$ = {tail:.1%}",
                xy=(0.80 * F, float(np.interp(0.80 * F, Kq, qp))),
                xytext=(0.03, 0.72), textcoords="axes fraction", fontsize=8,
                color=MAGENTA, arrowprops=dict(arrowstyle="->", color=MAGENTA, lw=1.2))
    im = int(np.argmax(q))
    ax.annotate(f"mode ≈ {Kq[im]:.0f}", xy=(Kq[im], q[im]), xytext=(10, -2),
                textcoords="offset points", fontsize=8, color=MUT)
    neg = q < 0
    if neg.any():
        ax.fill_between(Kq, q, 0, where=neg, color=RED, alpha=0.6,
                        label="q < 0: butterfly arbitrage")
        ax.legend(fontsize=9)
    ax.axhline(0, color=MUT, lw=1)
    ax.set_xlabel("$K$"); ax.set_ylabel("density")
    ax.set_title("Implied density  $q=\\partial^2C/\\partial K^2/Z$  (the same object)")
    st.pyplot(fig)

    narrate(f"At k=−0.15 (strike {F * np.exp(-0.15):.0f}, ≈86% of the forward) this smile "
            f"quotes {iv15:.1%} vs {atm:.1%} ATM ({(iv15 - atm) * 100:+.1f} vol points); "
            f"that shape gives the shaded tail $S_T<{Kthr:.0f}$ a {tail:.1%} risk-neutral "
            f"probability vs {flat_tail(0.85):.1%} under a flat Black–Scholes smile at the "
            f"same ATM vol — and the deep tail $S_T<{Kdeep:.0f}$ holds {deep:.1%} "
            f"vs {flat_tail(0.70):.1%} flat.")

    mass = float(np.trapezoid(qp, Kq))
    negmass = max(0.0, float(-np.trapezoid(np.minimum(q, 0), Kq)))
    lee = b_ * (1 + abs(rho))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("∫q dK", f"{mass:.3f}",
              help="should be ≈ 1 (some mass lives outside the strike range)")
    c2.metric("negative mass", f"{negmass:.4f}",
              delta="arbitrage!" if negmass > 1e-4 else "arb-free", delta_color="inverse")
    c3.metric("Lee wing bound b(1+|ρ|)", f"{lee:.2f} / 2.00",
              delta="violated" if lee > 2 else "ok", delta_color="inverse")
    c4.metric("ATM vol", f"{atm:.1%}")

    # exact price of the (K−1, K, K+1) butterfly at the worst strike: grid step is
    # 0.2, so ±5 indices = ±$1; B = C(K−1) − 2C(K) + C(K+1) ≈ Z·q(K)·1²
    i0 = int(np.argmin(q[25:-25])) + 25
    j = i0 + 1
    bfly = float(C[j - 5] - 2 * C[j] + C[j + 5])
    experiment("m2_exp_arb", "manufacture free money",
               "Push **b → 1.2** and **σ → 0.05** with ρ near −0.8. Watch the density "
               "panel and the *negative mass* metric.",
               f"Scaled butterfly prices **are** the density (see *The formula*), so "
               f"q < 0 means a butterfly with a negative price. Off your current smile "
               f"the worst spot is q({Kq[i0]:.0f}) = {q[i0]:.4f}: the "
               f"(K−1, K, K+1) butterfly there — a payoff between \\$0 and \\$1, never "
               f"negative — prices at **\\${bfly:+.4f}**. "
               + ("The market would *pay you* to own a non-negative payoff: free money, "
                  "unless the quotes are stale. " if q[i0] < 0 else
                  "Still positive, so no arbitrage yet — push further. ")
               + "Desks run exactly this q ≥ 0 scan before publishing a surface.")

    loc = (qp[1:-1] > qp[:-2]) & (qp[1:-1] >= qp[2:]) & (qp[1:-1] > 0.05 * max(float(qp.max()), 1e-12))
    n_modes = int(loc.sum())
    if n_modes <= 1:
        mode_msg = "Your current density has **one hump** — and it always will."
    elif negmass > 1e-4:
        mode_msg = (f"You count **{n_modes} local peaks**, but negative mass reads "
                    f"{negmass:.4f}: the extra hump was bought with butterfly "
                    "arbitrage next to it — not a market, a mispricing.")
    else:
        mode_msg = (f"You managed to dent the top into {n_modes} local maxima, "
                    "but not two genuinely separated humps.")
    experiment("m2_exp_bimodal", "try to make it bimodal",
               "Before an earnings print or a court ruling, traders expect *two* "
               "outcomes — jump or dump. Try any slider combination that gives the "
               "density two separated humps.",
               mode_msg + " SVI's five parameters allow exactly one smile minimum, and "
               "an arbitrage-free single-minimum smile produces an (essentially) "
               "unimodal density. Genuinely bimodal markets — earnings, FDA decisions, "
               "referenda — need richer smiles: desks switch to mixture models or "
               "non-parametric slice fits around such events.")

    tab_i, tab_f, tab_d = st.tabs(["Intuition", "The formula", "Deeper"])
    with tab_i:
        st.markdown(
            "A call's price is what the market charges for *all outcomes above K*, so "
            "call prices across strikes are a photograph of the probability density — "
            "taken in volatility coordinates. Lift the smile's left wing and you have "
            "made crash protection dearer: that *is* moving probability into the deep "
            "left tail. One object, two coordinate systems; every slider bends both "
            "panels at once.")
    with tab_f:
        st.latex(r"w(k) \;=\; a + b\left(\rho\,(k-m) + \sqrt{(k-m)^2+\sigma^2}\right),"
                 r"\qquad k=\ln(K/F),\quad w=\sigma_{\mathrm{imp}}^2\,T")
        st.caption("a = overall variance level · b = wing steepness · ρ = tilt (which "
                   "wing is steeper) · m = shifts the vertex · σ = rounds the vertex. "
                   "The wings grow linearly with slopes b(1±ρ); Lee's moment formula "
                   "requires b(1+|ρ|) ≤ 2.")
        st.latex(r"q(K)\;=\;\frac{1}{Z}\,\frac{\partial^2 C}{\partial K^2}"
                 r"\;=\;\lim_{h\to 0}\;\frac{C(K\!-\!h)-2C(K)+C(K\!+\!h)}{Z\,h^{2}}")
        st.caption("Breeden–Litzenberger: the second difference of the call curve is a "
                   "butterfly's price; divided by Z·h², it converges to the density. "
                   "Z = discount factor, C(K) = call priced off the smile.")
    with tab_d:
        st.markdown(
            "Before a surface is published, a desk runs three checks. **Butterfly**: "
            "q ≥ 0 at every strike — exactly the second difference above, taken on the "
            "smoothed fit, never on raw quotes. **Calendar**: total variance w(k,T) "
            "non-decreasing in T at fixed k, or calendar spreads arbitrage you. "
            "**Wings**: Lee's bound caps how fast w may grow in |k|, or far-strike "
            "moments explode. SVI is the industry's default slice precisely because "
            "these checks are cheap — sufficient no-arbitrage conditions are known in "
            "closed form, and its linear wings match Lee's bound by construction. "
            "Where one slice fails (event risk, bimodal outcomes), desks bolt on "
            "mixtures or fit slices non-parametrically.")


# ======================================================================= 3
def module_spanning():
    st.header("Spanning: calls are a basis — and the VIX is an application")
    st.markdown(
        "Carr–Madan: any smooth European payoff is exactly **a bond + a stock position "
        "+ a strip of options** — so pricing it needs no model, only the option chain. "
        "Build the replication one layer at a time below.")
    learn("decompose a payoff into level (bond) + slope (stock) + curvature (options)",
          "read strip weights as $g''(K)\\,\\Delta K$ and predict their shape from the target's bend",
          "explain why the VIX is a $1/K^2$ call strip and why skew lifts it above ATM vol")

    quiz("m3_quiz",
         "For the parabola target $(S/F-1)^2$, the strip weights across strikes will be…",
         ["increasing with K", "constant", "hump-shaped near F"], 1,
         "Weights are $w_i \\approx g''(K_i)\\,\\Delta K$, and a parabola's curvature "
         "$g''=2/F^2$ is the same at every strike — each one buys the same amount of bend.")

    S0, r, T = 100.0, 0.04, 0.25
    F, Z = S0 * np.exp(r * T), np.exp(-r * T)
    S = np.linspace(40, 200, 1601)

    targets = {
        "log contract  −ln(S/F)  (the VIX payoff)": lambda s: -np.log(s / F),
        "Gaussian bump (bet on a range)": lambda s: np.exp(-0.5 * ((s - 110) / 12) ** 2),
        "squared deviation  (S/F − 1)²": lambda s: (s / F - 1) ** 2,
    }
    c1, c2 = st.columns([2, 1])
    tname = c1.selectbox("Target payoff g(S_T)", list(targets), key="m3_target")
    dK = c2.select_slider("Strike spacing ΔK", [2.5, 5.0, 10.0, 20.0], value=5.0,
                          key="m3_dk")
    g = targets[tname]

    def build(dc, gfun):
        # piecewise-linear interpolant of g at the strikes == bond + stock + call strip
        ks = np.arange(45, 200, dc)
        gk = gfun(ks)
        seg = np.diff(gk) / np.diff(ks)          # segment slopes
        return ks, ks[1:-1], np.diff(seg), gk[0], seg[0]

    def strip_err(dc, gfun):
        ks, Kin, wt, b0, sl = build(dc, gfun)
        rp = b0 + sl * (S - ks[0]) + wt @ np.maximum(S[None, :] - Kin[:, None], 0.0)
        e = np.where((S > ks[0]) & (S < ks[-1]), np.abs(gfun(S) - rp), 0.0)
        return float(e.max()), float(S[np.argmax(e)])

    strikes, Ki, w, bonds, s0 = build(dK, g)
    kappa, N = strikes[0], len(Ki)
    gS, lin = g(S), bonds + s0 * (S - strikes[0])
    inside = (S > strikes[0]) & (S < strikes[-1])
    err, err_loc = strip_err(dK, g)
    err2, _ = strip_err(dK / 2, g)               # for the convergence claim

    stage = stepper("m3_stage", ["1 · target g", "2 · + bond (level)",
                                 "3 · + stock (tangent at κ)", "4 · + call strip"])
    n = N
    if stage == 3:
        n = st.slider("strikes added, left to right", 0, N, N, key=f"m3_nadd_{int(dK*10)}")

    labels = {1: f"bond only: {bonds:+.3f} × 1", 2: "bond + stock (tangent)",
              3: f"strip: {n}/{N} strikes"}
    partial = None
    if stage == 1:
        partial = np.full_like(S, bonds)
    elif stage == 2:
        partial = lin
    elif stage == 3:
        partial = lin + w[:n] @ np.maximum(S[None, :] - Ki[:n, None], 0.0)

    # curvature profile, for the stage-1 annotation & narration
    g2 = np.gradient(np.gradient(gS, S), S)
    core, Score = np.abs(g2[12:-12]), S[12:-12]
    flat_curv = np.ptp(core) < 0.05 * core.max()
    Sc = 120.0 if flat_curv else float(Score[np.argmax(core)])

    fig, axes = fig_axes(2)
    ax = axes[0]
    ax.plot(S, gS, color=INK, label="target $g$")
    if partial is not None:
        ax.plot(S, partial, color=BLUE, ls="--", label=labels[stage])
        ax.fill_between(S, partial, gS, color=MAGENTA, alpha=0.15)
    arrow = dict(arrowstyle="->", color=MUT, lw=1.2)
    if stage == 0:
        txt = ("constant curvature — every strike\nwill get the same weight" if flat_curv
               else f"bends hardest near S≈{Sc:.0f} —\noptions must buy this")
        ax.annotate(txt, xy=(Sc, g(Sc)), xytext=(25, 30),
                    textcoords="offset points", fontsize=8, color=MUT, arrowprops=arrow)
    elif stage == 1:
        ax.annotate(f"bond: flat at g(κ)={bonds:+.2f}\nno slope, no bend",
                    xy=(140.0, bonds), xytext=(10, 30),
                    textcoords="offset points", fontsize=8, color=MUT, arrowprops=arrow)
    elif stage == 2 or (stage == 3 and n == 0):
        xs = kappa + 20
        ax.annotate(f"tangent at κ={kappa:g}:\nslope g'(κ)≈{s0:+.3f}",
                    xy=(xs, float(np.interp(xs, S, lin))), xytext=(15, -35),
                    textcoords="offset points", fontsize=8, color=MUT, arrowprops=arrow)
    elif n < N:
        Kf = Ki[n - 1]
        ax.annotate(f"frontier: {w[n-1]:+.4f} calls @ K={Kf:g}\nbend the line here",
                    xy=(Kf, float(np.interp(Kf, S, partial))), xytext=(20, -35),
                    textcoords="offset points", fontsize=8, color=MUT, arrowprops=arrow)
    else:
        ax.annotate(f"worst residual {err:.4f}\n(between strikes)",
                    xy=(err_loc, g(err_loc)), xytext=(20, 30),
                    textcoords="offset points", fontsize=8, color=MUT, arrowprops=arrow)
    ax.set_xlabel("$S_T$"); ax.set_title("Static replication"); ax.legend(fontsize=9)

    ax = axes[1]
    n_add = n if stage == 3 else 0
    if n_add:
        ax.bar(Ki[:n_add], w[:n_add], width=0.6 * dK,
               color=[GREEN if x >= 0 else MAGENTA for x in w[:n_add]])
    if n_add < N:
        ax.bar(Ki[n_add:], w[n_add:], width=0.6 * dK, alpha=0.25,
               color=[GREEN if x >= 0 else MAGENTA for x in w[n_add:]],
               label="not yet added" if stage == 3 else "the strip to come")
    if "log contract" in tname:
        ax.plot(Ki, dK / Ki**2, color=YELLOW, lw=2, label="$\\Delta K/K^2$ — the VIX weights")
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.set_xlabel("strike"); ax.set_title("Strip weights  $w_i \\approx g''(K_i)\\,\\Delta K$")
    if n_add < N or "log contract" in tname:
        ax.legend(fontsize=9)
    st.pyplot(fig)

    lo, hi = strikes[0], strikes[-1]
    if stage == 0:
        if flat_curv:
            narrate(f"This parabola bends at the constant rate g''=2/F²≈{2/F**2:.6f} "
                    f"everywhere, so each of the {N} strikes will carry the same weight "
                    f"≈{2/F**2*dK:.5f} — bonds and shares contribute zero curvature.")
        else:
            narrate(f"This target bends hardest near S≈{Sc:.0f} (|g''|≈{core.max():.4f}); "
                    "a bond has no slope and a share has no bend, so every bit of that "
                    "curvature must be bought with options.")
    elif stage == 1:
        gap = float(np.max(np.abs(gS - partial)[inside]))
        narrate(f"{bonds:+.3f} bonds pin the level at g(κ)={bonds:.3f} for every $S_T$, "
                f"but the gap to the target is still {gap:.2f} at its worst.")
    elif stage == 2:
        gap = float(np.max(np.abs(gS - partial)[inside]))
        narrate(f"Adding {s0:+.4f} shares tilts the line to the tangent at κ={kappa:g} — "
                f"exact there, yet still off by up to {gap:.2f} where the target curves.")
    elif n < N:
        gap = float(np.max(np.abs(gS - partial)[inside]))
        narrate(f"With {n} of {N} strikes at ΔK={dK:g} the strip is within {gap:.4f} of "
                f"the target across ({lo:g}, {hi:g}); all {N} get it to {err:.4f}, and "
                f"halving ΔK to {dK/2:g} cuts that to {err2:.4f} (×{err/err2:.1f} better) "
                "— quadratic convergence.")
    else:
        narrate(f"With all {N} strikes at ΔK={dK:g} the strip is within {err:.4f} of the "
                f"target across ({lo:g}, {hi:g}); halving ΔK to {dK/2:g} cuts the error "
                f"to {err2:.4f} — ×{err/err2:.1f}, the quadratic convergence of a "
                "piecewise-linear fit.")

    m1, m2, m3 = st.columns(3)
    m1.metric("max strip error inside strikes", f"{err:.4f}",
              help="quarters (≈×4) when ΔK halves — piecewise-linear, O(ΔK²)")
    m2.metric("bond legs (level g(κ))", f"{bonds:+.3f}")
    m3.metric("stock legs (slope g'(κ))", f"{s0:+.4f}")

    # -- experiment (a): where coarse grids fail, computed for the Gaussian bump
    gb = targets["Gaussian bump (bet on a range)"]
    eb_f, loc_f = strip_err(2.5, gb)
    eb_c, loc_c = strip_err(20.0, gb)
    experiment(
        "m3_exp1", "coarse grids fail where curvature lives",
        "Pick the **Gaussian bump** target and stage 4 with all strikes, then step "
        "ΔK from 2.5 up to 20. Watch *where* the shaded gap grows, not just how much.",
        f"At ΔK=2.5 the worst error is {eb_f:.4f} (near S≈{loc_f:.0f}); at ΔK=20 it is "
        f"{eb_c:.4f} — about {eb_c/eb_f:.0f}× worse — and it concentrates near "
        f"S≈{loc_c:.0f}, the bump's peak, where $|g''|$ is largest. Straight segments "
        "cut the corner exactly where the payoff bends fastest, so desks spend their "
        "strike budget where the curvature (the gamma) lives.")

    # -- experiment (b): the log strip on module 2's skew vs one flat ATM vol.
    #    Strip price = Z·g(κ) + g'(κ)(S0 − κZ) + Σ w_i C(K_i);  E_Q[−ln(S_T/F)] = price/Z,
    #    and a VIX-style vol is sqrt(2/T · E_Q[−ln(S_T/F)]). Under flat BS vol σ,
    #    E[−ln(S_T/F)] = σ²T/2 exactly, so the flat strip must recover ATM vol.
    a2, b2, r2, m2_, sg2 = 0.012, 0.10, -0.35, 0.02, 0.25   # module-2 SVI defaults
    def svi_iv(Karr):
        k = np.log(np.asarray(Karr, float) / F)
        return np.sqrt((a2 + b2 * (r2 * (k - m2_) + np.sqrt((k - m2_) ** 2 + sg2**2))) / T)
    iv_atm = float(svi_iv([F])[0])
    ks_l, Ki_l, w_l, b_l, s_l = build(2.5, lambda s: -np.log(s / F))
    c_svi = float(w_l @ pl.bs_call_price(S0, Ki_l, T, r, svi_iv(Ki_l)))
    c_flat = float(w_l @ pl.bs_call_price(S0, Ki_l, T, r, iv_atm))
    lin_px = b_l * Z + s_l * (S0 - ks_l[0] * Z)             # bond + stock legs, model-free
    vol_svi = float(np.sqrt(2 / T * (lin_px + c_svi) / Z))
    vol_flat = float(np.sqrt(2 / T * (lin_px + c_flat) / Z))
    experiment(
        "m3_exp2", "why the VIX sits above ATM vol",
        "Select the **log contract** at ΔK=2.5 — the strip below is (2/T ×) the VIX "
        "portfolio. Prediction: does its price change if every call is re-priced on "
        "module 2's *skewed* SVI smile instead of one flat ATM vol?",
        f"Priced on a single flat vol equal to ATM ({iv_atm:.1%}), the strip implies a "
        f"VIX-style vol of {vol_flat:.1%} — it recovers ATM, as Black–Scholes demands "
        "($\\mathbf{E}[-\\ln(S_T/F)]=\\sigma^2 T/2$). Re-pricing the *same weights* on "
        f"module 2's default skew costs {c_svi:.4f} vs {c_flat:.4f} per contract "
        f"(+{c_svi - c_flat:.4f}) and implies {vol_svi:.1%} — "
        f"{(vol_svi - vol_flat) * 100:.1f} vol points above ATM. The $1/K^2$ weights "
        "load on low strikes, exactly the OTM puts that skew makes dear: this is why "
        "VIX ≈ a 30-day variance-swap strike > ATM implied.")

    t_int, t_frm, t_deep = st.tabs(["Intuition", "The formula", "Deeper"])
    with t_int:
        st.markdown(
            "A bond sets the level; a stock sets the slope; neither can *bend*. Options "
            "are pure curvature — a call kinks at its strike and nowhere else. So build "
            "any smooth payoff like a spline: pin level and tangent at one anchor κ, "
            "then buy $g''(K)\\,\\Delta K$ of bend at each strike — long calls where the "
            "target curves up, short where it curves down. Your strike grid is your "
            "resolution; curvature you can't buy is error you must hold.")
    with t_frm:
        st.latex(r"g(S_T)=g(\kappa)+g'(\kappa)\,(S_T-\kappa)"
                 r"+\int_0^{\kappa}\! g''(K)(K-S_T)^+dK+\int_{\kappa}^{\infty}\! g''(K)(S_T-K)^+dK")
        st.markdown(
            "- $g(\\kappa)$ — bonds: the level at the anchor $\\kappa$\n"
            "- $g'(\\kappa)(S_T-\\kappa)$ — shares: the tangent line at $\\kappa$\n"
            "- $g''(K)\\,dK$ — option quantity at strike $K$: curvature density\n"
            "- $(K-S_T)^+,\\ (S_T-K)^+$ — put payoffs below $\\kappa$, calls above "
            "(interchangeable by put–call parity)\n"
            "- exact for any twice-differentiable $g$, for **any** distribution of $S_T$ "
            "— that is why pricing the strip needs no model")
    with t_deep:
        st.latex(r"\mathrm{VIX}^2=\frac{2}{T}\sum_i \frac{\Delta K_i}{K_i^2}\,e^{rT}Q(K_i)"
                 r"-\frac{1}{T}\left(\frac{F}{K_0}-1\right)^2")
        st.markdown(
            "Cboe's VIX is exactly this strip priced off the SPX chain: OTM quotes "
            "$Q(K_i)$, the log contract's weights $\\Delta K_i/K_i^2$ (its curvature is "
            "$g''=1/K^2$), anchor $K_0$ = first strike below $F$, and the "
            "$(F/K_0-1)^2$ term correcting the bond-and-stock part. The 'fear index' is "
            "a Carr–Madan integral in production. Variance swaps are dealt the same "
            "way: the dealer who sells realized variance buys the strip and "
            "delta-hedges the log contract, so realized-vs-implied variance is the "
            "cleanest density-disagreement trade there is. Wing truncation biases the "
            "strip low; coarse strikes bias it *high* — the piecewise-linear strip "
            "over-replicates the convex log payoff, so discretization overstates the "
            "variance strike. Crash skew fattens the $1/K^2$-weighted put cost and "
            "holds VIX above ATM implied vol.")


# ======================================================================= 4
def module_inverse():
    st.header("The inverse problem: from disagreement to payoff")
    learn("turn a density disagreement into the growth-optimal payoff $g^*=(W/Z)\\,p/q$",
          "read KL(p‖q) as your edge in nats and the Kelly fraction $f$ as honest shrinkage",
          "price the cost of conviction: P(loss) and median wealth from a 20,000-path simulation")

    quiz("m4_quiz",
         "Below there is a Kelly-fraction slider $f$. Set $f=0$ (zero conviction). "
         "The optimal payoff $g^*$ becomes…",
         ["a zero-coupon bond", "the stock", "zero — no position"], 0,
         "With f = 0 your blended density $p_f$ collapses onto the market's $q$, the ratio "
         "$p_f/q \\equiv 1$, and $g^* = W/Z$ — a constant payoff: a zero-coupon bond costing "
         "exactly your budget and paying it back grown at the risk-free rate ($W e^{rT}$). "
         "No disagreement, no trade; options enter only where the densities differ.")

    c = st.columns(4)
    ticker = c[0].text_input("Ticker", "SOXX", key="m4_ticker")
    dte = c[1].slider("Days to expiry", 14, 180, 45, key="m4_dte")
    fraction = c[2].slider("Kelly fraction f", 0.0, 1.0, 0.5, 0.05, key="m4_f",
                           help="p_f ∝ p^f q^(1−f); since (p_f/q)=(p/q)^f this *is* fractional Kelly")
    wealth = c[3].number_input("Budget ($)", 1_000, 1_000_000, 10_000, step=1_000, key="m4_wealth")

    @st.cache_data(ttl=300, show_spinner="Fetching chain…")
    def load(ticker, dte):
        prov = pl.get_provider()
        return prov.chain(ticker, dte_target=dte), prov.history(ticker, years=8), type(prov).__name__

    @st.cache_data(ttl=300, show_spinner=False)
    def solve(ticker, dte, fraction, wealth):
        """chain -> q -> p_f -> Kelly payoff -> replicated ticket, one cacheable step."""
        chain, hist, _ = load(ticker, dte)
        q = pl.implied_distribution(chain)
        horizon = max(int(chain.T * 252), 1)
        p_full = pl.historical_view(hist, horizon, chain.spot, q.grid, seed=1)
        p_view = pl.geometric_blend(p_full, q.pdf, q.grid, f=fraction)
        g = pl.log_optimal_payout(q.grid, p_view, q.pdf, wealth=wealth, Z=chain.Z, ratio_cap=8)
        ticket = pl.replicate(q.grid, g, chain)
        summary = pl.summarize(q.grid, p_view, q.pdf, g, ticket, chain, wealth=wealth)
        kl = float(np.trapezoid(p_view * np.log(np.maximum(p_view, 1e-300) /
                                                np.maximum(q.pdf, 1e-300)), q.grid))
        return chain, q, p_view, g, ticket, summary, kl

    @st.cache_data(ttl=300, show_spinner=False)
    def simulate(ticker, dte, fraction, wealth, n=20_000, seed=11):
        """Terminal wealth of the REPLICATED ticket over n draws of S_T ~ p_f (inverse-CDF)."""
        _, q, p_view, _, ticket, _, _ = solve(ticker, dte, fraction, wealth)
        cdf = np.concatenate([[0.0], np.cumsum((p_view[1:] + p_view[:-1]) / 2 * np.diff(q.grid))])
        cdf = cdf / cdf[-1]
        u = np.random.default_rng(seed).random(n)
        return ticket.payoff(np.interp(u, cdf, q.grid))

    _, _, prov_name = load(ticker, dte)
    if prov_name == "SyntheticProvider":
        st.info("No MASSIVE_API_KEY — synthetic data (SVI smile + fat-tailed history). "
                "Set the key and this page runs on live Massive quotes unchanged.")

    chain, q, p_view, g, ticket, summary, kl = solve(ticker, dte, fraction, wealth)
    grid = q.grid
    # display ratio capped at 8 to mirror log_optimal_payout(ratio_cap=8)
    ratio_c = np.minimum(p_view / np.maximum(q.pdf, 1e-12), 8.0)

    stage = stepper("m4_stage", ["1 · two densities", "2 · the ratio p/q",
                                 "3 · payoff = (W/Z) × ratio", "4 · make it tradable"])

    def draw_densities(ax):
        ax.plot(grid, q.pdf, color=INK, label="market $q$")
        ax.plot(grid, p_view, color=BLUE, label=f"your $p_f$ (f={fraction:.2f})")
        ax.axvline(chain.spot, color=MUT, lw=1, ls=":")
        diff = p_view - q.pdf
        if diff.max() > 1e-5:
            j = int(np.argmax(diff))
            ax.annotate("your $p_f$ piles extra\nmass here", xy=(grid[j], p_view[j]),
                        xytext=(0.72, 0.55), textcoords="axes fraction", fontsize=8,
                        color=INK, arrowprops=dict(arrowstyle="->", color=MUT))
        ax.set_xlabel("$S_T$"); ax.set_title("The disagreement"); ax.legend(fontsize=9)

    def draw_ratio(ax):
        ax.plot(grid, ratio_c, color=MAGENTA)
        ax.axhline(1, color=MUT, lw=1)
        ax.set_yscale("log")
        if ratio_c.max() > 1.02:
            i = int(np.argmax(ratio_c))
            ax.annotate(f">1: you think ${grid[i]:.0f} is\nunderpriced ({ratio_c[i]:.1f}× the market)",
                        xy=(grid[i], ratio_c[i]), xytext=(0.05, 0.80), textcoords="axes fraction",
                        fontsize=8, color=INK, arrowprops=dict(arrowstyle="->", color=MUT))
        else:
            ax.annotate("f = 0: $p_f = q$, ratio ≡ 1 — nothing to trade", xy=(0.5, 0.6),
                        xycoords="axes fraction", ha="center", fontsize=8, color=INK)
        ax.set_xlabel("$S_T$"); ax.set_ylabel("$p_f/q$ (log scale)")
        ax.set_title("The mispricing map  $p_f/q$")

    def draw_payoff(ax, with_repl):
        ax.plot(grid, g, color=INK, label="ideal $g^*=(W/Z)\\,p_f/q$")
        if with_repl:
            ax.plot(grid, ticket.payoff(grid), color=BLUE, ls="--", label="listed-strike replication")
            ax.annotate("flat past the last strike:\nbounded bet",
                        xy=(grid[-1], float(ticket.payoff(grid[-1:])[0])),
                        xytext=(0.60, 0.10), textcoords="axes fraction", fontsize=8,
                        color=INK, arrowprops=dict(arrowstyle="->", color=MUT))
        ax.axhline(wealth, color=MUT, lw=1, ls=":")
        cross = np.where(np.diff(np.sign(g - wealth)) != 0)[0]
        if len(cross):
            i = int(cross[-1])
            ax.annotate(f"breakeven: $g^*=W$\nat $S_T \\approx {grid[i]:.0f}$", xy=(grid[i], wealth),
                        xytext=(0.05, 0.75), textcoords="axes fraction", fontsize=8,
                        color=INK, arrowprops=dict(arrowstyle="->", color=MUT))
        ax.set_xlabel("$S_T$"); ax.set_title("The payoff that monetizes it"); ax.legend(fontsize=9)

    if stage == 0:
        fig, (ax,) = fig_axes()
        draw_densities(ax)
    elif stage == 1:
        fig, axes = fig_axes(2)
        draw_densities(axes[0]); draw_ratio(axes[1])
    elif stage == 2:
        fig, axes = fig_axes(2)
        draw_ratio(axes[0]); draw_payoff(axes[1], with_repl=False)
    else:
        fig, axes = fig_axes(2)
        draw_densities(axes[0]); draw_payoff(axes[1], with_repl=True)
    st.pyplot(fig)

    # narration: top decile of q, your probability of it, average ticket payout there
    x90 = float(np.interp(0.9, q.cdf, grid))
    mask = grid >= x90
    q_tail = float(np.trapezoid(q.pdf[mask], grid[mask]))
    p_tail = float(np.trapezoid(p_view[mask], grid[mask]))
    h_grid = ticket.payoff(grid)
    avg_pay = float(np.trapezoid(p_view[mask] * h_grid[mask], grid[mask])) / max(p_tail, 1e-12)
    p_loss = float(summary["P(lose money) under your view"])
    narrate(f"You price a finish above \\${x90:,.0f} — the market's top decile ({q_tail:.0%} under q) — "
            f"at {p_tail:.0%}, i.e. {p_tail / max(q_tail, 1e-12):.1f}× the market; the ticket pays "
            f"\\${avg_pay:,.0f} on average up there ({avg_pay / wealth:.2f}× budget), and the price of "
            f"that tilt is a {p_loss:.0%} chance of losing money under your own view.")

    m = st.columns(4)
    m[0].metric("KL(p‖q)", f"{kl:.4f} nats", help="your edge over the option's life")
    m[1].metric("annualized log growth (replicated)",
                f"{summary['annualized (replicated)']:.1%}",
                help="includes the risk-free carry: at f=0 this is just r, while the KL edge is 0")
    m[2].metric("P(lose) under your view", f"{p_loss:.0%}",
                help="log-optimal payoffs lose often, win big — by design")
    m[3].metric("ticket cost", f"${ticket.cost:,.0f}")

    if stage == 3:
        with st.expander("Order ticket (educational — mid fills, no fees; not investment advice)"):
            st.caption(f"{ticket.stock:+,.1f} shares, ${ticket.bonds:,.0f} bond face "
                       f"(negative = borrow). Left wing in OTM puts via parity.")
            legs = ticket.legs.assign(side=np.where(ticket.legs["qty"] >= 0, "BUY", "SELL"))
            st.dataframe(legs[["type", "strike", "side", "qty", "mid", "cost"]].round(2),
                         hide_index=True)

    # ---------------------------------------------------------- outcome simulator
    wT = simulate(ticker, dte, fraction, wealth)
    med = float(np.median(wT))
    ploss_sim = float(np.mean(wT < ticket.cost))
    lo = max(float(wT.min()) * 0.95, 1.0)
    hi = max(float(wT.max()) * 1.05, lo * 1.01)
    bins = np.geomspace(lo, hi, 60)
    fig2, (ax2,) = fig_axes(1, height=2.9)
    ax2.hist(wT[wT < ticket.cost], bins=bins, color=MAGENTA, alpha=0.85,
             label=f"below cost ${ticket.cost:,.0f} — P(loss) {ploss_sim:.0%}")
    ax2.hist(wT[wT >= ticket.cost], bins=bins, color=GREEN, alpha=0.85, label="above cost")
    ax2.set_xscale("log")
    ax2.minorticks_off()                 # minor tick labels collide at the left edge
    ax2.axvline(med, color=INK, lw=1.5)
    ax2.annotate(f"median ${med:,.0f}", xy=(med, ax2.get_ylim()[1] * 0.80),
                 xytext=(0.74, 0.55), textcoords="axes fraction", fontsize=8, color=INK,
                 bbox=dict(boxstyle="round,pad=0.2", fc="#fcfcfb", ec="none", alpha=0.9),
                 arrowprops=dict(arrowstyle="->", color=MUT))
    ax2.set_xlabel("terminal wealth of the ticket ($, log scale)")
    ax2.set_title("Outcome simulator — 20,000 expiries drawn from your $p_f$")
    ax2.legend(fontsize=9, loc="upper left")
    st.pyplot(fig2)

    # ------------------------------------------------------------- experiments
    # Reveals are computed lazily: a toggle's session_state key is already True on
    # the rerun it triggers, so the extra (cached) pipeline runs happen only once
    # the learner actually asks for the answer.
    if st.session_state.get("m4_exp_f"):
        rows = []
        for f_ in (1.0, 0.25):
            _, _, _, _, tk_, _, _ = solve(ticker, dte, f_, wealth)
            w_ = simulate(ticker, dte, f_, wealth)
            rows.append((float(np.mean(w_ < tk_.cost)), float(np.median(w_)),
                         float(np.percentile(w_, 5)), float(np.percentile(w_, 95))))
        (pl1, md1, lo1, hi1), (pl2, md2, lo2, hi2) = rows
        reveal_f = (f"Computed live — f=1.00: P(loss) {pl1:.0%}, median ${md1:,.0f}, 5th–95th pct "
                    f"${lo1:,.0f}–${hi1:,.0f}.  f=0.25: P(loss) {pl2:.0%}, median ${md2:,.0f}, "
                    f"${lo2:,.0f}–${hi2:,.0f}. The fraction mostly rescales *magnitudes*, not "
                    "frequency: since $(p_f/q)=(p/q)^f$, quarter-Kelly places the same bets at a "
                    "quarter of the log-strength, compressing both tails toward the bond. Full Kelly "
                    "maximizes expected log growth but only if $p$ is exactly right.")
    else:
        reveal_f = "…"
    experiment("m4_exp_f", "full vs quarter Kelly",
               "Set **f = 1.0** and note the simulator's P(loss), median, and the worst bar. "
               "Now set **f = 0.25**. Which changed more — how *often* you lose, or how *much*?",
               reveal_f)

    if st.session_state.get("m4_exp_dte"):
        _, _, _, _, _, _, kl14 = solve(ticker, 14, fraction, wealth)
        ch14, _, _ = load(ticker, 14)
        _, _, _, _, _, _, kl180 = solve(ticker, 180, fraction, wealth)
        ch180, _, _ = load(ticker, 180)
        t14, t180 = ch14.T, ch180.T
        reveal_d = (f"Computed live at f={fraction:.2f}: KL over the trade's life is {kl14:.3f} nats "
                    f"at 14 DTE and {kl180:.3f} at 180 — {'more' if kl180 > kl14 else 'less'} total "
                    f"edge at the long horizon — but per year that is {kl14 / t14:.2f} vs "
                    f"{kl180 / t180:.2f} nats. Drift needs time to beat noise, so lifetime edge "
                    "builds with DTE; yet every extra day asks the block bootstrap to extrapolate "
                    "further from the same 8 years of history. Model risk compounds with horizon "
                    "at least as fast as edge — another argument for f < 1.")
    else:
        reveal_d = "…"
    experiment("m4_exp_dte", "does edge compound with horizon?",
               "Push **DTE from 14 to 180** and watch the KL metric and the payoff's shape "
               "(the tilt gets bigger and smoother). Is the *annualized* edge growing too?",
               reveal_d)

    # ------------------------------------------------------------------- theory
    tab_i, tab_f, tab_d = st.tabs(["Intuition", "The formula", "Deeper"])
    with tab_i:
        st.markdown(
            "A bookie posts odds on every terminal price; those odds *are* $q$. Kelly's rule for "
            "growing a bankroll: split your wealth across outcomes in proportion to **your** "
            "probabilities $p$, regardless of the odds. Wealth delivered in state $S$ costs "
            "$Z\\,q(S)$ per unit, so allocating budget $W\\cdot p(S)$ to each state buys "
            "$g=(W/Z)\\,p/q$ — automatically big where the market undercharges your view. "
            "If $p=q$, you have just bought the bond back.")
    with tab_f:
        st.latex(r"\max_g\;\mathbf{E}_p[\log g]\qquad\text{s.t.}\qquad Z\,\mathbf{E}_q[g]=W")
        st.latex(r"\mathcal{L}=\int p\log g\,dS-\lambda\Big(Z\!\int q\,g\,dS-W\Big)"
                 r"\;\Rightarrow\;\frac{p}{g}=\lambda Z q\;\Rightarrow\;g=\frac{p}{\lambda Z q}")
        st.latex(r"\text{budget}\Rightarrow\lambda=\tfrac{1}{W}:\qquad g^*=\frac{W}{Z}\,\frac{p}{q},"
                 r"\qquad\mathbf{E}_p\!\left[\log\tfrac{g^*}{W}\right]=\mathrm{KL}(p\,\|\,q)-\log Z")
        st.markdown(
            "$p$: your density (block-bootstrapped history, shrunk by $f$) · $q$: the market-implied "
            "density of module 2 · $Z$: the zero-coupon bond price $e^{-rT}$ · $W$: your budget · "
            "$\\lambda$: the budget's Lagrange multiplier · KL: your edge, in nats, over the "
            "option's life.")
    with tab_d:
        st.markdown(
            "Fractional Kelly is geometric-view blending: $p_f\\propto p^f q^{1-f}$, so "
            "$(p_f/q)=(p/q)^f$ — the payoff is the full-Kelly payoff tilted to the $f$-th power, "
            "the classic 'bet a fraction $f$ of the Kelly bet' done in density space.\n\n"
            "Two honest caveats. **P vs Q**: part of any measured gap between $p$ and $q$ is risk "
            "premium, not edge — $q$ is reweighted by marginal utility, so crash states look "
            "'overpriced' to a bootstrap by construction; a rational agent leaves some KL on the "
            "table. **Implementation**: the ticket assumes mid fills, European exercise (single-name "
            "US options are American), listed strikes truncate the tails (hence the ratio cap of 8), "
            "and the bootstrap assumes the future resamples the past 8 years. Full Kelly on a "
            "mis-specified $p$ is how accounts die; $f<1$ is the apology.")


[module_payoff_algebra, module_smile_density,
 module_spanning, module_inverse][MODULES.index(module)]()
