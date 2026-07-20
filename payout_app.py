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
           "4 · The Inverse Problem", "5 · Greeks"]
try:
    _default = int(st.query_params.get("m", 0))
except (TypeError, ValueError):
    _default = 0

with st.sidebar:
    st.title("📐 Payout Lab")
    module = st.radio("Module", MODULES, index=min(_default, len(MODULES) - 1))
    st.caption("**The big idea:** option prices encode the market's probability "
               "distribution for a stock's price at a future date — and you can read "
               "that distribution straight out of them. Options are the measurement "
               "instrument; the distribution is the prize.")
    st.caption("Every finance term is introduced the first time you need it. "
               "New to options? Start with Module 1 and build up.")


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

    st.header("Module 1 · The instrument: what options pay")
    st.caption("This app's mission: option prices encode the market's probability "
               "distribution for a stock's price at a future date. This page builds the "
               "measurement instrument itself — the option — and shows how option payoffs "
               "snap together like building blocks.")
    learn("read a payoff diagram: every bend sits at a strike, every slope change is a quantity",
          "separate what a position pays at expiry from what you actually earn — its P&L",
          "price a \\$1 bet that the stock finishes above 100, and find that its price is a probability")

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

    st.markdown(
        "### The idea\n\n"
        "An **option** is a contract about a stock's future price. A **call** gives you "
        "the right — not the obligation — to *buy* the stock at a fixed price $K$ (the "
        "**strike**) on a fixed date $T$ (the **expiry**). A **put** is the mirror image: "
        "the right to *sell* at $K$. What you pay for the contract today is the **premium**.\n\n"
        "Say a stock trades at \\$100 and you pay a \\$4 premium for a call with strike "
        "$K = 105$ expiring in three months. If the stock finishes at \\$120, you buy at "
        "105, sell at 120, and collect \\$15. If it finishes at \\$90, you walk away — "
        "you only ever lose the \\$4.\n\n"
        "Write $S_T$ for the stock price on the expiry date. At expiry:")
    st.latex(r"\text{a call pays } (S_T - K)^+ \qquad\qquad \text{a put pays } (K - S_T)^+")
    st.markdown(
        "- $S_T$ — the stock price at expiry. Lowercase $t$ means *today*; capital $T$ "
        "means *expiry*.\n"
        "- $K$ — the **strike**: the buy-or-sell price locked in by the contract.\n"
        "- $(x)^+$ — shorthand for $\\max(x, 0)$, the positive part: an option's payoff "
        "is never negative.")
    st.markdown(
        "### How to read this page\n\n"
        f"- **The setup is fixed:** the stock is at $S_0 = 100$ today, the interest rate "
        f"is {r:.0%}, and expiry is in three months ($T = {T}$ years).\n"
        f"- **Money today vs money at expiry.** \\$1 delivered at expiry is worth "
        f"slightly less than \\$1 today; call that value $Z$ — here $Z \\approx {Z:.4f}$. "
        f"A premium paid today is worth premium$/Z$ at expiry.\n"
        f"- **Payoff vs P&L.** The payoff is what you *receive* at expiry. Your **P&L** "
        f"(profit and loss) subtracts what you *paid*, grown for interest: "
        f"P&L $=$ payoff $-$ premium$/Z$. In the \\$4 example: "
        f"15 − 4/{Z:.4f} ≈ \\$10.96.\n"
        "- **Shorthand:** C@95 is a call with strike 95; P@90 is a put with strike 90. "
        "**Long** means you bought it. **Short** means you *sold* it and owe its payoff "
        "— its payoff line flips upside down.\n"
        "- **Legs and recipes:** one option in a package is a **leg**; the dropdown's "
        "names (spread, straddle, butterfly…) are named recipes of legs. Payoffs add, "
        "leg by leg.\n"
        "- **Every chart** plots position value against $S_T$. Each bend (a **kink**) "
        "sits exactly at some leg's strike.\n"
        "- **Implied volatility** $\\sigma$ is the pricing model's one dial for how big "
        "the stock's moves are expected to be — $\\sigma = 0.30$ means moves on the order "
        "of 30% per year. Bigger expected moves make options worth more; the slider below "
        "changes exactly this.\n"
        "- **\"Risk-neutral\":** every probability and price quoted on this page comes "
        "from the pricing model's own assumed distribution for $S_T$ — a lognormal — "
        "called the **risk-neutral** distribution. Read it as *the market's betting "
        "odds*, not a forecast; Module 2 unpacks where it comes from.\n"
        "- **A \"digital\"** is a bet paying \\$1 if $S_T > K$. The last recipe in the "
        "dropdown builds one, and the final experiment prices it.")

    name = st.selectbox("Structure", list(presets), key="m1_structure")
    st.caption("Each structure is a named recipe of calls and puts. Pick one, then "
               "assemble it piece by piece with **Build it up** below — the chart shows "
               "what each recipe pays.")

    quiz("m1_quiz",
         "A **straddle** is one call plus one put at the same strike — it profits from a "
         "big move in either direction. If implied volatility (the expected size of "
         "moves) rises, the price of the straddle… "
         "(pick the straddle above and sweep the σ slider to check)",
         ["rises", "falls", "stays the same"], 0,
         "Rises — a straddle wins when the stock moves a lot, in either direction. "
         "Higher implied volatility means the model expects bigger moves, so both the "
         "call and the put become more valuable together. Check it: pick the straddle, "
         "sweep the σ slider below, and watch the premium column in the table climb.")

    c1, c2 = st.columns(2)
    vol = c1.slider("Implied volatility σ — expected size of yearly moves (0.30 ≈ 30%)",
                    0.10, 0.90, 0.30, 0.05, key="m1_vol",
                    help="The pricing model's dial for how big stock moves are expected "
                         "to be. Bigger expected moves make options cost more.")
    show_pnl = c2.toggle("Show P&L — payoff minus what you paid up front",
                         value=True, key="m1_pnl",
                         help="Payoff is what you receive at expiry; P&L subtracts the "
                              "premium you paid today, grown at the interest rate.")

    def leg_desc(t_, K_, q_):
        return f"{'long' if q_ > 0 else 'short'} {abs(q_):g}× {t_}@{K_:g}"

    legs_all = presets[name]
    descs = [leg_desc(*lg) for lg in legs_all]
    stages = [descs[0]] + [f"+ {d}" for d in descs[1:]]
    st.caption("**Reading the shorthand:** “long 1× C@95” = *buy one call, strike 95*; "
               "“short 1× C@105” = *sell one call, strike 105 — you owe its payoff*. "
               "Each option is a **leg**; the black line is the sum of the legs shown "
               "so far.")
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
    if nshow > 1:                       # a lone leg IS the sum — don't hide it under itself
        for j, lg in enumerate(vis):
            ax.plot(S, payout_of(S, [lg]), color=leg_cols[j % 4], ls="--", lw=1.1,
                    alpha=0.9, label=leg_desc(*lg))
    sum_label = ("sum: payoff at expiry $T$" if nshow > 1
                 else f"{leg_desc(*vis[0])} — payoff at expiry $T$")
    ax.plot(S, total, color=INK, lw=2.4, label=sum_label)
    if show_pnl:
        paid_word = "collected" if premium < 0 else "paid"
        ax.plot(S, pnl, color=BLUE, ls="--",
                label=f"P&L (you {paid_word} \\${abs(premium):,.2f} today)")
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
    ax.set_xlabel("$S_T$ — stock price at expiry")
    ax.set_ylabel("position value at expiry (\\$)")
    ax.set_title(f"{name} — {nshow}/{len(legs_all)} legs shown")
    ax.legend(fontsize=8)
    st.pyplot(fig)

    def money(x):
        return ("−\\$" if x < 0 else "\\$") + f"{abs(x):,.2f}"

    bev_txt = ((" (P&L crosses zero at S_T ≈ " + ", ".join(f"{x:.2f}" for x in roots)
                + f" — the breakeven{'s' if len(roots) > 1 else ''})") if roots else "")
    # net-short calls above the top strike => the visible subset loses without
    # bound as S_T rises: never report a finite kink minimum in that state
    downside = (f"losses grow WITHOUT LIMIT above {max(K_ for _, K_, _ in vis):g}"
                if slope_up < -1e-9 else f"worst case {money(worst)}")
    upside = ("gains grow without cap on the upside" if slope_up > 1e-9
              else f"best case {money(float(pnl_kinks.max()))}")
    narrate(f"Showing {nshow} of {len(legs_all)} legs of the {name}: this position "
            f"{'collects' if premium < 0 else 'costs'} {money(abs(premium))} up front and, "
            f"under the model's assumed lognormal for S_T at {vol:.0%} vol (its "
            f"risk-neutral odds), ends with positive P&L {prob:.0%} of the "
            f"time{bev_txt} — {downside}, {upside}.")

    st.dataframe(pd.DataFrame(vis, columns=["type", "strike K", "qty"])
                   .assign(premium=np.round(prems, 3)),
                 hide_index=True, width=380)
    st.caption("type: C = call, P = put · qty > 0 = long (you bought it), qty < 0 = "
               "short (you sold it) · premium = what each leg costs today; a negative "
               "premium means that leg *pays you*.")

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
        "Select the **bull call spread**, reveal both legs, keep P&L on, and sweep the σ "
        "slider. Watch where the blue P&L line crosses zero — the **breakeven**. Can you "
        "park that crossing exactly at $S_T = 100$? And why does it move at all — the "
        "payoff lines never change?",
        f"Only the premium moves. At {vol:.0%} vol the 95/105 spread costs {money(prem_bs)} "
        f"today; grown for interest that is {prem_bs:.2f}/{Z:.4f} at expiry, so P&L crosses "
        f"zero at 95 + {prem_bs:.2f}/{Z:.4f} = **{be:.2f}**. Raising vol makes this spread "
        f"*cheaper*: the 105 strike sits closer, in percentage terms, to where the model "
        f"expects the stock to end up — the **forward** $F = S_0$ grown at interest "
        f"$\\approx {F:.2f}$ — so the 105 call's price responds more strongly to vol than "
        f"the 95 call's, and you are *short* the 105. A cheaper spread pulls the breakeven "
        f"down; it lands exactly on 100 at vol ≈ **{vstar:.1%}**. (That responds-to-vol "
        f"sensitivity has a name — **vega** — waiting in Module 5.)")

    dprem = 10.0 * float(pl.bs_call_price(S0, 100.0, T, r, vol)
                         - pl.bs_call_price(S0, 100.1, T, r, vol))
    d2 = (np.log(S0 / 100.0) + (r - vol**2 / 2) * T) / (vol * np.sqrt(T))
    nd2 = float(norm.cdf(d2))
    experiment(
        "m1_exp_b", "a premium that is a probability",
        "Select **digital ≈ tight call spread**, reveal both legs, and read the total "
        "premium off the table. The structure pays \\$1 whenever $S_T > 100$ (strictly, "
        "the payoff climbs from \\$0 to \\$1 over the tiny window between $S_T = 100$ and "
        "$100.10$ — the “ramp”). Divide the premium by the \\$1 payout — what number is "
        "the market quoting you?",
        f"At {vol:.0%} vol the tight spread costs {money(dprem)} per \\$1 of payout. "
        f"Un-grow that for interest — divide by $Z$ — and you get {dprem / Z:.4f}: "
        f"essentially **the model's probability that $S_T$ finishes above 100**. The price "
        f"of a \\$1 bet *is* a probability (times $Z$). Exactly: the model values the "
        f"clean \\$1 bet at $Z\\,N(d_2) = {Z:.4f} \\times {nd2:.4f} = "
        f"\\mathbf{{{Z * nd2:.4f}}}$, where $N$ is the standard normal CDF and "
        f"$N(d_2) = {nd2:.4f}$ is precisely the risk-neutral $P(S_T > 100) = "
        f"{nd2:.1%}$. The spread agrees to within \\${abs(dprem - Z * nd2):.4f} — it costs a "
        f"touch *less*, because over the 10-cent ramp it pays a little less than the full "
        f"\\$1. The general identity behind this lives in the “How the pros use it” tab.")

    t1, t2, t3 = st.tabs(["Why this works", "The math, gently", "How the pros use it"])
    with t1:
        st.markdown(
            "**Payoffs are Lego.** Four bricks — the **bond** (a contract worth exactly "
            "\\$1 at expiry), the stock $S_T$, calls $(S_T-K)^+$ and puts $(K-S_T)^+$ "
            "(recall $(x)^+ = \\max(x,0)$) — snap together by addition. Every kink in a "
            "diagram sits at a strike, and the slope change across it equals the signed "
            "quantity struck there. To reverse-engineer any hockey-stick picture, walk left "
            "to right: each time the slope jumps by $q$, write down \"$q$ calls at that "
            "strike.\"")
    with t2:
        st.markdown("Any piecewise-straight payoff you can draw is a sum of the four "
                    "bricks. In symbols:")
        st.latex(r"g(S_T)\;=\;\alpha\cdot 1\;+\;\delta\cdot S_T\;+\;"
                 r"\sum_i q_i\,(S_T-K_i)^+\;+\;\sum_j p_j\,(K_j-S_T)^+")
        st.markdown(
            "- $g(S_T)$ — the payoff you are building, as a function of the final stock "
            "price\n"
            "- $\\alpha$: how many \\$1-at-expiry bonds you hold (each costs $Z$ today) — "
            "sets the height of the flat part\n"
            "- $\\delta$: how many shares you hold (each costs $S_0$ today) — sets the "
            "baseline slope\n"
            "- $q_i,\\ p_j$: call and put quantities at strikes $K_i, K_j$ — each adds one "
            "kink\n"
            "- puts are redundant given calls, by **put–call parity** — hold a call, sell "
            "one share, keep $K$ bonds, and you have rebuilt the put:")
        st.latex(r"(K-S_T)^+=(S_T-K)^+-S_T+K"
                 r"\quad\Longrightarrow\quad P=C-S_0+K\,Z")
        st.caption("The payoff identity holds for every possible $S_T$, so it must hold "
                   "in price too — that parity is exactly how every put leg above is "
                   "priced off the call pricer.")
    with t3:
        st.markdown(
            "*Fair warning: the vocabulary deepens here — but you now have every tool it "
            "uses.*\n\n"
            "**The payoff-vs-P&L trap.** A structure that \"never pays less than zero\" is "
            "not free money: P&L subtracts the interest-grown premium, and at fair prices "
            "the risk-neutral expected P&L of *any* package is exactly zero. The line that "
            "matters for risk is the blue one, not the black one.\n\n"
            "**The digital as a limit.** Squeeze the call spread's width to zero while "
            "scaling up the quantity, and its price converges to a derivative — the "
            "identity Experiment (b) approximated:")
        st.latex(r"\lambda\left(C_K - C_{K+1/\lambda}\right)"
                 r"\xrightarrow{\;\lambda\to\infty\;}"
                 r"-\frac{\partial C}{\partial K}\;=\;Z\,\mathbb{Q}(S_T>K)")
        st.markdown(
            "Read it as: the tighter the spread (width $1/\\lambda$), the closer its price "
            "gets to $Z$ times the probability of finishing above $K$ — where "
            "$\\mathbb{Q}$ denotes probabilities under the risk-neutral distribution. "
            "Module 2 differentiates once more and recovers the entire density.\n\n"
            "**How dealers quote digitals.** Nobody trades the limit. The seller offsets "
            "the risk (a **hedge**) with the slightly-generous call spread on "
            "$[K-\\varepsilon, K]$, which pays at least the \\$1 everywhere; the buyer is "
            "shown the slightly-stingy one on $[K, K+\\varepsilon]$, which never pays "
            "more; the client crosses the gap between the two prices. That is why quoted "
            "digital prices widen near the strike as expiry approaches — the spread hedge "
            "concentrates all its risk right at the strike (that risk has a name, "
            "**gamma** — Module 5).")


# ======================================================================= 2
def module_smile_density():
    from scipy.stats import norm

    st.header("The smile is a probability distribution in disguise")
    st.caption("The **smile** is the curve of option prices across strikes, drawn in "
               "volatility units. The **density** is the probability curve those same "
               "prices imply for where the stock ends up. This page shows the two are "
               "one object — every term in that sentence is defined below.")
    learn("see how the pattern of option prices across strikes reveals the market's "
          "fear of a crash",
          "recover the market's implied probability density from those prices",
          "check a price curve for internal contradictions — spots where someone is "
          "giving away free money")

    st.markdown("### The idea")
    st.markdown(
        "This page is the heart of the whole app: turning quoted option prices into "
        "a full probability distribution.\n\n"
        "Start with the instrument. A **call option** is a contract: the right — not "
        "the obligation — to buy a stock at a fixed price $K$ (the **strike**) on a "
        "fixed date $T$ (the **expiry**). What you pay today for that right is the "
        "**premium**. Say a stock trades at \\$100 and you pay a \\$4 premium for a "
        "call struck at $K=105$. If the stock finishes at \\$120, you buy at 105, "
        "sell at 120, and pocket \\$15 — \\$11 after the premium. Finish below 105 "
        "and the contract expires worthless. (New here? Module 1 builds all of this "
        "from scratch.)\n\n"
        "Write $S_T$ for the stock price at expiry. The call's payoff is:")
    st.latex(r"\text{call payoff at expiry}\;=\;\max(S_T - K,\; 0)")
    st.markdown(
        "- $S_T$ — the stock's price on the expiry date: a random variable, and the "
        "object whose distribution we are after\n"
        "- $K$ — the **strike**: the buying price the contract locks in\n"
        "- $T$ — the **expiry** date; here $T$ = 3 months\n\n"
        "The market quotes not one call but dozens — one per strike — so there is a "
        "whole curve of prices across $K$. Its natural center is the **forward** "
        "$F$: today's price grown at the interest rate, about \\$101 here. We "
        "measure a strike's distance from center as $k=\\ln(K/F)$: zero means "
        "strike = forward, negative means below it.\n\n"
        "Traders restate each dollar price as an **implied volatility** — how bumpy "
        "a ride would justify that price in the standard pricing formula "
        "(**Black–Scholes**). If that textbook formula were exactly right, every "
        "strike would show one number: a flat line. Real markets refuse. Strikes "
        "away from center carry higher implied vols, and the resulting U-shaped "
        "curve is the **smile**. A lopsided smile — one side higher — is called "
        "**skew**. The center point, strike = forward, is **at-the-money** (**ATM**).")

    st.markdown("### From prices to probabilities")
    st.markdown(
        "Here is the bridge — and where your probability fluency pays off. A call's "
        "price is what the market charges for *all outcomes above $K$*. So the rate "
        "at which call prices fall as $K$ rises encodes probability. Sharpen that: "
        "buy one call at $K-h$, sell two at $K$, buy one at $K+h$. That package is "
        "called a **butterfly**. It pays something only if the stock lands near $K$, "
        "and never pays less than zero. Its price is therefore the market's price "
        "for *landing near $K$* — and dividing by $h^2$ and the **discount factor** "
        "$Z$ (a dollar at expiry is worth $Z$ today) turns it into the probability "
        "density $q(K)$ as $h\\to 0$. That recipe is the **Breeden–Litzenberger** "
        "second difference, and it is why the smile and the density are the same "
        "object in two coordinate systems.\n\n"
        "One caution. The density read out of prices is the **risk-neutral** "
        "density: the market's *betting odds*, not its honest forecast. Prices bake "
        "in risk premiums — insurance against bad states costs extra, which tilts "
        "the odds toward those states. When this page says \"the market's chance\", "
        "it means these odds.\n\n"
        "And one free lunch. A butterfly never pays less than zero, so its price "
        "can never legitimately be negative. If a smile implies $q<0$ anywhere, the "
        "prices contradict each other and anyone can collect free money "
        "(**arbitrage**). Trading desks scan for $q \\ge 0$ before publishing "
        "prices; you will run the same scan below.")

    st.markdown("### Draw a smile, watch the density follow")
    st.markdown(
        "The five sliders below draw a smile using the industry's standard recipe, "
        "**SVI** (\"Stochastic Volatility Inspired\" — a name, not a description: "
        "it is a five-parameter curve). The app converts your smile into its "
        "density, live.\n\n"
        "**What each knob does:** $a$ sets the overall height. $b$ sets how steep "
        "the two sides — the **wings** — are. $\\rho$ (rho) tilts the smile: drag "
        "it negative and the left side rises, so insurance against a fall costs "
        "more; drag it positive and big up-moves cost more. $m$ slides the low "
        "point sideways. $\\sigma$ rounds the bottom — careful, this $\\sigma$ is a "
        "shape knob, **not** the implied volatility on the chart's y-axis.")

    # -- presets seed the sliders; a guard lets manual slider moves survive reruns
    PRESETS = {
        "flat — textbook world: one vol at every strike":
            dict(a=0.020, b=0.01, rho=0.00,  m=0.00, sg=0.25),
        "equity-index skew — crash insurance priced up (the usual stock-market shape)":
            dict(a=0.001, b=0.04, rho=-0.35, m=0.02, sg=0.25),
        "symmetric smile — both big moves priced up (currency-style)":
            dict(a=0.010, b=0.25, rho=0.00,  m=0.00, sg=0.15),
        "steep crash fear — extreme downside pricing":
            dict(a=0.012, b=0.30, rho=-0.80, m=0.00, sg=0.25),
    }
    preset = st.radio("Start from a market shape:", list(PRESETS),
                      key="m2_preset", horizontal=True, index=1)
    if st.session_state.get("m2_last_preset") != preset:
        st.session_state["m2_last_preset"] = preset
        for pk, v in PRESETS[preset].items():
            st.session_state["m2_" + pk] = v

    st.markdown(
        "**Ground the prediction first.** The density's *tails* are its far ends — "
        "the probabilities of extreme moves. Stock-index markets trade with "
        "$\\rho < 0$: crash insurance costs extra, the smile's left side rides "
        "high, and the density's far left tail fattens.")
    quiz("m2_quiz_rho",
         "Now flip it: drag ρ *positive*, so the **right** side of the smile rises "
         "instead. Which **far** tail of the density gets fatter?",
         ["the left tail", "the right tail", "both equally"], 1,
         "Positive ρ makes the right side of the smile steeper — the market charges "
         "more for bets on big up-moves — and that is the same thing as putting more "
         "probability in the far **right** tail. Real stock-index markets do the "
         "opposite (ρ < 0): they price up crash bets, fattening the far left tail. "
         "(Near the center the effect can flip sign — see *The math, gently*.)")

    c = st.columns(5)
    a_ = c[0].slider("a — overall level", 0.001, 0.10, step=0.001, format="%.3f",
                     key="m2_a",
                     help="Raises or lowers the whole smile (in total-variance units).")
    b_ = c[1].slider("b — wing steepness", 0.01, 1.50, step=0.01, key="m2_b",
                     help="How fast implied vol climbs as strikes move away from "
                          "center. Steeper wings = fatter tails.")
    rho = c[2].slider("ρ — tilt: negative = crash fear", -0.99, 0.99, step=0.01,
                      key="m2_rho",
                      help="Tilts the smile. Negative raises the left side (fall "
                           "insurance costs more); positive raises the right "
                           "(melt-up bets cost more).")
    m_ = c[3].slider("m — sideways shift", -0.5, 0.5, step=0.01, key="m2_m",
                     help="Slides the smile's low point left or right along the "
                          "strike axis.")
    sg = c[4].slider("σ — rounds the bottom (not vol!)", 0.01, 1.0, step=0.01,
                     key="m2_sg",
                     help="Smooths the kink at the smile's bottom. A shape knob — "
                          "not the implied volatility on the chart's y-axis.")

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
    ax.annotate(f"ATM (strike = forward) {atm:.0%}", xy=(0, atm), xytext=(12, 12),
                textcoords="offset points", fontsize=8, color=MUT,
                bbox=dict(boxstyle="round,pad=0.2", fc="#fcfcfb", ec="none", alpha=0.9),
                arrowprops=dict(arrowstyle="-", color=MUT, lw=0.8))
    ax.set_xlabel("strike's distance from center  $k=\\ln(K/F)$")
    ax.set_ylabel("implied vol")
    ax.set_title("The smile — option prices in vol units")
    ax = axes[1]
    ax.plot(Kq, q, color=INK)
    lmask = Kq <= Kthr
    ax.fill_between(Kq, qp, 0, where=lmask, color=MAGENTA, alpha=0.35, lw=0)
    ax.annotate(f"← same information\n$P(S_T<{Kthr:.0f})$ = {tail:.1%}",
                xy=(0.80 * F, float(np.interp(0.80 * F, Kq, qp))),
                xytext=(0.02, 0.85), textcoords="axes fraction", fontsize=8, color=MAGENTA,
                bbox=dict(boxstyle="round,pad=0.2", fc="#fcfcfb", ec="none", alpha=0.85),
                arrowprops=dict(arrowstyle="->", color=MAGENTA, lw=1.2))
    im = int(np.argmax(q))
    ax.annotate(f"most likely finish ≈ {Kq[im]:.0f}", xy=(Kq[im], q[im]), xytext=(10, -2),
                textcoords="offset points", fontsize=8, color=MUT)
    neg = q < 0
    if neg.any():
        ax.fill_between(Kq, q, 0, where=neg, color=RED, alpha=0.6,
                        label="q < 0 — negative probability: free money")
        ax.legend(fontsize=9)
    ax.axhline(0, color=MUT, lw=1)
    ax.set_xlabel("$K$ — strike"); ax.set_ylabel("probability density")
    ax.set_title("The implied density — the market's odds for $S_T$")
    st.pyplot(fig)
    st.caption("**Left:** each point of the curve is one option — x is its strike's "
               "distance from the forward (0 = at the forward, negative = below), "
               "y is the implied vol that reproduces its market price. **Right:** "
               "the probability density for $S_T$ that those same prices imply. The "
               "shaded smile wing and the shaded density tail carry the same "
               "information — that is the magenta arrow's claim.")

    narrate(f"Calls struck at {F * np.exp(-0.15):.0f} — about 14% below the forward "
            f"of {F:.0f} — trade at {iv15:.1%} implied vol, versus {atm:.1%} at the "
            f"money ({(iv15 - atm) * 100:+.1f} vol points). Priced that way, the "
            f"market's odds of finishing below {Kthr:.0f} are {tail:.1%} (a flat "
            f"smile at the same ATM vol would say {flat_tail(0.85):.1%}), and the "
            f"odds of a severe drop below {Kdeep:.0f} are {deep:.1%} "
            f"(flat: {flat_tail(0.70):.1%}).")

    mass = float(np.trapezoid(qp, Kq))
    negmass = max(0.0, float(-np.trapezoid(np.minimum(q, 0), Kq)))
    lee = b_ * (1 + abs(rho))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("total probability (should be ≈ 1)", f"{mass:.3f}",
              help="∫q⁺dK over the plotted strikes. Slightly under 1 is fine — some "
                   "probability lives beyond the plotted strike range. Above 1 means "
                   "arbitrage has created negative density elsewhere on the curve.")
    c2.metric("negative probability", f"{negmass:.4f}",
              delta="arbitrage!" if negmass > 1e-4 else "arb-free", delta_color="inverse",
              help="A legitimate density is never below zero. If your sliders push "
                   "q negative anywhere, the prices contradict each other — free "
                   "money (arbitrage). Try the first experiment below.")
    c3.metric("wing-steepness limit b(1+|ρ|)", f"{lee:.2f} / 2.00",
              delta="violated" if lee > 2 else "ok", delta_color="inverse",
              help="A theorem (Lee, 2004) caps how steep a smile's wings can grow: "
                   "past 2, the prices implied for extreme-strike bets blow up. "
                   "Details in 'The math, gently'.")
    c4.metric("ATM vol", f"{atm:.1%}",
              help="The implied volatility at the center of the smile, where "
                   "strike = forward.")

    # exact price of the (K−1, K, K+1) butterfly at the worst strike: grid step is
    # 0.2, so ±5 indices = ±$1; B = C(K−1) − 2C(K) + C(K+1) ≈ Z·q(K)·1²
    i0 = int(np.argmin(q[25:-25])) + 25
    j = i0 + 1
    bfly = float(C[j - 5] - 2 * C[j] + C[j + 5])
    experiment("m2_exp_arb", "manufacture free money",
               "Push **b → 1.2** (very steep wings) and **σ → 0.05** (a sharp "
               "bottom) with ρ near −0.8. Watch the density panel and the *negative "
               "probability* metric.",
               f"Scaled butterfly prices **are** the density (see *The math, "
               f"gently*), so q < 0 means a butterfly with a negative price. "
               f"Remember the package: buy one call struck at K−1, sell two at K, "
               f"buy one at K+1 — it pays between \\$0 and \\$1, never less than "
               f"zero. Off your current smile the worst spot is "
               f"q({Kq[i0]:.0f}) = {q[i0]:.4f}: the butterfly there prices at "
               f"**\\${bfly:+.4f}**. "
               + ("The market would *pay you* to own a never-negative payoff: free "
                  "money, unless the quotes are stale. " if q[i0] < 0 else
                  "Still positive, so no arbitrage yet — push further. ")
               + "Trading desks run exactly this q ≥ 0 scan before publishing "
                 "prices.")

    loc = (qp[1:-1] > qp[:-2]) & (qp[1:-1] >= qp[2:]) & (qp[1:-1] > 0.05 * max(float(qp.max()), 1e-12))
    n_modes = int(loc.sum())
    if n_modes <= 1:
        mode_msg = ("Your current density has **one hump** — and SVI will fight hard "
                    "to keep it that way.")
    elif negmass > 1e-4:
        mode_msg = (f"You count **{n_modes} local peaks**, but negative probability "
                    f"reads {negmass:.4f}: the extra hump was bought with free-money "
                    "butterfly prices next to it — not a market, a mispricing.")
    else:
        mode_msg = (f"You managed to dent the top into {n_modes} local maxima, "
                    "but not two genuinely separated humps.")
    experiment("m2_exp_bimodal", "try to make it bimodal",
               "Before an earnings report or a court ruling, traders expect *two* "
               "outcomes — jump or dump — so the honest density has two humps. Try "
               "any slider combination that gives the density two separated humps.",
               mode_msg + " SVI's five parameters allow exactly one smile minimum, "
               "and an arbitrage-free single-minimum smile produces an (essentially) "
               "unimodal density. Around genuinely two-outcome events — earnings, "
               "drug approvals, referenda — professionals set this five-knob curve "
               "aside for more flexible ones that can draw two humps (named in *How "
               "the pros use it*).")

    tab_i, tab_f, tab_d = st.tabs(["Why this works", "The math, gently",
                                   "How the pros use it"])
    with tab_i:
        st.markdown(
            "A call's price is what the market charges for *all outcomes above K*. "
            "So the family of call prices across strikes is a photograph of the "
            "probability density. The left chart re-plots those same prices in "
            "volatility units — a different ruler laid on the same picture.\n\n"
            "Lift the smile's left wing and you have made crash insurance dearer: "
            "that *is* moving probability into the deep left tail. One subtlety "
            f"worth watching: raising the left wing can shift probability from a "
            f"mild drop (below {Kthr:.0f}) to a severe one (below {Kdeep:.0f}). The "
            "blue box quotes both numbers so you can watch them move separately.\n\n"
            "One object, two coordinate systems; every slider bends both panels at "
            "once.")
    with tab_f:
        st.markdown(
            "Smiles are drawn in **total implied variance** — implied variance "
            "accumulated from now to expiry, $w=\\sigma_{\\mathrm{imp}}^2 T$. SVI "
            "says $w$ is a simple function of the strike's distance from center:")
        st.latex(r"w(k) \;=\; a + b\left(\rho\,(k-m) + \sqrt{(k-m)^2+\sigma^2}\right),"
                 r"\qquad k=\ln(K/F),\quad w=\sigma_{\mathrm{imp}}^2\,T")
        st.markdown(
            "- $k=\\ln(K/F)$ — the strike's distance from the forward, in log "
            "units\n"
            "- $a$ — the overall variance level\n"
            "- $b$ — wing steepness: far from center, $w$ grows linearly with "
            "slopes $b(1-\\rho)$ on the left and $b(1+\\rho)$ on the right\n"
            "- $\\rho$ — the tilt: which wing gets the steeper slope\n"
            "- $m$ — shifts the vertex (the low point) sideways\n"
            "- $\\sigma$ — rounds the vertex; **not** the implied volatility\n\n"
            "Lee's moment formula (2004) caps the wings: $b(1+|\\rho|) \\le 2$, or "
            "the distribution's far-strike moments — and the implied prices of "
            "extreme-strike bets — become infinite. That is the wing-steepness "
            "metric above. One precise caveat behind the quiz: near the center, "
            "$P(S_T<K)$ depends on the smile's local *slope* as well as its level, "
            "so the tilt's effect on near-center probabilities can flip sign — the "
            "far tails keep the clean story.\n\n"
            "Now the price-to-density recipe. Difference the call curve twice:")
        st.latex(r"q(K)\;=\;\frac{1}{Z}\,\frac{\partial^2 C}{\partial K^2}"
                 r"\;=\;\lim_{h\to 0}\;\frac{C(K\!-\!h)-2C(K)+C(K\!+\!h)}{Z\,h^{2}}")
        st.markdown(
            "- $C(K)$ — the call's price at strike $K$, computed off the smile\n"
            "- $h$ — the strike spacing; the numerator is exactly the butterfly's "
            "price\n"
            "- $Z=e^{-rT}$ — the **discount factor**: a dollar paid at expiry is "
            "worth $Z$ today\n\n"
            "That limit is **Breeden–Litzenberger** (1978): butterfly price ÷ "
            "($Z h^2$) → density. The tail numbers in the blue box use the "
            "first-derivative twin of the same identity, "
            "$P(S_T<K) = 1 + e^{rT}\\,\\partial C/\\partial K$ — slope gives the "
            "CDF, curvature gives the density.")
    with tab_d:
        st.markdown(
            "*Fair warning: trader vocabulary runs deeper here.* Before a bank's "
            "trading desk publishes a smile, it runs three free-money checks. "
            "**Butterfly**: q ≥ 0 at every strike — exactly the second difference "
            "above, taken on the smoothed fit, never on raw quotes. **Calendar**: "
            "total variance w(k,T) must be non-decreasing in T at fixed k — "
            "otherwise a longer-dated option would cost less than a shorter one "
            "covering the same risk, and a *calendar spread* (sell the short-dated, "
            "buy the long-dated) collects the difference for free. **Wings**: Lee's "
            "bound caps how fast w may grow in |k| — otherwise the far-strike "
            "moments of the distribution, and with them the implied prices of "
            "extreme-strike bets, become infinite. SVI is the industry's default "
            "slice precisely because these checks are cheap: sufficient "
            "no-arbitrage conditions are known in closed form, and its linear wings "
            "match Lee's bound by construction. Where one slice fails — event risk, "
            "bimodal outcomes — desks bolt on *mixture models* (a weighted blend of "
            "two or three densities) or fit each expiry non-parametrically.")


# ======================================================================= 3
def module_spanning():
    st.header("Spanning: build any payoff from a bond, shares, and call options")
    st.markdown(
        "Any curve of dollars-versus-final-stock-price you can draw can be manufactured "
        "from three traded ingredients — a bond, some shares, and a pile of call options. "
        "Wall Street's fear index, the **VIX**, is exactly such a pile. "
        "*New here? Module 1 builds the option vocabulary from scratch.*")

    st.markdown("### The idea")
    st.markdown(
        "First, the one instrument this page builds with. A **call option** is a contract "
        "giving you the right — never the obligation — to buy a stock at a locked-in price "
        "(the **strike**, $K$) on a fixed future date (the **expiry**, $T$). Say the stock "
        "trades at \\$100 and you pay \\$4 today (the **premium**) for a call struck at "
        "\\$105. If the stock finishes at \\$120, you buy at \\$105, sell at \\$120, and "
        "collect \\$15. If it finishes at \\$90, you walk away and lose only the \\$4.\n\n"
        "A **payoff diagram** plots the dollars a position delivers at expiry against the "
        "final stock price $S_T$. Three primitive shapes matter. A **bond** (a guaranteed "
        "\\$1 at expiry) is a horizontal line: pure *level*. A share is a 45° line: pure "
        "*slope*. A call pays $\\max(S_T-K,\\,0)$ — a hockey stick, flat at zero and then "
        "kinking upward at its strike: pure *bend*, delivered at exactly one spot.\n\n"
        "**A bond sets the level; a stock sets the slope; neither can bend.** So every "
        "bend in a target curve must be bought with options. A famous result by Carr and "
        "Madan makes this exact. Pin the target's height and slope at one **anchor** "
        "strike $\\kappa$; then at every strike buy calls in proportion to how sharply "
        "the target bends there — its second derivative $g''(K)$ (a straight line has "
        "$g''=0$). A bundle of calls across many strikes is a **strip**, and the quantity "
        "held at each strike is its **weight**:")
    st.latex(r"w_i \;\approx\; g''(K_i)\,\Delta K"
             r"\;=\;\text{bend at that strike}\times\text{spacing}")
    st.markdown(
        "- $g(S_T)$ — the **target payoff**: the dollars you want, as a function of the "
        "final stock price $S_T$\n"
        "- $K_i$ — the $i$-th **strike** on the exchange's menu of quoted options "
        "(the **option chain**)\n"
        "- $\\Delta K$ — the spacing between neighboring strikes on that menu\n"
        "- $g''(K_i)$ — how sharply the target bends at $K_i$\n"
        "- $w_i$ — calls held at $K_i$: positive = buy, negative = sell, and fractional "
        "amounts are normal (quantities scale with the payoff's dollar size)\n"
        "- $F$ — the **forward**: today's price grown at the interest rate, ≈ today's "
        "price; $\\kappa$ — the anchor strike where level and slope are pinned\n\n"
        "Module 2 read the market's probability distribution for $S_T$ out of option "
        "prices. This page is the converse: because options **span** every smooth payoff, "
        "any payoff can be priced by adding up visible quotes — same payoff, same price, "
        "no model of the future required. The **VIX** — the market's index of how much "
        "movement it expects over the next 30 days — is one such strip, weighted by "
        "$1/K^2$. It sits above the volatility quoted at strikes near today's price "
        "(**at-the-money**) because of **skew** — the market charging extra for "
        "crash protection (Module 2).")

    learn("split any payoff curve into level (a bond) + slope (shares) + bend (calls)",
          "read how many calls each strike needs from how sharply the target bends there "
          "— $w_i \\approx g''(K_i)\\,\\Delta K$",
          "explain why the VIX is a call strip weighted by $1/K^2$, and why crash-"
          "protection pricing (skew) lifts it above at-the-money volatility")

    quiz("m3_quiz",
         "The target $(S/F-1)^2$ — $S$ the stock price at expiry, $F$ the forward, a "
         "reference near today's price — is a parabola, so it bends by the same amount "
         "everywhere. If each call's job is to supply bend at its own strike, how should "
         "the quantities vary across strikes?",
         ["increasing with K", "constant", "hump-shaped near F"], 1,
         "The quantity at each strike matches how much the target bends there. A parabola "
         "bends equally everywhere, so every strike gets the same amount — the bars below "
         "will be flat. (In symbols: $w_i \\approx g''(K_i)\\,\\Delta K$ with "
         "$g''=2/F^2$ constant.) The dropdown starts on this squared-deviation target — "
         "check the flat bars yourself.")

    S0, r, T = 100.0, 0.04, 0.25
    F, Z = S0 * np.exp(r * T), np.exp(-r * T)
    S = np.linspace(40, 200, 1601)

    targets = {
        "squared deviation  (S/F − 1)²": lambda s: (s / F - 1) ** 2,
        "Gaussian bump (bet on a range)": lambda s: np.exp(-0.5 * ((s - 110) / 12) ** 2),
        "log contract  −ln(S/F) — the payoff behind the VIX": lambda s: -np.log(s / F),
    }
    c1, c2 = st.columns([2, 1])
    tname = c1.selectbox("Target payoff g(S_T) — the dollars-vs-final-price curve to build",
                         list(targets), key="m3_target",
                         help="The curve of dollars you want to receive at expiry, as a "
                              "function of the final stock price S_T. The lab will "
                              "manufacture it from a bond, shares, and calls.")
    dK = c2.select_slider("Strike spacing ΔK — how far apart the strikes are",
                          [2.5, 5.0, 10.0, 20.0], value=5.0, key="m3_dk",
                          help="The gap between neighboring strikes on your menu. "
                               "Finer spacing = more strikes to build with.")
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

    stage = stepper("m3_stage", ["1 · the target g", "2 · + bond (set the level)",
                                 "3 · + stock (match the slope at the anchor κ)",
                                 "4 · + call strip (buy the bend)"])
    n = N
    if stage == 3:
        n = st.slider("strikes added so far, left to right", 0, N, N,
                      key=f"m3_nadd_{int(dK*10)}")

    labels = {1: f"bond only: {bonds:+.3f} bonds", 2: "bond + shares (tangent line)",
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
    at_edge = (not flat_curv) and int(np.argmax(core)) == 0
    if stage == 0:
        txt = ("bends the same amount everywhere —\nevery strike will get the same weight"
               if flat_curv
               else ("bends harder and harder toward low prices —\nthe low-strike end is "
                     "where calls must work" if at_edge
                     else f"bends hardest near S≈{Sc:.0f} —\ncalls near here will do the work"))
        ax.annotate(txt, xy=(Sc, g(Sc)), xytext=(25, 30),
                    textcoords="offset points", fontsize=8, color=MUT, arrowprops=arrow)
    elif stage == 1:
        ax.annotate(f"bond: flat at g(κ)={bonds:+.2f}\nno slope, no bend",
                    xy=(140.0, bonds), xytext=(10, 30),
                    textcoords="offset points", fontsize=8, color=MUT, arrowprops=arrow)
    elif stage == 2 or (stage == 3 and n == 0):
        xs = kappa + 20
        ax.annotate(f"tangent at the anchor κ={kappa:g}:\nslope g'(κ)≈{s0:+.3f}",
                    xy=(xs, float(np.interp(xs, S, lin))), xytext=(15, -35),
                    textcoords="offset points", fontsize=8, color=MUT, arrowprops=arrow)
    elif n < N:
        Kf = Ki[n - 1]
        ax.annotate(f"next strike: {w[n-1]:+.4f} calls @ K={Kf:g}\nto bend the line here",
                    xy=(Kf, float(np.interp(Kf, S, partial))), xytext=(20, -35),
                    textcoords="offset points", fontsize=8, color=MUT, arrowprops=arrow)
    else:
        ax.annotate(f"worst remaining gap {err:.4f}\n(between strikes)",
                    xy=(err_loc, g(err_loc)), xytext=(20, 30),
                    textcoords="offset points", fontsize=8, color=MUT, arrowprops=arrow)
    ax.set_xlabel("stock price at expiry, $S_T$")
    ax.set_ylabel("payoff (dollars at expiry)")
    ax.set_title("Building the payoff piece by piece"); ax.legend(fontsize=9)

    ax = axes[1]
    n_add = n if stage == 3 else 0
    if n_add:
        ax.bar(Ki[:n_add], w[:n_add], width=0.6 * dK,
               color=[GREEN if x >= 0 else MAGENTA for x in w[:n_add]])
    if n_add < N:
        ax.bar(Ki[n_add:], w[n_add:], width=0.6 * dK, alpha=0.25,
               color=[GREEN if x >= 0 else MAGENTA for x in w[n_add:]],
               label="not yet added" if stage == 3 else "the strip stage 4 will add")
    if "log contract" in tname:
        ax.plot(Ki, dK / Ki**2, color=YELLOW, lw=2,
                label="$\\Delta K/K^2$ — the VIX's weights")
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.set_xlabel("strike price $K$")
    ax.set_ylabel("calls held at that strike")
    ax.set_title("Calls per strike:  $w_i \\approx g''(K_i)\\,\\Delta K$ = bend × spacing")
    if n_add < N or "log contract" in tname:
        ax.legend(fontsize=9)
    st.pyplot(fig)

    lo, hi = strikes[0], strikes[-1]
    if stage == 0:
        if flat_curv:
            narrate(f"This parabola bends at the same rate everywhere (g''=2/F²"
                    f"≈{2/F**2:.6f}), so each of the {N} strikes will carry the same "
                    f"weight ≈{2/F**2*dK:.5f} — bonds and shares contribute zero bend.")
        elif at_edge:
            narrate(f"This curve bends harder and harder as the price falls (its bend is "
                    f"1/S², already {core.max():.4f} by S≈{Sc:.0f}). A bond can't tilt "
                    "and a share can't bend — so the bending must come from calls, "
                    "mostly at low strikes.")
        else:
            narrate(f"This target bends hardest near S≈{Sc:.0f} (bend ≈{core.max():.4f} "
                    "there). A bond can't tilt and a share can't bend, so every bit of "
                    "that bending must be bought with calls.")
    elif stage == 1:
        gap = float(np.max(np.abs(gS - partial)[inside]))
        narrate(f"{bonds:+.3f} bonds pin the level at g(κ)={bonds:.3f} — the same "
                f"dollars no matter where the stock ends up. The gap to the target is "
                f"still {gap:.2f} at its worst.")
    elif stage == 2:
        gap = float(np.max(np.abs(gS - partial)[inside]))
        narrate(f"Adding {s0:+.4f} shares tilts the flat line into the tangent at the "
                f"anchor κ={kappa:g} — exact right there, yet still off by up to "
                f"{gap:.2f} wherever the target bends.")
    elif n < N:
        gap = float(np.max(np.abs(gS - partial)[inside]))
        narrate(f"With {n} of {N} strikes at spacing ΔK={dK:g}, the build sits within "
                f"{gap:.4f} of the target across ({lo:g}, {hi:g}). All {N} strikes get "
                f"it to {err:.4f}, and halving the spacing to {dK/2:g} cuts that to "
                f"{err2:.4f} (×{err/err2:.1f} better) — the error shrinks with the "
                "square of the spacing.")
    else:
        narrate(f"With all {N} strikes at spacing ΔK={dK:g}, the build sits within "
                f"{err:.4f} of the target across ({lo:g}, {hi:g}). Halve the spacing to "
                f"{dK/2:g} and the error drops to {err2:.4f} — ×{err/err2:.1f}, "
                "shrinking with the square of the spacing. (Numerics people call this "
                "quadratic convergence.)")

    # metrics appear as their objects enter the construction — no spoilers
    m1, m2, m3 = st.columns(3)
    if stage >= 1:
        m1.metric("bonds held (set the level g(κ))", f"{bonds:+.3f}")
    if stage >= 2:
        m2.metric("shares held (set the slope g'(κ))", f"{s0:+.4f}")
    if stage >= 3:
        m3.metric("worst error inside the strike range", f"{err:.4f}",
                  help="Halve the spacing ΔK and this drops to about a quarter (≈×4) — "
                       "the error of a straight-segment build shrinks like ΔK².")

    # -- experiment (a): where coarse grids fail, computed for the Gaussian bump
    gb = targets["Gaussian bump (bet on a range)"]
    eb_f, loc_f = strip_err(2.5, gb)
    eb_c, loc_c = strip_err(20.0, gb)
    experiment(
        "m3_exp1", "a coarse strike menu fails exactly where the bend lives",
        "Pick the **Gaussian bump** target and stage 4 with all strikes, then step the "
        "spacing ΔK from 2.5 up to 20. Watch *where* the shaded gap grows, not just how "
        "much.",
        f"At ΔK=2.5 the worst error is {eb_f:.4f} (near S≈{loc_f:.0f}); at ΔK=20 it is "
        f"{eb_c:.4f} — about {eb_c/eb_f:.0f}× worse — and it concentrates near "
        f"S≈{loc_c:.0f}, the bump's peak, where the target bends fastest. Straight "
        "segments cut the corner exactly where the curve bends most, so trading firms "
        "spend their limited strikes where the bend is. (That bend has a famous name — "
        "**gamma**, coming in Module 5.)")

    # -- experiment (b): the log strip on module 2's skew vs one flat ATM vol.
    #    Strip price = Z·g(κ) + g'(κ)(S0 − κZ) + Σ w_i C(K_i);  E_Q[−ln(S_T/F)] = price/Z,
    #    and a VIX-style vol is sqrt(2/T · E_Q[−ln(S_T/F)]). Under flat BS vol σ,
    #    E[−ln(S_T/F)] = σ²T/2 exactly, so the flat strip must recover ATM vol.
    a2, b2, r2, m2_, sg2 = 0.001, 0.04, -0.35, 0.02, 0.25   # module-2 equity-skew preset
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
        "m3_exp2", "why the VIX sits above at-the-money volatility",
        "Background from Module 2: the market quotes a different **implied volatility** "
        "— its price-implied forecast of how much the stock will move — at every strike. "
        "That curve of vols (the **smile**) tilts upward toward low strikes: crash "
        "protection costs extra (**skew**). Now select the **log contract** target at "
        "ΔK=2.5 — the strip below, scaled by 2/T, is the VIX portfolio. **Predict:** if "
        "every call is re-priced with those realistic tilted vols instead of one single "
        "at-the-money number, does the strip get cheaper, dearer, or stay the same?",
        f"Priced with one flat vol equal to the at-the-money value ({iv_atm:.1%}), the "
        f"strip implies a VIX-style vol of {vol_flat:.1%} — the same number handed back, "
        "up to the strike-spacing error. Re-priced on module 2's tilted smile, the *same "
        f"weights* cost {c_svi:.4f} instead of {c_flat:.4f} (+{c_svi - c_flat:.4f}) and "
        f"imply {vol_svi:.1%} — {(vol_svi - vol_flat) * 100:.1f} vol points above "
        "at-the-money. The $1/K^2$ weighting buys most heavily at low strikes — exactly "
        "the crash-protection options the market charges extra for. That surcharge is "
        "why the VIX prints above at-the-money volatility. (The exact flat-vol identity "
        "behind this check lives in 'The math, gently'.)")

    t_int, t_frm, t_deep = st.tabs(["Why this works", "The math, gently",
                                    "How the pros use it"])
    with t_int:
        st.markdown(
            "A bond sets the level; a stock sets the slope; neither can *bend*. A call "
            "is pure bend — its payoff kinks upward at its strike and is straight "
            "everywhere else. So trace any smooth payoff the way you'd bend a flexible "
            "ruler along a curve: pin the height and slope at one anchor $\\kappa$, then "
            "buy $g''(K)\\,\\Delta K$ of bend at each strike — positive amounts (long "
            "calls) where the target curves up, negative (sold calls) where it curves "
            "down.\n\n"
            "Your strike menu is your resolution: bend you can't buy is error you must "
            "hold. And because every piece trades at a visible price, the finished "
            "payoff is priced by adding up quotes. Same payoff, same price — the "
            "market's whole probability distribution for $S_T$ is already baked into "
            "those quotes, so it never has to be written down.")
    with t_frm:
        st.markdown(
            "In words: the target equals its height at the anchor, plus its tangent "
            "line, plus bend bought strike by strike — options below the anchor, "
            "options above it. Exactly:")
        st.latex(r"g(S_T)=g(\kappa)+g'(\kappa)\,(S_T-\kappa)"
                 r"+\int_0^{\kappa}\! g''(K)(K-S_T)^+dK+\int_{\kappa}^{\infty}\! g''(K)(S_T-K)^+dK")
        st.markdown(
            "- $(x)^+$ means $\\max(x, 0)$ — keep the positive part, else zero\n"
            "- $g(\\kappa)$ — bonds: the level at the anchor $\\kappa$\n"
            "- $g'(\\kappa)(S_T-\\kappa)$ — shares: the tangent line at $\\kappa$\n"
            "- $g''(K)\\,dK$ — option quantity at strike $K$: curvature density\n"
            "- $(S_T-K)^+$ — a call's payoff. $(K-S_T)^+$ is a **put** — the mirror-image "
            "right to *sell* at $K$; a put converts into a call plus stock and bond "
            "(**put–call parity**, Module 1), so using puts below $\\kappa$ and calls "
            "above is a convenience, not a new assumption\n"
            "- the identity is exact for any twice-differentiable $g$ and for **any** "
            "distribution of $S_T$ — that is why pricing the strip needs no model\n\n"
            "One precise statement behind experiment (b): the strip's price, divided by "
            "the bond price $Z$, is $\\mathbb{E}_Q[-\\ln(S_T/F)]$ — an expectation under "
            "the **risk-neutral** distribution $Q$, the betting odds read out of prices "
            "(they bake in risk premiums, so they are not the market's forecast). In a "
            "flat Black–Scholes world — one lognormal with a single volatility "
            "$\\sigma$ — that expectation is exactly $\\sigma^2 T/2$, which is why the "
            "flat-vol strip hands back the at-the-money vol.")
    with t_deep:
        st.markdown(
            "*Heads-up: this tab uses trader vocabulary more freely — it shows where "
            "the machinery runs in production.*")
        st.latex(r"\mathrm{VIX}^2=\frac{2}{T}\sum_i \frac{\Delta K_i}{K_i^2}\,e^{rT}Q(K_i)"
                 r"-\frac{1}{T}\left(\frac{F}{K_0}-1\right)^2")
        st.markdown(
            "Cboe's VIX is exactly this strip priced off the S&P 500 option chain: "
            "quoted prices $Q(K_i)$ of **out-of-the-money** options (puts below $F$, "
            "calls above — the ones that pay only after a move), the log contract's "
            "weights $\\Delta K_i/K_i^2$ (its bend is $g''=1/K^2$), anchor $K_0$ = "
            "first strike below $F$, and the $(F/K_0-1)^2$ term correcting the "
            "bond-and-stock part. The 'fear index' is a Carr–Madan strip in "
            "production — a **static replication**: assemble once, no re-trading "
            "needed.\n\n"
            "Two biases desks check. Truncating the far ends of the strike range "
            "(traders call these the **wings**) drops real bend, biasing the strip "
            "*low*. Coarse strikes bias it *high*: the straight-segment build "
            "over-replicates the convex log payoff, so discretization overstates the "
            "variance strike. And crash skew fattens the $1/K^2$-weighted cost of low "
            "strikes, holding the VIX above at-the-money implied vol.\n\n"
            "*Preview of Modules 4–5, in trader shorthand:* variance swaps are dealt "
            "the same way — the dealer who sells realized variance buys the strip and "
            "delta-hedges the log contract, making realized-vs-implied variance the "
            "cleanest density-disagreement trade there is.")


# ======================================================================= 4
def module_inverse():
    st.header("Your distribution vs the market's")
    st.caption("Modules 1–3 solved the *forward* problem: from option prices to the market's "
               "probability distribution. This page inverts it: start from the distribution "
               "**you** believe, and build the option position that profits where the two differ.")
    learn("start from odds you believe and build the bet that grows your money fastest if you're right",
          "boil the whole disagreement down to one number and read it as your expected edge",
          "see what conviction costs — how often the bet loses, and what you typically end with, "
          "over 20,000 simulated expiries")

    # ------------------------------------------------------------- start here
    st.markdown("### The idea")
    st.markdown(
        "Every option is a bet that settles on a fixed date — the **expiry**. Throughout this "
        "page, $S_T$ is the stock's price on that date. A **payoff** is a curve telling you how "
        "many dollars you receive for each possible $S_T$.\n\n"
        "Say a stock trades at \\$100 today. A **call** with **strike** \\$110 — the right to buy "
        "at a locked-in \\$110 — pays $S_T-110$ if the stock finishes above \\$110, and nothing "
        "otherwise. You might pay \\$4 today (the **premium**) for that right. Real options are "
        "listed only at fixed strikes (\\$100, \\$105, \\$110, …). *New here? Module 1 builds all "
        "of this from scratch.*")
    st.markdown(
        "**Two densities live on this page.** Option prices reveal the odds the market charges "
        "for every finish; module 2 turned those prices into a full density over $S_T$ — call it "
        "$q$, the market's odds. Your own density $p$ has a different source: replaying 8 years "
        "of the stock's actual history. Wherever $p$ and $q$ differ, some finishes are "
        "**underpriced** to you. The ratio $p/q$ maps exactly where — 1 means fairly priced.\n\n"
        "**How to bet on the difference.** Think of the option market as a bookie posting odds "
        "on every finish. **Kelly's rule** for growing a bankroll over repeated bets: split your "
        "wealth across outcomes in proportion to *your* probabilities, whatever the odds. The "
        "result is the **growth-optimal** payoff — the one maximizing your expected log-wealth:")
    st.latex(r"g^*(S_T)\;=\;\frac{W}{Z}\,\frac{p(S_T)}{q(S_T)}")
    st.markdown(
        "- $S_T$ — the stock price at expiry\n"
        "- $p$ — your density for $S_T$ (from history) · $q$ — the market's (from option prices)\n"
        "- $W$ — your budget in dollars\n"
        "- $Z$ — the price today of \\$1 delivered at expiry; parking the whole budget safely "
        "(a **zero-coupon bond**) turns $W$ into $W e^{rT}$\n"
        "- $g^*$ — your budget, grown at the safe rate, then tilted toward the finishes you "
        "think are underpriced")
    st.markdown(
        "One dial and one scoreboard complete the setup. The **Kelly fraction** $f$ is a caution "
        "dial: it blends your density toward the market's, giving the $p_f$ in every chart. "
        "$f=0$ means take the market's odds — no bet; $f=1$ means fully back your own. The "
        "scoreboard is **KL($p_f$‖$q$)** — the Kullback–Leibler divergence, one number for the "
        "whole disagreement. It is measured in **nats** (natural-log units) and, as 'The math, "
        "gently' shows, it equals your expected log-growth edge over the trade: 0.02 nats ≈ a "
        "2% edge, and 0 means no disagreement, no trade.")

    quiz("m4_quiz",
         "Below there is a caution dial $f$ — how much you trust your own odds over the "
         "market's. You set $f=0$: no opinion at all. The best possible use of your budget "
         "becomes…",
         ["park it in the safe bond and collect interest", "buy the stock",
          "bet big on both tails"], 0,
         "No disagreement means no bet. At $f=0$ the blended density $p_f$ equals the market's "
         "$q$, the ratio $p_f/q$ is 1 everywhere, and $g^* = W/Z$ — a flat payoff. That is "
         "exactly the zero-coupon bond: pay your budget $W$ today, receive $W e^{rT}$ at expiry "
         "no matter where the stock finishes. Options enter only where the densities differ.")

    c = st.columns(4)
    ticker = c[0].text_input("Ticker", "SOXX", key="m4_ticker",
                             help="Which stock's option market to read. SOXX is a "
                                  "semiconductor-sector fund.")
    dte = c[1].slider("Days to expiry (DTE)", 14, 180, 45, key="m4_dte",
                      help="Options settle on a fixed date. This sets how many days away that "
                           "date is; S_T in the charts is the stock price on that day.")
    fraction = c[2].slider("f — trust in your own odds (0 = market's, 1 = fully yours)",
                           0.0, 1.0, 0.5, 0.05, key="m4_f",
                           help="The caution dial. f = 0: take the market's odds — no bet, just "
                                "the safe bond. f = 1: fully back your own odds. In between: the "
                                "same bets, shrunk toward caution. The exact blending formula "
                                "lives in 'The math, gently'.")
    wealth = c[3].number_input("Budget ($)", 1_000, 1_000_000, 10_000, step=1_000, key="m4_wealth",
                               help="W in the formulas: the total you are willing to stake.")

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
        st.info("Running on realistic simulated data — no live-market API key is set. Every "
                "chart and number works identically; with a key, this page reads real option "
                "quotes instead.")

    chain, q, p_view, g, ticket, summary, kl = solve(ticker, dte, fraction, wealth)
    grid = q.grid
    # display ratio capped at 8 to mirror log_optimal_payout(ratio_cap=8)
    ratio_c = np.minimum(p_view / np.maximum(q.pdf, 1e-12), 8.0)

    stage = stepper("m4_stage", ["1 · the market's odds vs yours", "2 · where you disagree (p/q)",
                                 "3 · the bet that exploits it", "4 · build it from real options"])

    def draw_densities(ax):
        ax.plot(grid, q.pdf, color=INK, label="market's odds $q$")
        ax.plot(grid, p_view, color=BLUE, label=f"your odds $p_f$ (f={fraction:.2f})")
        ax.axvline(chain.spot, color=MUT, lw=1, ls=":", label="today's price")
        diff = p_view - q.pdf
        if diff.max() > 1e-5:
            j = int(np.argmax(diff))
            ax.annotate("you think this region is more\nlikely than the market does",
                        xy=(grid[j], p_view[j]),
                        xytext=(0.68, 0.55), textcoords="axes fraction", fontsize=8,
                        color=INK, arrowprops=dict(arrowstyle="->", color=MUT))
        ax.set_xlabel("$S_T$ — stock price at expiry")
        ax.set_ylabel("probability density")
        ax.set_title("The market's odds vs yours"); ax.legend(fontsize=9)

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
            ax.annotate("f = 0: $p_f = q$, ratio ≡ 1 — nothing to bet on", xy=(0.5, 0.6),
                        xycoords="axes fraction", ha="center", fontsize=8, color=INK)
        ax.set_xlabel("$S_T$ — stock price at expiry"); ax.set_ylabel("$p_f/q$ (log scale)")
        ax.set_title("Where you disagree — the ratio $p_f/q$")

    def draw_payoff(ax, with_repl):
        ax.plot(grid, g, color=INK, label="ideal bet $g^*=(W/Z)\\,p_f/q$")
        if with_repl:
            ax.plot(grid, ticket.payoff(grid), color=BLUE, ls="--",
                    label="copy built from listed strikes")
            ax.annotate("flat past the last listed strike:\nthe bet is bounded",
                        xy=(grid[-1], float(ticket.payoff(grid[-1:])[0])),
                        xytext=(0.60, 0.10), textcoords="axes fraction", fontsize=8,
                        color=INK, arrowprops=dict(arrowstyle="->", color=MUT))
        ax.axhline(wealth, color=MUT, lw=1, ls=":")
        cross = np.where(np.diff(np.sign(g - wealth)) != 0)[0]
        if len(cross):
            i = int(cross[-1])                # a wavy g* can cross W several times
            ax.annotate(f"last breakeven: $g^*=W$\nat $S_T \\approx {grid[i]:.0f}$ "
                        f"({len(cross)} crossing{'s' if len(cross) > 1 else ''} in all)",
                        xy=(grid[i], wealth),
                        xytext=(0.05, 0.75), textcoords="axes fraction", fontsize=8,
                        color=INK, arrowprops=dict(arrowstyle="->", color=MUT))
        ax.set_xlabel("$S_T$ — stock price at expiry")
        ax.set_ylabel("dollars received at expiry")
        ax.set_title("The bet that exploits it"); ax.legend(fontsize=9)

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
    st.caption([
        "Black: the market's odds $q$, read out of option prices (module 2). Blue: your odds "
        "$p_f$ — 8 years of history, blended toward the market by the caution dial $f$. The "
        "y-axis is probability density; the area under each curve is 1.",
        "Right panel — above the line at 1, you think that finish is more likely than the "
        "market is charging for: those states are cheap to you. Log scale: each gridline step "
        "is a constant multiple.",
        "Reading the payoff panel: x is the finish $S_T$, y is the dollars the bet delivers. "
        "The ideal bet is the ratio curve times $W/Z$ — the safe bond, tilted toward your view.",
        "Real options exist only at listed strikes. The dashed curve is the closest tradable "
        "copy of the ideal bet built from those; past the highest listed strike it must go flat.",
    ][stage])

    # narration: top decile of q, your probability of it, average payout there.
    # Before the "build it from real options" stage, speak of the ideal g*, not the ticket.
    x90 = float(np.interp(0.9, q.cdf, grid))
    mask = grid >= x90
    q_tail = float(np.trapezoid(q.pdf[mask], grid[mask]))
    p_tail = float(np.trapezoid(p_view[mask], grid[mask]))
    pay_curve = ticket.payoff(grid) if stage == 3 else g
    pay_name = "the traded ticket" if stage == 3 else "the ideal bet $g^*$"
    avg_pay = float(np.trapezoid(p_view[mask] * pay_curve[mask], grid[mask])) / max(p_tail, 1e-12)
    p_loss = float(summary["P(lose money) under your view"])
    loss_moral = ("growth-optimal bets accept frequent small losses in exchange for rare "
                  "large wins" if p_loss >= 0.05 else
                  "at this caution setting there is barely a bet, so there is barely "
                  "anything to lose")
    narrate(f"The market's prices give finishes above \\${x90:,.0f} a {q_tail:.0%} chance — its "
            f"top decile. You give them {p_tail:.0%}, {p_tail / max(q_tail, 1e-12):.1f}× the "
            f"market, and {pay_name} pays \\${avg_pay:,.0f} on average when that happens "
            f"({avg_pay / wealth:.2f}× budget). By your own odds it loses money "
            f"{p_loss:.0%} of the time — {loss_moral}.")

    # metrics appear as their objects enter the construction — no spoilers
    m = st.columns(4)
    m[0].metric("KL(p_f‖q) — disagreement", f"{kl:.4f} nats",
                help="One number for how far your odds sit from the market's "
                     "(Kullback–Leibler divergence). 0 = no disagreement, no trade. It equals "
                     "your expected log-growth edge over the trade's life: 0.02 nats ≈ a 2% "
                     "edge. A nat is the natural-log unit of information.")
    if stage >= 2:
        m[1].metric("P(lose) by your own odds", f"{p_loss:.0%}",
                    help="High on purpose: growth-optimal bets lose often and win big. The "
                         "simulator below re-estimates this number from 20,000 draws — any "
                         "difference is sampling noise.")
    if stage >= 3:
        m[2].metric("annualized log growth (as traded)",
                    f"{summary['annualized (replicated)']:.1%}",
                    help="includes the risk-free carry: at f = 0 this is exactly the interest "
                         "rate r, while the disagreement (and the edge) is 0")
        m[3].metric("ticket cost", f"${ticket.cost:,.0f}",
                    help="below budget: listed strikes cut off the ideal bet's tails, so the "
                         "tradable copy costs less than the ideal one")

    if stage == 3:
        with st.expander("Order ticket — the exact options to buy and sell "
                         "(educational: mid-quote fills, no fees; not investment advice)"):
            st.caption(f"{ticket.stock:+,.1f} shares plus \\${ticket.bonds:,.0f} face value of "
                       "the safe bond (negative = borrowing), plus the option legs below. Below "
                       "today's price the bet is built from **puts** — the right to *sell* at "
                       "the strike — because those are the actively traded contracts down "
                       "there; puts and calls are interchangeable via put–call parity "
                       "(module 1). In the table, **mid** is each contract's market price — "
                       "the midpoint between the best buy and sell quotes.")
            legs = ticket.legs.assign(side=np.where(ticket.legs["qty"] >= 0, "BUY", "SELL"))
            st.dataframe(legs[["type", "strike", "side", "qty", "mid", "cost"]].round(2),
                         hide_index=True)

    # ------ outcome simulator: only once the TICKET exists (stage 4, "build it from real options")
    if stage == 3:
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
        ax2.minorticks_off()             # minor tick labels collide at the left edge
        ax2.axvline(med, color=INK, lw=1.5)
        ax2.annotate(f"median ${med:,.0f}", xy=(med, ax2.get_ylim()[1] * 0.80),
                     xytext=(0.74, 0.55), textcoords="axes fraction", fontsize=8, color=INK,
                     bbox=dict(boxstyle="round,pad=0.2", fc="#fcfcfb", ec="none", alpha=0.9),
                     arrowprops=dict(arrowstyle="->", color=MUT))
        ax2.set_xlabel("your final wealth (dollars, log scale)")
        ax2.set_title("Outcome simulator — 20,000 expiries drawn from your odds $p_f$")
        ax2.legend(fontsize=9, loc="upper left")
        st.pyplot(fig2)
        st.caption(f"The traded ticket costs \\${ticket.cost:,.0f} of the \\${wealth:,.0f} "
                   "budget — listed strikes cut off the ideal bet's tails — so 'losing' here "
                   f"means finishing below the \\${ticket.cost:,.0f} actually paid.")

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
        reveal_f = (f"Computed live — f = 1.00: you lose {pl1:.0%} of the time, median "
                    f"${md1:,.0f}, middle 90% of outcomes ${lo1:,.0f}–${hi1:,.0f}.  f = 0.25: "
                    f"lose {pl2:.0%}, median ${md2:,.0f}, ${lo2:,.0f}–${hi2:,.0f}. The caution "
                    "dial mostly rescales *sizes*, not frequency: quarter-Kelly makes the same "
                    "bets at a quarter of the intensity, so you lose about as often, but both "
                    "the best and worst outcomes squeeze in toward the safe bond. (Precisely: "
                    "$(p_f/q)\\propto(p/q)^f$ — every ratio raised to the $f$-th power, up to "
                    "the normalizing constant of $p_f$.) Betting full strength grows wealth "
                    "fastest only if your odds $p$ are exactly right — and they never are.")
    else:
        reveal_f = "…"
    experiment("m4_exp_f", "full vs quarter Kelly",
               "Set **f = 1.0** (fully back your own odds) and note the simulator's P(loss), "
               "the median, and the worst bar. Now set **f = 0.25**. Which changed more — how "
               "*often* you lose, or how *much*?",
               reveal_f)

    if st.session_state.get("m4_exp_dte"):
        _, _, _, _, _, _, kl14 = solve(ticker, 14, fraction, wealth)
        ch14, _, _ = load(ticker, 14)
        _, _, _, _, _, _, kl180 = solve(ticker, 180, fraction, wealth)
        ch180, _, _ = load(ticker, 180)
        t14, t180 = ch14.T, ch180.T
        reveal_d = (f"Computed live at f = {fraction:.2f}: disagreement over the trade's life "
                    f"is {kl14:.3f} nats at 14 days to expiry and {kl180:.3f} at 180 — "
                    f"{'more' if kl180 > kl14 else 'less'} lifetime edge at the long horizon — "
                    f"but per year that is {kl14 / t14:.2f} vs {kl180 / t180:.2f} nats. An edge "
                    "in the average return needs time to stand out from day-to-day randomness, "
                    "so lifetime edge builds as expiry moves out. But your odds $p$ come from "
                    "replaying chunks of the same 8 years of history, and every extra day "
                    "stretches that limited history further. The risk that $p$ itself is wrong "
                    "grows with the horizon at least as fast as the edge does — one more reason "
                    "to keep $f$ below 1.")
    else:
        reveal_d = "…"
    experiment("m4_exp_dte", "does edge compound with horizon?",
               "Push **days to expiry (DTE) from 14 to 180** and watch the disagreement metric "
               "and the payoff's shape (the tilt gets bigger and smoother). Is the *per-year* "
               "edge growing too?",
               reveal_d)

    # ------------------------------------------------------------------- theory
    tab_why, tab_math, tab_pros = st.tabs(["Why this works", "The math, gently",
                                           "How the pros use it"])
    with tab_why:
        st.markdown(
            "**The bookie, in full.** The option market posts odds on every possible finish "
            "$S_T$; those odds are exactly the density $q$ you extracted in module 2. Kelly's "
            "rule for growing a bankroll over repeated bets: split your wealth across outcomes "
            "in proportion to **your** probabilities $p$, regardless of the odds on offer.\n\n"
            "Why does that produce $g^*=(W/Z)\\,p/q$? A dollar delivered only in state $S$ "
            "costs $Z\\,q(S)$ per unit today. Allocate the slice $W\\cdot p(S)$ of your budget "
            "to state $S$ and it buys $g(S)=(W/Z)\\,p(S)/q(S)$ — automatically large exactly "
            "where the market undercharges relative to your view. And if $p=q$, you have bought "
            "the same amount in every state: the safe bond, back again.")
    with tab_math:
        st.markdown(
            "The whole page is one optimization: choose the payoff $g$ that maximizes your "
            "expected log-wealth, subject to being able to afford it at the market's prices.")
        st.latex(r"\max_g\;\mathbf{E}_p[\log g]\qquad\text{s.t.}\qquad Z\,\mathbf{E}_q[g]=W")
        st.markdown(
            "- $\\mathbf{E}_p[\\log g]$ — expected log of final wealth under **your** density "
            "$p$; maximizing it is what \"growth-optimal\" means\n"
            "- $Z\\,\\mathbf{E}_q[g]$ — the payoff's price today: any payoff's price is its "
            "expectation under the market's density $q$, discounted by $Z$ (module 3, run "
            "forward)\n"
            "- $W$ — the budget that price must equal\n\n"
            "One Lagrange multiplier $\\lambda$ (the shadow price of budget) does the rest:")
        st.latex(r"\mathcal{L}=\int p\log g\,dS-\lambda\Big(Z\!\int q\,g\,dS-W\Big)"
                 r"\;\Rightarrow\;\frac{p}{g}=\lambda Z q\;\Rightarrow\;g=\frac{p}{\lambda Z q}")
        st.latex(r"\text{budget}\Rightarrow\lambda=\tfrac{1}{W}:\qquad g^*=\frac{W}{Z}\,\frac{p}{q},"
                 r"\qquad\mathbf{E}_p\!\left[\log\tfrac{g^*}{W}\right]=\mathrm{KL}(p\,\|\,q)-\log Z")
        st.markdown(
            "That last identity is the scoreboard's meaning: expected log growth = "
            "KL disagreement plus the safe rate ($-\\log Z = rT$).\n\n"
            "The caution dial blends densities geometrically:")
        st.latex(r"p_f\;\propto\;p^{f}\,q^{1-f}\qquad\Longrightarrow\qquad"
                 r"\frac{p_f}{q}\;\propto\;\Big(\frac{p}{q}\Big)^{f}")
        st.markdown(
            "($\\propto$ means \"proportional to\": equal up to the constant that makes $p_f$ "
            "integrate to 1.) So fractional Kelly *is* the classic \"bet a fraction $f$ of the "
            "Kelly bet\", done in density space.\n\n"
            "- $p$ — your density: multi-day chunks of the last 8 years of returns, resampled "
            "(a block bootstrap), then blended toward $q$ by $f$\n"
            "- $q$ — the market-implied density of module 2\n"
            "- $Z$ — the zero-coupon bond price $e^{-rT}$ · $W$ — your budget\n"
            "- KL — your expected log-growth edge, in nats, over the trade's life\n\n"
            "One naming note: $q$ is called the **risk-neutral** density. It is the market's "
            "*betting odds*, not its forecast — prices bake in what people pay for insurance, "
            "so $q$ need not be anyone's true belief.")
    with tab_pros:
        st.markdown(
            "*Fair warning: this tab talks the way trading desks do.*\n\n"
            "**Not all disagreement is edge.** People rationally pay extra for insurance "
            "against crashes, so crash-region prices look \"too high\" to any history-based "
            "model — that part of the gap between $p$ and $q$ is compensation for risk, not a "
            "mistake. Technically: $q$ is the real-world density reweighted by marginal "
            "utility, so crash states look 'overpriced' to a bootstrap by construction; a "
            "rational agent deliberately leaves some KL on the table.\n\n"
            "**Implementation reality.** The ticket assumes fills at the quote midpoint (real "
            "fills are worse) and European exercise — settle only at expiry — while single-name "
            "US options are American, exercisable any day. Listed strikes truncate the ideal "
            "payoff's tails (hence the cap of 8 on the ratio $p_f/q$), and the bootstrap "
            "assumes the future keeps resampling the past 8 years. Full Kelly on a "
            "mis-specified $p$ is how accounts die; $f<1$ is the apology.")


# ======================================================================= 5
def module_greeks():
    from scipy.stats import norm
    from matplotlib.colors import LinearSegmentedColormap

    st.header("The Greeks: the slope, the bend, and the daily rent")
    st.caption("**Greeks** are traders' nicknames — Greek letters — for how an option's "
               "price reacts when one of its inputs moves. This page builds three of them, "
               "one at a time, out of a single price curve.")

    st.markdown("### The idea")
    st.markdown(
        "This app has one mission: option prices encode the market's probability "
        "distribution for a stock's price on a future date, and you can read that "
        "distribution out of them. This page is the flip side of the story. The same "
        "curvature that stores the distribution is a live risk someone must manage — "
        "and the Greeks are how traders measure it.\n\n"
        "The instrument here is a **call option**: a contract giving you the right, "
        "not the obligation, to buy a stock at a locked-in price (the **strike**) on a "
        "set future date (the **expiry**). You pay for that right up front — the "
        "**premium**. New here? Module 1 builds these from scratch.\n\n"
        "Say the stock trades at \\$105 and you hold a call with strike \\$100 expiring "
        "today. Buy at \\$100, sell at \\$105: the contract is worth \\$5. Had the stock "
        "been at \\$95, the right to pay \\$100 would be worth nothing. So at expiry the "
        "call's **payoff** — what the contract hands you — is $\\max(S-100,\\,0)$: flat "
        "at zero, then diagonal. That is the dotted hockey-stick in the first chart.\n\n"
        "Before expiry the outcome is still uncertain, so the call has a smooth value at "
        "every stock price — the black curve $C(S)$. This lab computes it with the "
        "standard **Black–Scholes model**, the textbook pricing formula:")
    st.latex(r"C \;=\; C_{\mathrm{BS}}(S,\,K,\,\sigma,\,T,\,r)")
    st.markdown(
        "- $S$ — today's stock price (traders call it the **spot**); your first slider\n"
        "- $K$ — the **strike**, the locked-in buy price; fixed at 100 in this lab\n"
        "- $\\sigma$ — **volatility**: the stock's yearly wiggle-rate, a standard "
        "deviation of returns. $\\sigma = 0.30$ means a typical year moves the stock "
        "about ±30%. When this number is read *out of* option prices it is called "
        "**implied volatility**; here it is your second slider\n"
        "- $T$ — years until expiry; the days slider ÷ 365\n"
        "- $r$ — the interest rate a bank deposit earns; fixed at 4% here\n\n"
        "Three more pieces and you can read everything below.\n\n"
        "**Where the stock sits.** At $S \\approx 100$ the call is **at-the-money** "
        "(stock at the strike). Above 100 it is **in-the-money**; below, "
        "**out-of-the-money**.\n\n"
        "**The dealer.** Whoever sold you the call loses when the stock rises. So she "
        "**short-sells** shares — borrows them and sells, a position that profits when "
        "the stock falls — to cancel that risk. Balanced this way she is **hedged**, or "
        "**flat**: immune to small moves. The right balance shifts as the stock moves, "
        "so she must re-adjust constantly.\n\n"
        "**The Greeks.** **Δ (delta)** is the price curve's slope: dollars gained per "
        "\\$1 stock move — and the dealer's share count. **Γ (gamma)** is the bend: how "
        "fast Δ itself changes. **Θ (theta)** is the rent: dollars the option loses per "
        "calendar day from time alone. (A fourth, **vega**, waits in \"The math, "
        "gently\".)\n\n"
        "One memory to bring along: module 2 built the market's probability density for "
        "where the stock ends up — a bell curve. Keep it in mind. Γ will turn out to be "
        "that same bell.")

    learn("read Δ, the price curve's slope, as the number of shares the dealer shorts to stay flat",
          "see Γ, the bend — and recognize its bell shape as module 2's probability "
          "density in different clothes",
          "read Θ as the daily rent paid for the bend, with a Black–Scholes bookkeeping "
          "identity making rent and bend offset")

    quiz("m5_quiz",
         "The stock sits exactly at the 100 strike (**at-the-money**) and expiry is days "
         "away. As the clock runs out, Γ — the bend of the price curve —…",
         ["shrinks toward 0", "explodes", "stays constant"], 1,
         "Explodes. With months left the price curve is a gentle arc; at the very end it "
         "must sharpen into the payoff's corner at 100, and the bend at that corner grows "
         "without limit. Check it below: park the stock at 100 and drag *days to expiry* "
         "toward 1 — stage 2 adds a heatmap where you can watch it too. Formula, for "
         "later: $\\Gamma_{ATM} \\approx \\varphi(d_1)/(S\\sigma\\sqrt{T})$, where "
         "$\\varphi$ is the standard normal density and $d_1$ a model intermediate (both "
         "glossed in \"The math, gently\") — the $\\sqrt{T}$ in the denominator blows up "
         "as $T \\to 0$.")

    K = 100.0
    r = 0.04
    st.caption("**Fixed for this lab:** one call option, strike K = 100 (the locked-in "
               "buy price), interest rate r = 4%. Your three sliders control everything "
               "else.")
    c1, c2, c3 = st.columns(3)
    Sspot = c1.slider("Stock price today (spot S)", 60.0, 140.0, 105.0, 1.0, key="m5_S",
                      help="Where the stock trades right now — the blue dot on the charts.")
    vol = c2.slider("Volatility σ — yearly wiggle-rate", 0.10, 0.90, 0.30, 0.05,
                    key="m5_vol",
                    help="0.30 ≈ the market expects a typical year to move the stock "
                         "about ±30%. 'Implied' volatility is this number read out of "
                         "option prices; here you set it directly.")
    days = c3.slider("Days to expiry", 1, 365, 90, key="m5_days",
                     help="Calendar days until the contract's end date. T in the "
                          "formulas is this ÷ 365.")
    T = days / 365.0

    def d1d2(S, t=T, v=vol):
        S = np.asarray(S, float)
        d1 = (np.log(S / K) + (r + v**2 / 2) * t) / (v * np.sqrt(t))
        return d1, d1 - v * np.sqrt(t)

    def call(S, t=T, v=vol):
        d1, d2 = d1d2(S, t, v)
        return np.asarray(S) * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)

    def delta(S, t=T, v=vol):
        return norm.cdf(d1d2(S, t, v)[0])

    def gamma(S, t=T, v=vol):
        return norm.pdf(d1d2(S, t, v)[0]) / (np.asarray(S) * v * np.sqrt(t))

    def theta_day(S, t=T, v=vol):          # per calendar day
        d1, d2 = d1d2(S, t, v)
        yearly = (-np.asarray(S) * norm.pdf(d1) * v / (2 * np.sqrt(t))
                  - r * K * np.exp(-r * t) * norm.cdf(d2))
        return yearly / 365.0

    stage = stepper("m5_stage", ["1 · price & Δ (the slope)", "2 · + Γ (the bend)",
                                 "3 · + Θ (the rent)"], )

    S = np.linspace(60, 140, 801)
    C = call(S)
    D0, G0, Th0 = float(delta(Sspot)), float(gamma(Sspot)), float(theta_day(Sspot))
    C0 = float(call(Sspot))

    fig, axes = fig_axes(2)
    ax = axes[0]
    ax.plot(S, C, color=INK, label=f"call value $C(S)$, {days} days left")
    ax.plot(S, np.maximum(S - K, 0), color=MUT, ls=":", lw=1.2,
            label="payoff at expiry: max(S − 100, 0)")
    tang = C0 + D0 * (S - Sspot)
    ax.plot(S, tang, color=BLUE, ls="--", lw=1.4, label=f"tangent line: slope Δ = {D0:.2f}")
    ax.plot([Sspot], [C0], marker="o", ms=6, color=BLUE)
    ax.annotate(f"here: C = {C0:.2f}, Δ = {D0:.2f}\nthe dealer shorts {D0:.2f}\n"
                "shares to stay flat",
                xy=(Sspot, C0), xytext=(-95, 30), textcoords="offset points", fontsize=8,
                color="#52514e",
                bbox=dict(boxstyle="round,pad=0.25", fc="#fcfcfb", ec="none", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color=MUT, lw=1))
    ax.set_xlabel("$S$"); ax.set_ylabel("value")
    ax.set_title("The price curve and its tangent")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    if stage == 0:
        ax.plot(S, delta(S), color=BLUE)
        ax.plot([Sspot], [D0], marker="o", ms=6, color=BLUE)
        ax.set_ylabel("Δ")
        ax.set_title("Δ(S) — the slope, at every stock price")
        ax.annotate("ramps 0 → 1 as S crosses the\n100 strike; how steep this ramp\n"
                    "is will be our next letter, Γ",
                    xy=(K, float(delta(K))), xytext=(12, -34), textcoords="offset points",
                    fontsize=8, color=MUT,
                    arrowprops=dict(arrowstyle="->", color=MUT, lw=1))
    elif stage == 1:
        ax.plot(S, gamma(S), color=GREEN)
        ax.plot([Sspot], [G0], marker="o", ms=6, color=GREEN)
        ax.set_ylabel("Γ")
        ax.set_title("Γ(S) — the bend")
        ax.annotate("a bell centered near 100 — module 2's\nmarket density of where the "
                    "stock ends up\n(there plotted vs strike K, here vs spot S)",
                    xy=(K, float(gamma(K))), xytext=(-120, -40), textcoords="offset points",
                    fontsize=8, color="#52514e",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#fcfcfb", ec="none", alpha=0.9),
                    arrowprops=dict(arrowstyle="->", color=MUT, lw=1))
    else:
        ax.plot(S, theta_day(S), color=MAGENTA)
        ax.plot([Sspot], [Th0], marker="o", ms=6, color=MAGENTA)
        ax.axhline(0, color=MUT, lw=1)
        ax.set_ylabel("Θ ($/day)")
        ax.set_title("Θ(S) — the rent: most negative at the strike")
        ax.annotate("deepest exactly where Γ peaks:\nthe rent pays for the bend",
                    xy=(K, float(theta_day(K))), xytext=(-115, -34),
                    textcoords="offset points", fontsize=8, color="#52514e",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#fcfcfb", ec="none", alpha=0.9),
                    arrowprops=dict(arrowstyle="->", color=MUT, lw=1))
    ax.set_xlabel("$S$")
    st.pyplot(fig)

    move = 5.0
    hedge_gap = 0.5 * G0 * move**2
    carry = 0.5 * vol**2 * Sspot**2 * G0 / 365.0
    if stage == 0:
        narrate(f"At S = {Sspot:g} with {days} days left, the 100-strike call is worth "
                f"\\${C0:.2f} and Δ = {D0:.2f}: a \\$1 rally adds ≈\\${D0:.2f} to the "
                f"call, so the dealer holds {D0:.2f} short shares against it. Small "
                "stock moves now cancel out — she neither gains nor loses (\"flat\") "
                "until the slope itself changes.")
    elif stage == 1:
        narrate(f"Γ = {G0:.4f}: a \\$1 rally lifts Δ from {D0:.2f} to "
                f"≈{float(delta(Sspot + 1)):.2f}, and after a ±\\${move:.0f} move the "
                f"tangent misprices the call by ≈½Γ·{move:.0f}² = \\${hedge_gap:.2f} — "
                "always in the option owner's favor, because a curved line always ends "
                "up above its own tangent. That built-in edge (**convexity**) is not "
                "free: the next stage shows Θ, the daily fee for holding it.")
    else:
        narrate(f"Θ = −\\${abs(Th0):.3f} per day of rent, vs ½σ²S²Γ ≈ \\${carry:.3f} "
                f"per day of expected wiggle-earnings if the stock moves at {vol:.0%} "
                "volatility — notice the two numbers roughly match. No accident: the "
                "Black–Scholes model is built so decay and expected wiggle-earnings "
                "cancel, leaving a hedged position earning only bank-deposit interest "
                "(the 4% rate). The exact bookkeeping is in \"The math, gently\".")

    if stage >= 1:
        Sg = np.linspace(70, 130, 121)
        dg = np.arange(1, 121)
        GG = np.array([gamma(Sg, t=d / 365.0) for d in dg])
        cmap = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#cde2fb",
                                                           "#2a78d6", "#0d366b"])
        figh, (axh,) = fig_axes(1, height=3.0)
        im = axh.pcolormesh(Sg, dg, GG, cmap=cmap, shading="auto")
        axh.axvline(K, color="#fcfcfb", lw=1, ls="--")
        axh.annotate("darkest = biggest bend Γ; near the 100 strike\nwith days running "
                     "out it spikes, and the dealer\nre-hedges frantically (\"pin risk\", "
                     "experiment 1)",
                     xy=(K, 4), xytext=(0.52, 0.30), textcoords="axes fraction",
                     fontsize=8, color=INK,
                     bbox=dict(boxstyle="round,pad=0.25", fc="#fcfcfb", ec="none", alpha=0.9),
                     arrowprops=dict(arrowstyle="->", color=INK, lw=1))
        if 70 <= Sspot <= 130 and days <= 120:
            axh.plot([Sspot], [days], marker="o", ms=6, color=YELLOW, mec=INK, mew=0.8)
            axh.annotate("your sliders", xy=(Sspot, days), xytext=(6, 6),
                         textcoords="offset points", fontsize=8, color=INK)
        axh.set_xlabel("$S$"); axh.set_ylabel("days to expiry")
        axh.set_title("Γ(S, t) — where the dealer must trade hardest")
        figh.colorbar(im, ax=axh, label="Γ (the bend)")
        st.pyplot(figh)

    g_now = float(gamma(K, t=T))
    g_2d = float(gamma(K, t=2 / 365.0))
    experiment(
        "m5_exp_expiry", "watch the bend explode into expiry",
        "Park the stock-price slider at 100 (the strike) and drag **days to expiry** "
        "from 90 down to 2. Watch the Γ panel's y-axis — and, from stage 2 on, the "
        "bright band in the heatmap.",
        f"Computed live: at-the-money Γ is {g_now:.4f} at {days} days and {g_2d:.4f} at "
        f"2 days — ×{g_2d / max(g_now, 1e-12):.0f}. Each halving of the time left "
        "multiplies at-the-money Γ by ≈√2, without bound. Traders call the endgame "
        "**pin risk**: with the stock stuck near 100 at the very end, Δ snaps between "
        "0 and 1 on tiny moves, so the dealer must buy and sell nearly her whole share "
        "position over and over. (Formula, optional: $\\Gamma_{{ATM}} \\approx "
        "\\varphi(d_1)/(S\\sigma\\sqrt{{T}})$ — the $\\sqrt{{T}}$ in the denominator "
        "does the exploding.)")
    g_v1 = float(gamma(Sspot, v=vol))
    g_v2 = float(gamma(Sspot, v=min(vol * 2, 0.9)))
    experiment(
        "m5_exp_vol", "higher volatility spreads the bend out",
        f"Note Γ at your current volatility ({vol:.0%}), then double the σ slider. "
        "Predict first: does Γ rise or fall?",
        f"Falls (at the money): Γ = {g_v1:.4f} at {vol:.0%} vs {g_v2:.4f} at "
        f"{min(vol * 2, 0.9):.0%}. Higher volatility means the final stock price could "
        "land almost anywhere, so the price curve bends gently everywhere instead of "
        "sharply at 100 — the same total bend, spread wider and flatter. That is "
        "exactly a density growing a bigger standard deviation: taller-and-narrow "
        "trades for shorter-and-wide, as with module 2's bells. (Formula, optional: "
        "the $1/\\sigma\\sqrt{{T}}$ in $\\varphi(d_1)/(S\\sigma\\sqrt{{T}})$ does the "
        "flattening.)")

    tab_why, tab_math, tab_pros = st.tabs(["Why this works", "The math, gently",
                                           "How the pros use it"])
    with tab_why:
        st.markdown(
            "**Δ is the slope, Γ the bend, Θ the rent.** Once the dealer hedges away "
            "the slope, what remains is pure bend against pure rent. Every stock wiggle "
            "of size ΔS pays the option's owner ≈½Γ·(ΔS)² — a curved line always "
            "finishes above its own tangent — and every calendar day costs Θ.\n\n"
            "The σ slider is the wiggle-rate you *paid for*: the **implied** "
            "volatility, baked into the option's price. What the stock *actually* does "
            "afterward is the **realized** volatility. Wiggle more than you paid for, "
            "and the ½Γ·(ΔS)² earnings beat the Θ rent; wiggle less, and the rent wins. "
            "That gap is the entire volatility-trading business in one sentence.\n\n"
            "And the reason this page sits in this app: the Γ bell you have been "
            "dragging around *is* the market's probability density from module 2, "
            "wearing risk clothes. The last tab makes that exact.")
    with tab_math:
        st.markdown(
            "Recurring symbols first — all standard-normal objects you already know:\n"
            "- $N(\\cdot)$ — the standard normal CDF\n"
            "- $\\varphi(\\cdot)$ — the standard normal density (the bell curve itself)\n"
            "- $d_1,\\ d_2$ — Black–Scholes intermediates: roughly, how many standard "
            "deviations the stock sits from the strike on a log scale, with "
            "$d_2 = d_1 - \\sigma\\sqrt{T}$\n\n"
            "With those in hand, the Greeks of a call:")
        st.latex(r"\Delta = N(d_1)\qquad \Gamma = \frac{\varphi(d_1)}{S\sigma\sqrt{T}}"
                 r"\qquad \nu = S\varphi(d_1)\sqrt{T}\qquad"
                 r"\Theta = -\frac{S\varphi(d_1)\sigma}{2\sqrt{T}} - rKe^{-rT}N(d_2)")
        st.markdown(
            "- $\\Delta$ — the slope; also the dealer's share count\n"
            "- $\\Gamma$ — the bend: Δ's own rate of change per \\$1 of stock move\n"
            "- $\\nu$ (**vega**) — a bonus Greek: dollars gained per one-percentage-"
            "point rise in σ (traders say \"per vol point\")\n"
            "- $\\Theta$ — dollars lost per year from time alone; the app divides by "
            "365 to show per day\n\n"
            "They are not independent. The Black–Scholes model ties them into one "
            "bookkeeping identity:")
        st.latex(r"\Theta + rS\Delta + \tfrac{1}{2}\sigma^2 S^2 \Gamma = rC")
        st.markdown(
            "Read it left to right: time decay, plus interest on the Δ shares, plus "
            "expected wiggle-earnings from the bend, add up to bank-deposit interest "
            "on the option's value. A hedged option earns exactly the interest rate — "
            "by construction, not coincidence. This is the precise form of stage 3's "
            "\"the two numbers roughly match\": the small mismatch you saw is exactly "
            "those interest terms.")
    with tab_pros:
        st.markdown(
            "*Fair warning: desk vocabulary ahead — each new term gets one gloss as it "
            "lands.*\n\n"
            "**Γ is module 2's density.** Black–Scholes prices scale: double the stock "
            "price and the strike together and the price doubles — "
            "$C(\\lambda S, \\lambda K) = \\lambda C(S,K)$. At flat volatility this "
            "forces $S^2\\,\\partial^2C/\\partial S^2 = K^2\\,\\partial^2C/\\partial "
            "K^2$. In plain words: the bend measured along the stock price (Γ, this "
            "page) equals the bend measured along the strike — and module 2 showed the "
            "strike-direction bend *is* the market's probability density. Same object, "
            "photographed along two different axes.\n\n"
            "So a dealer's *aggregate* gamma map doubles as a map of where the market "
            "has bought its probability mass — in module 2's betting-odds sense (the "
            "**risk-neutral** distribution: risk premiums baked in, not an honest "
            "forecast). It even moves the market. When many contracts are outstanding "
            "at one strike (heavy **open interest**) and the dealers on net *own* that "
            "bend (they are \"**long gamma**\" — the mirror image of this page's "
            "call-selling dealer), their re-hedging sells every rally and buys every "
            "dip (\"**fading**\" the moves). That constant leaning can **pin** the "
            "stock to the strike into expiry.\n\n"
            "Desk arithmetic for one day: a hedged long option breaks even when the "
            "stock moves about $\\sigma S \\sqrt{1/252}$ — one implied daily standard "
            "deviation (252 trading days per year). Move more, and Γ earnings beat the "
            "Θ rent; move less, and the rent wins.")


[module_payoff_algebra, module_smile_density,
 module_spanning, module_inverse, module_greeks][MODULES.index(module)]()
