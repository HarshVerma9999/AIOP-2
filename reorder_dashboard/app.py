"""Inventory Control Room: Streamlit dashboard for the AI-enhanced reorder-point project (AI in Operations, slide 35).
Run:  streamlit run app.py      (multipage: only the open page is computed)"""
import hashlib, json, math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import engine as E
import theme as T

st.set_page_config(page_title="Inventory Control Room · Store 308", page_icon="📦", layout="wide", initial_sidebar_state="auto")
st.markdown(T.CSS, unsafe_allow_html=True)
DATA = Path(__file__).parent / "data"


@st.cache_resource(show_spinner=False)
def _clear_cache_for(code_hash):
    """Runs once per code version: a redeploy keeps cached results of functions whose own code didn't change, so drop them."""
    st.cache_data.clear()
    return code_hash


_clear_cache_for(hashlib.sha1(b"".join((Path(__file__).parent / f).read_bytes() for f in ("app.py", "engine.py", "data/meta.json"))).hexdigest())


# =============================== data ===============================
@st.cache_data(show_spinner=False)
def load():
    rd = lambda n: pd.read_parquet(DATA / f"{n}.parquet")
    d = {n: rd(n) for n in ["store_daily", "hourly_lost", "recovery_validation", "model_summary", "test_week_forecasts",
                            "forecast_errors", "next_week_forecast", "plan", "sensitivity", "live_log", "audit"]}
    g = np.load(DATA / "grid.npz"); d["grid"] = {k: g[k] for k in g.files}
    d["meta"] = json.loads((DATA / "meta.json").read_text())
    return d


D = load(); M = D["meta"]
SEED, HERO, STORE = M["seed"], M["hero"], M["store"]
PLAN = D["plan"].set_index("product")
NAIVE = D["store_daily"][D["store_daily"].t < 97].groupby("product_id").sale_amount.mean()
PRODUCTS = PLAN.sort_values("avg demand/day", ascending=False).index.tolist()
LT_PRESETS = {"Normal supplier (base case)": tuple(M["lt_probs"]), "Reliable (always 2 days)": (0.0, 1.0, 0.0, 0.0),
              "Unreliable (often 3–4 days)": (0.15, 0.30, 0.30, 0.25), "Disruption (4 days half the time)": (0.05, 0.20, 0.25, 0.50)}
BASE = dict(sku=HERO, lt_preset="Normal supplier (base case)", service=M["service_level"], h=M["h_cost"], p=M["p_cost"], w=M["w_cost"],
            k=M["k_cost"], shelf=M["shelf_life"], fill=M["min_fill"], waste=M["max_waste"], cap=M["capacity_days"])
POLICY_COLORS = {"Class calculator": T.AMBER, "Formula + EOQ": T.CORAL, "Monte Carlo + EOQ": T.VIOLET, "AI · Bayesian": T.TEAL}
TIPS = {
    "s": "Reorder point: when stock on the shelf plus stock already ordered falls to this level, place an order.",
    "Q": "Order quantity: how many units each order brings.",
    "wape": "Forecast error (WAPE): total absolute error divided by total demand. 0.32 means forecasts are off by 32% of demand on average.",
    "fill": "Share of customer demand actually served from the shelf (the rest is lost to empty shelves).",
    "waste": "Share of delivered stock that spoils before it is sold.",
    "ss": "Safety stock: extra units kept to cover demand and delivery-time surprises.",
    "csl": "The formula's service level is the chance of not running out before a delivery arrives. It is not the same as the share of demand served.",
}


def product_inputs(p):
    fc = D["next_week_forecast"].query("product_id == @p").sort_values("t").forecast.to_numpy()
    e = D["forecast_errors"].query("product_id == @p").error.to_numpy()
    return fc, e


# =============================== control panel ===============================
for k, v in BASE.items():
    st.session_state.setdefault(k, v)


def reset():
    for k, v in BASE.items():
        st.session_state[k] = v


with st.sidebar:
    st.markdown('<div class="hero-kicker" style="margin-top:4px">Control panel</div>', unsafe_allow_html=True)
    st.markdown("### Mission parameters")
    st.selectbox("Product (SKU)", PRODUCTS, key="sku", help="Every page follows this product.",
                 format_func=lambda p: f"SKU {p} · {PLAN.loc[p, 'avg demand/day']:.2f}/day" + ("  ★ notebook focus" if p == HERO else ""))
    with st.expander("Supplier lead time", expanded=True):
        st.radio("Delivery pattern", list(LT_PRESETS), key="lt_preset", label_visibility="collapsed")
        lp = LT_PRESETS[st.session_state.lt_preset]
        st.caption("Chance of a delivery taking 1 / 2 / 3 / 4 days: " + " / ".join(f"{x:.0%}" for x in lp))
    with st.expander("Service target & costs", expanded=False):
        st.slider("Service level for the formula", 0.80, 0.99, key="service", step=0.01, format="%.2f", help=TIPS["csl"])
        st.slider("Holding cost (per unit per day)", 0.01, 0.10, key="h", step=0.01)
        st.slider("Stock-out cost (per unit not served)", 0.25, 5.0, key="p", step=0.25)
        st.slider("Waste cost (per spoiled unit)", 0.0, 3.0, key="w", step=0.25)
        st.slider("Ordering cost (per order)", 0.1, 2.0, key="k", step=0.1)
        st.slider("Shelf life (days)", 2, 8, key="shelf")
    with st.expander("Business limits", expanded=False):
        st.slider("Minimum share of demand served", 0.85, 0.99, key="fill", step=0.01, format="%.2f", help=TIPS["fill"])
        st.slider("Maximum waste", 0.01, 0.15, key="waste", step=0.01, format="%.2f", help=TIPS["waste"])
        st.slider("Storage limit (days of demand)", 3, 10, key="cap")
    is_base = all(st.session_state[k] == v for k, v in BASE.items() if k != "sku")
    st.markdown(T.chip("NOTEBOOK SETTINGS", "ok") if is_base else T.chip("WHAT-IF SETTINGS", "warn"), unsafe_allow_html=True)
    st.button("Reset to notebook settings", on_click=reset, width="stretch")
    st.caption("Costs are in unit-cost equivalents (1.0 = the purchase cost of one unit). Quantities are in the dataset's scaled units.")

S = st.session_state
SKU, LT = int(S.sku), LT_PRESETS[S.lt_preset]
LT_MEAN, LT_SD = E.lead_time_stats(LT)
ARGS = (SKU, LT, S.service, S.h, S.p, S.w, S.k, int(S.shelf), S.fill, S.waste, S.cap)


# =============================== live calculations (cached) ===============================
def sim_kw(lt, h, pc, w, k, shelf):
    return dict(lt_probs=np.array(lt), shelf=int(shelf), p_cost=pc, h_cost=h, w_cost=w, k_cost=k)


@st.cache_data(show_spinner=False)
def live_safety(p, lt, service):
    fc, e = product_inputs(p); lm, ls = E.lead_time_stats(lt)
    f = E.safety_stock_formula(fc, e, lm, ls, service)
    mc = E.safety_stock_mc(fc, e, np.array(lt), service, seed=SEED)
    return f, {"SS": mc["SS"], "ROP": mc["ROP"]}, mc["ltd"]


@st.cache_data(show_spinner=False)
def live_optimise(p, lt, service, h, pc, w, k, shelf, fill, waste, cap):
    fc, e = product_inputs(p); f, _, _ = live_safety(p, lt, service)
    lim = dict(min_fill=fill, max_waste=waste, cap_days=cap)
    bo = E.optimise(fc, e, f["D_bar"], sim_kw=sim_kw(lt, h, pc, w, k, shelf), **lim, margin=M["opt_margin"], n_trials=40, seed=SEED)
    rs = E.optimise(fc, e, f["D_bar"], sim_kw=sim_kw(lt, h, pc, w, k, shelf), **lim, margin=M["opt_margin"], n_trials=40, seed=SEED, sampler="random")
    nb_args = (tuple(M["lt_probs"]), M["service_level"], M["h_cost"], M["p_cost"], M["w_cost"], M["k_cost"], M["shelf_life"],
               M["min_fill"], M["max_waste"], M["capacity_days"])
    if p == HERO and np.allclose(np.hstack([lt, (service, h, pc, w, k, shelf, fill, waste, cap)]), np.hstack([nb_args[0], nb_args[1:]])):
        # Notebook product at notebook settings: show the notebook's own answer. The Gaussian-process search can land a hair
        # differently on other library versions (e.g. Streamlit Cloud), so the saved result keeps the numbers identical.
        s_, q_ = M["bo_best"]["s"], M["bo_best"]["Q"]
        r = E.simulate(s_, q_, fc, e, **sim_kw(lt, h, pc, w, k, shelf))
        bo = {**bo, "s": s_, "Q": q_, "cost": r["cost"], "result": r, "feasible": E.feasible(r, f["D_bar"], fill, waste, cap), "pinned": True}
    return bo, rs


@st.cache_data(show_spinner=False)
def live_landscape(p, lt, service, h, pc, w, k, shelf, fill, waste, cap, n=24):
    fc, e = product_inputs(p); f, _, _ = live_safety(p, lt, service); d = f["D_bar"]; kw = sim_kw(lt, h, pc, w, k, shelf)
    s_grid = np.linspace(0, 8 * d, n); q_grid = np.linspace(0.25 * d, 6 * d, n)
    cost = np.full((n, n), np.nan); ok = np.zeros((n, n))
    for i, s_ in enumerate(s_grid):
        for j, q_ in enumerate(q_grid):
            r = E.simulate(s_, q_, fc, e, **kw); cost[j, i] = r["cost"]; ok[j, i] = E.feasible(r, d, fill, waste, cap)
    return s_grid, q_grid, cost, ok


def policy_settings(p, lt, service, h, pc, w, k, shelf, fill, waste, cap):
    f, mc, _ = live_safety(p, lt, service); bo, _ = live_optimise(p, lt, service, h, pc, w, k, shelf, fill, waste, cap)
    d = f["D_bar"]; naive = float(NAIVE[p])
    q_text = max(min(math.sqrt(2 * k * d / h), d * shelf), 0.25 * d)        # EOQ, capped at shelf life, at least a quarter day
    return {"Class calculator": (naive * 4, 2 * naive), "Formula + EOQ": (f["ROP"], q_text),
            "Monte Carlo + EOQ": (mc["ROP"], q_text), "AI · Bayesian": (bo["s"], bo["Q"])}, d


@st.cache_data(show_spinner=False)
def live_policies(p, lt, service, h, pc, w, k, shelf, fill, waste, cap):
    fc, e = product_inputs(p)
    pols, d = policy_settings(p, lt, service, h, pc, w, k, shelf, fill, waste, cap)
    rows = []
    for name, (s_, q_) in pols.items():
        r = E.simulate(s_, q_, fc, e, **sim_kw(lt, h, pc, w, k, shelf), seed=2024)        # 300 futures nobody was tuned on
        parts = r.pop("cost_parts")
        rows.append(dict(policy=name, s=s_, Q=q_, **r, **{f"part_{k_}": v for k_, v in parts.items()}, ok=E.feasible(r, d, fill, waste, cap)))
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def live_trajectories(p, lt, service, h, pc, w, k, shelf, fill, waste, cap):
    fc, e = product_inputs(p)
    pols, _ = policy_settings(p, lt, service, h, pc, w, k, shelf, fill, waste, cap)
    return {name: pd.DataFrame(E.trajectory(*pols[name], fc, e, lt_probs=np.array(lt), shelf=int(shelf)))
            for name in ["AI · Bayesian", "Formula + EOQ"]}


@st.cache_data(show_spinner=False)
def live_risk(p):
    fc, e = product_inputs(p); rows = []
    for name, probs in LT_PRESETS.items():
        lm, ls = E.lead_time_stats(probs)
        r = {"scenario": name.split(" (")[0], "avg lead time": lm}
        for sl in (0.90, 0.95, 0.99):
            r[f"MC {int(sl * 100)}"] = E.safety_stock_mc(fc, e, np.array(probs), sl, seed=SEED)["SS"]
            r[f"F {int(sl * 100)}"] = E.safety_stock_formula(fc, e, lm, ls, sl)["SS"]
        r["ROP 95"] = E.safety_stock_mc(fc, e, np.array(probs), 0.95, seed=SEED)["ROP"]
        rows.append(r)
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def live_plan(lt, service):
    return pd.DataFrame([dict(product=p, **{f"{k}_{m}": v[k] for m, v in zip(["formula", "mc"], live_safety(p, lt, service)[:2]) for k in ("SS", "ROP")})
                         for p in PRODUCTS]).set_index("product")


@st.cache_data(show_spinner=False)
def live_scenarios(p):
    fc, e = product_inputs(p); f, _, _ = live_safety(p, tuple(M["lt_probs"]), M["service_level"]); d = f["D_bar"]
    base = dict(lt=tuple(M["lt_probs"]), shelf=M["shelf_life"], pc=M["p_cost"], fill=M["min_fill"])
    rows = []
    for name, ch in [("Base case", {}), ("Reliable supplier (lead time always 2 days)", dict(lt=(0, 1.0, 0, 0))),
                     ("Unreliable supplier (often 3-4 days)", dict(lt=(0.15, 0.30, 0.30, 0.25))), ("Shorter shelf life (3 days)", dict(shelf=3)),
                     ("Longer shelf life (6 days)", dict(shelf=6)), ("Cheap stock-outs (cost 0.75)", dict(pc=0.75)),
                     ("Expensive stock-outs (cost 3.0)", dict(pc=3.0)), ("Higher service target (fill ≥ 98%)", dict(fill=0.98)),
                     ("Lower service target (fill ≥ 90%)", dict(fill=0.90))]:
        c = {**base, **ch}
        b = E.optimise(fc, e, d, sim_kw=sim_kw(c["lt"], M["h_cost"], c["pc"], M["w_cost"], M["k_cost"], c["shelf"]), min_fill=c["fill"],
                       max_waste=M["max_waste"], cap_days=M["capacity_days"], margin=M["opt_margin"], n_trials=50, seed=SEED)
        lm, ls = E.lead_time_stats(c["lt"])
        rows.append({"scenario": name, "optimal s": b["s"], "optimal Q": b["Q"], "daily cost": b["cost"], "fill rate": b["result"]["fill_rate"],
                     "waste": b["result"]["waste_rate"], "meets constraints": b["feasible"],
                     "formula ROP": E.safety_stock_formula(fc, e, lm, ls, c["fill"] if "fill" in ch else M["service_level"])["ROP"]})
    return pd.DataFrame(rows)


# =============================== shared bits ===============================
def fig_base(height=360, title=None):
    f = go.Figure(); f.update_layout(height=height, title=title); return f


def show(fig, key):
    st.plotly_chart(fig, width="stretch", key=f"chart_{key}", config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def header():
    st.markdown(f"""
<div class="hero">
  <div>
    <div class="hero-kicker">AI in Operations · Slide 35 · Reorder point & inventory</div>
    <div class="hero-title">Inventory Control Room</div>
    <div class="hero-sub">Fresh-grocery store <b>{STORE}</b> · {M['n_products']} products · FreshRetailNet-50K (Dingdong) ·
    now viewing <b>SKU {SKU}</b>. Personal data slice from the seed of <b>{M['student_id']}</b>.</div>
  </div>
  <div style="display:flex; gap:10px; flex-wrap:wrap">
    <span class="live"><span class="dot"></span>INTERACTIVE</span>
    <span class="live" style="border-color:rgba(163,175,196,.35);background:rgba(163,175,196,.06)">DATA · MAR–JUL 2024</span>
    <span class="live" style="border-color:rgba(255,181,71,.35);background:rgba(255,181,71,.06)"><span class="dot amber"></span>{'NOTEBOOK SETTINGS' if is_base else 'WHAT-IF'}</span>
  </div>
</div>""", unsafe_allow_html=True)


def pct(x):
    return f"{x:.1%}"


# =============================== pages ===============================
def page_overview():
    with st.spinner("Optimising and replaying the four policies on 300 futures…"):
        pol = live_policies(*ARGS)
    P = pol.set_index("policy"); ai, cl, fo = P.loc["AI · Bayesian"], P.loc["Class calculator"], P.loc["Formula + EOQ"]
    n_ok = int(pol.ok.sum())
    c = st.columns(4)
    c[0].markdown(T.kpi("Cost vs class calculator", f"{ai.cost / cl.cost - 1:+.0%}", f"{ai.cost:.3f} vs {cl.cost:.3f} per day · "
                        f"{ai.cost / fo.cost - 1:+.0%} vs textbook formula", chip=T.chip("AI POLICY", "ok" if ai.cost < cl.cost else "bad"),
                        tip="Average daily cost of holding stock, lost sales, waste and ordering, in unit-cost equivalents."), unsafe_allow_html=True)
    c[1].markdown(T.kpi("Food thrown away", pct(ai.waste_rate), f"textbook formula: {pct(fo.waste_rate)} · class: {pct(cl.waste_rate)}",
                        accent=T.NEON_AMBER, tip=TIPS["waste"]), unsafe_allow_html=True)
    c[2].markdown(T.kpi("Customers served", pct(ai.fill_rate), f"target ≥ {S.fill:.0%}", tip=TIPS["fill"]), unsafe_allow_html=True)
    c[3].markdown(T.kpi("Policies within all limits", f"{n_ok} of 4", "tested on 300 futures none was tuned on", accent="#8E7EF0",
                        chip=T.chip("ONLY THE AI POLICY", "ok") if (n_ok == 1 and ai.ok) else (T.chip("AI WITHIN LIMITS", "ok") if ai.ok else T.chip("AI CANNOT MEET ALL LIMITS", "bad"))),
                  unsafe_allow_html=True)
    ai_best = bool(ai.ok and ai.cost <= pol.cost.min() + 1e-12 and ai.waste_rate < fo.waste_rate)
    cost_txt = f'costs <b>{abs(ai.cost / cl.cost - 1):.0%} {"less" if ai.cost < cl.cost else "more"}</b> than the class calculator'
    if ai.ok:
        ratio = fo.waste_rate / ai.waste_rate if ai.waste_rate > 0 else float("inf")
        if 1.5 < ratio < 20:
            waste_txt = f'throws away <b>{ratio:.{1 if ratio < 3 else 0}f}× less food</b> than the textbook formula'
        elif ratio >= 20:
            waste_txt = f'throws away <b>{pct(ai.waste_rate)}</b> of its stock (textbook formula: {pct(fo.waste_rate)})'
        else:
            waste_txt = 'meets every limit'
        st.markdown(f'<div class="impact">For SKU {SKU}, the AI policy {waste_txt} and {cost_txt}, '
                    f'while still serving <b>{pct(ai.fill_rate)}</b> of customers.</div>', unsafe_allow_html=True)
    elif n_ok == 0:
        st.markdown('<div class="callout">Under these settings none of the four policies (including the optimiser\'s 40 tries) meets every limit for this '
                    'product. The figures show the closest one found. Relax a limit, or fix the cause (for example, a more reliable supplier).</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="callout">The optimiser found no policy within all limits in its 40 tries, but {n_ok} rule-based '
                    f'polic{"y does" if n_ok == 1 else "ies do"} pass. Use that one for this product, or give the optimiser more tries.</div>',
                    unsafe_allow_html=True)
    if not is_base:
        st.markdown('<div class="callout" style="margin-top:10px">What-if settings are active: every number on this page uses your control-panel settings.</div>',
                    unsafe_allow_html=True)

    st.markdown(T.section("01", "Why the AI policy wins" if ai_best else "How the four policies compare",
                          "Left: where each policy lands against the limits (the shaded box meets all of them). Right: what each policy's daily cost is made of."),
                unsafe_allow_html=True)
    a, b = st.columns([1, 1.1])
    with a:
        f = fig_base(400, "Customers served vs food thrown away")
        f.add_shape(type="rect", x0=0, x1=S.waste, y0=S.fill, y1=1.0, fillcolor="rgba(46,230,197,0.08)", line=dict(color=T.NEON_TEAL, width=1, dash="dot"), layer="below")
        f.add_annotation(x=S.waste / 2, y=S.fill, text="all limits met", showarrow=False, xanchor="center", yanchor="bottom", yshift=4, font=dict(color=T.NEON_TEAL, size=10.5, family="JetBrains Mono"))
        pos = {"Class calculator": "top center", "Formula + EOQ": "bottom center", "Monte Carlo + EOQ": "top center", "AI · Bayesian": "bottom center"}
        for _, r in pol.iterrows():
            f.add_scatter(x=[r.waste_rate], y=[r.fill_rate], mode="markers+text", text=[r.policy], textposition=pos[r.policy], textfont=dict(color=T.TEXT, size=11),
                          marker=dict(size=16 if "AI" in r.policy else 12, color=POLICY_COLORS[r.policy], line=dict(color=T.SURFACE, width=2)), name=r.policy,
                          hovertemplate=f"<b>{r.policy}</b><br>waste %{{x:.1%}} · served %{{y:.1%}}<br>cost {r.cost:.3f}/day<extra></extra>")
        lo = min(S.fill - 0.02, pol.fill_rate.min() - 0.01)
        f.update_xaxes(title="share of delivered stock that spoils", tickformat=".0%", range=[0, max(S.waste, pol.waste_rate.max()) * 1.2])
        f.update_yaxes(title="share of demand served", tickformat=".0%", range=[lo, 1.0]); f.update_layout(showlegend=False); show(f, "cmap")
    with b:
        f = fig_base(400, "Daily cost broken down (lower is better)")
        parts = [("ordering", "deliveries (ordering)", T.SKY), ("holding", "keeping stock (holding)", T.VIOLET),
                 ("lost_sales", "empty shelves (lost sales)", T.AMBER), ("waste", "spoiled food (waste)", T.CORAL)]
        order = pol.sort_values("cost").policy.tolist()
        Pp = pol.set_index("policy").loc[order]
        for key, label, col in parts:
            f.add_bar(y=order, x=Pp[f"part_{key}"], name=label, orientation="h", marker=dict(color=col, line=dict(color=T.SURFACE, width=2)),
                      hovertemplate=f"%{{y}}<br>{label}: %{{x:.3f}}/day<extra></extra>")
        f.add_scatter(y=order, x=Pp.cost * 1.02, mode="text", text=[f"{v:.3f} {'✓' if ok else '✕'}" for v, ok in zip(Pp.cost, Pp.ok)], textposition="middle right",
                      textfont=dict(color=T.TEXT, family="JetBrains Mono", size=12), hoverinfo="skip", showlegend=False)
        f.update_layout(barmode="stack", bargap=0.35, legend=dict(orientation="h", y=-0.22, yanchor="top"), margin=dict(b=90))
        f.update_xaxes(title="cost per day (✓ within all limits · ✕ breaks a limit)", range=[0, Pp.cost.max() * 1.32]); f.update_yaxes(autorange="reversed")
        show(f, "parts")
    top = Pp.loc[order[-1]]; worst_part = max(["ordering", "holding", "lost_sales", "waste"], key=lambda k_: top[f"part_{k_}"])
    st.markdown(f'<div class="callout teal">Reading the breakdown: the most expensive policy ({order[-1]}) loses most of its money to '
                f'<b>{worst_part.replace("_", " ")}</b>. '
                + (f'The AI policy pays for more frequent, smaller deliveries, and in return only {pct(ai.waste_rate)} of its stock spoils. ' if ai_best else '')
                + 'The reorder-point formula never sees spoilage or the cost of empty shelves; the optimiser balances all four costs together.</div>',
                unsafe_allow_html=True)

    st.markdown(T.section("02", "How the project gets there", "Each stage replaces a hand-typed number in the class calculator with something learned from data."),
                unsafe_allow_html=True)
    ms = D["model_summary"].set_index("model")
    st.markdown(f"""<div class="pipe">
      <div class="pipe-step"><div class="n">STAGE 0</div><div class="t">Recover true demand</div><div class="m">+{M['understatement']:.1%}</div>
        <div class="d">of demand was hidden by empty shelves (store-wide)</div></div>
      <div class="pipe-step"><div class="n">STEP 1</div><div class="t">Forecast demand</div><div class="m">{ms.loc[M['best_model'], 'FINAL TEST WAPE (new week)']:.0%} error</div>
        <div class="d">{M['best_model'].split(' (')[0]} on the unseen week, vs {ms.loc['Same weekday last week', 'FINAL TEST WAPE (new week)']:.0%} for "same day last week"</div></div>
      <div class="pipe-step"><div class="n">STEP 2</div><div class="t">Dynamic safety stock</div><div class="m">{PLAN['MC vs formula'].mean():+.1%}</div>
        <div class="d">more protection from Monte Carlo than the textbook formula, store average</div></div>
      <div class="pipe-step"><div class="n">STEP 3</div><div class="t">Bayesian optimisation</div><div class="m">40 runs</div>
        <div class="d">to find the best valid policy; a brute-force search needs 1,600</div></div>
    </div>""", unsafe_allow_html=True)
    st.dataframe(pol.assign(within=np.where(pol.ok, "✓ yes", "✕ no"))[["policy", "within", "cost", "fill_rate", "waste_rate", "s", "Q", "orders_per_week"]],
                 hide_index=True, width="stretch", column_config={
                     "within": st.column_config.TextColumn("within limits"), "cost": st.column_config.NumberColumn("cost/day", format="%.3f"),
                     "fill_rate": st.column_config.NumberColumn("served", format="percent"), "waste_rate": st.column_config.NumberColumn("waste", format="percent"),
                     "s": st.column_config.NumberColumn("reorder point", format="%.2f", help=TIPS["s"]),
                     "Q": st.column_config.NumberColumn("order qty", format="%.2f", help=TIPS["Q"]),
                     "orders_per_week": st.column_config.NumberColumn("orders/week", format="%.1f")})


def page_demand():
    sd = D["store_daily"]
    st.markdown(T.section("01", "Demand hidden by empty shelves",
                          f"When a shelf is empty, recorded sales understate what customers wanted. Across the store that gap is {M['understatement']:.1%}."),
                unsafe_allow_html=True)
    day = sd.groupby("dt")[["sale_amount", "demand"]].sum().reset_index()
    f = fig_base(330, "Store total per day: recorded sales vs recovered true demand")
    f.add_scatter(x=day.dt, y=day.sale_amount, name="recorded sales", line=dict(color=T.AMBER, width=2), hovertemplate="sales %{y:.1f}<extra></extra>")
    f.add_scatter(x=day.dt, y=day.demand, name="true demand (recovered)", line=dict(color=T.TEAL, width=2.5), fill="tonexty",
                  fillcolor="rgba(46,230,197,0.10)", hovertemplate="true demand %{y:.1f}<extra></extra>")
    f.add_vline(x=pd.Timestamp(M["new_week"][0]).timestamp() * 1000, line=dict(color=T.MUTED, dash="dot"))
    f.add_annotation(x=pd.Timestamp(M["new_week"][0]), y=day.demand.max(), text="unseen test week", showarrow=False, xanchor="left", font=dict(color=T.TEXT_2, size=11))
    f.update_layout(hovermode="x unified"); f.update_yaxes(title="units (scaled)"); show(f, "timeline")

    hl = D["hourly_lost"]; late = hl.loc[hl.hour >= 16, "lost"].sum() / hl.lost.sum()
    a, b = st.columns([1.5, 1])
    with a:
        f = fig_base(340, "When sales are lost: demand turned away by empty shelves, by hour")
        f.add_bar(x=hl.hour, y=hl.lost, marker=dict(color=[T.CORAL if h >= 16 else "#65739C" for h in hl.hour], cornerradius=4),
                  customdata=np.c_[hl.lost_rate, hl.stockout_rate], hovertemplate="%{x}:00<br>lost demand %{y:.0f} units<br>"
                  "%{customdata[0]:.0%} of that hour's demand lost<br>%{customdata[1]:.0%} of products out of stock<extra></extra>")
        f.update_xaxes(title="hour of day", dtick=2); f.update_yaxes(title="units of demand lost (whole period)"); f.update_layout(showlegend=False); show(f, "lost")
    with b:
        st.markdown(T.kpi("Lost after 4 pm", f"{late:.0%}", "of all demand turned away by empty shelves happens in the late afternoon and evening",
                          accent=T.CRIT), unsafe_allow_html=True)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="callout">Action for the store: shelves run empty exactly when evening shoppers arrive. '
                    'A mid-afternoon top-up delivery, or a higher reorder point for evening-heavy products, would recover most of these lost sales.</div>',
                    unsafe_allow_html=True)

    v = D["recovery_validation"].set_index("method")
    st.markdown(T.section("02", "Can we trust the recovered demand?",
                          "Tested on fully stocked days where the truth is known: hide the evening, then try to recover the day's total."), unsafe_allow_html=True)
    c = st.columns(3)
    c[0].markdown(T.kpi("Recovery error", f"{v.loc['Hourly-profile recovery', 'error (WAPE)']:.1%}",
                        f"bias {v.loc['Hourly-profile recovery', 'bias']:+.1%}: almost none", tip=TIPS["wape"]), unsafe_allow_html=True)
    c[1].markdown(T.kpi("If sales were taken as demand", f"{v.loc['Treat sales as demand', 'error (WAPE)']:.1%}",
                        f"bias {v.loc['Treat sales as demand', 'bias']:+.1%}: always too low", accent=T.CRIT), unsafe_allow_html=True)
    c[2].markdown(T.kpi("Days too empty to estimate", f"{M['imputed_share']:.1%}", "filled from the product's previous two weeks", accent=T.NEON_AMBER),
                  unsafe_allow_html=True)

    st.markdown(T.section("03", "Step 1 · Forecasting: which model, and how sure is it?",
                          "Six models trained only on the past and scored on weeks they never saw (lower error is better)."), unsafe_allow_html=True)
    ms = D["model_summary"].sort_values("FINAL TEST WAPE (new week)")
    a, b = st.columns([1, 1.15])
    with a:
        f = fig_base(380, "Forecast error by model")
        f.add_bar(y=ms.model, x=ms["Validation WAPE (days 69-89)"], name="validation weeks", orientation="h", marker=dict(color=T.VIOLET, cornerradius=4),
                  hovertemplate="%{y}<br>validation %{x:.1%}<extra></extra>")
        f.add_bar(y=ms.model, x=ms["FINAL TEST WAPE (new week)"], name="final unseen week", orientation="h", marker=dict(color=T.TEAL, cornerradius=4),
                  hovertemplate="%{y}<br>final week %{x:.1%}<extra></extra>")
        f.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08); f.update_xaxes(tickformat=".0%", title="error (WAPE)"); f.update_yaxes(autorange="reversed")
        show(f, "models")
    with b:
        tw = D["test_week_forecasts"].query("product_id == @SKU").sort_values("t")
        ev = D["forecast_errors"].query("product_id == @SKU and t < 90").error.to_numpy()      # errors from earlier weeks only (honest band)
        lo_q, hi_q = np.quantile(ev, [0.10, 0.90]); fc_ = tw[M["best_model"]].to_numpy()
        lo, hi = np.maximum(fc_ + lo_q, 0), fc_ + hi_q
        inside = int(((tw.demand.to_numpy() >= lo) & (tw.demand.to_numpy() <= hi)).sum())
        f = fig_base(380, f"SKU {SKU}: forecast with its 80% range vs what happened")
        f.add_scatter(x=tw.dt, y=hi, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False)
        f.add_scatter(x=tw.dt, y=lo, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(46,230,197,0.16)", name="80% range", hoverinfo="skip")
        T.glow_line(f, tw.dt, fc_, T.NEON_TEAL, "forecast", hover="%{y:.2f}")
        f.add_scatter(x=tw.dt, y=tw.demand, mode="markers", name="actual demand", marker=dict(size=11, color=T.NEON_AMBER, symbol="diamond", line=dict(color=T.SURFACE, width=2)),
                      hovertemplate="actual %{y:.2f}<extra></extra>")
        f.update_layout(hovermode="x unified", legend=dict(orientation="h", y=-0.2, yanchor="top"), margin=dict(b=80)); f.update_yaxes(title="units")
        show(f, "fan")
        st.caption(f"{inside} of 7 days fell inside the range, which was built only from errors on earlier weeks. "
                   "The width of this band is what Step 2 turns into safety stock.")
    lstm = ms.set_index("model").loc["LSTM (deep learning)"]
    st.markdown(f'<div class="callout teal">Gradient Boosting, one model learned across 2,499 product histories, wins. The LSTM forecast '
                f'{abs(lstm["Final test bias"]):.0%} too low on the final week: 90 days is short for deep learning.</div>', unsafe_allow_html=True)


def page_safety():
    f_l, mc_l, ltd = live_safety(SKU, LT, S.service)
    st.markdown(T.section("01", f"SKU {SKU} · how much safety stock?",
                          "Recomputed live from the control panel: change the supplier pattern or the service level and every number updates."), unsafe_allow_html=True)
    c = st.columns(5)
    c[0].markdown(T.kpi("Avg demand / day", f"{f_l['D_bar']:.2f}", "forecast, bias-corrected"), unsafe_allow_html=True)
    c[1].markdown(T.kpi("Forecast error sd", f"{f_l['sigma_D']:.2f}", "σ_D from real past errors", accent="#8E7EF0",
                        tip="Standard deviation of the forecast's real past errors. This is the demand uncertainty the formula uses."), unsafe_allow_html=True)
    c[2].markdown(T.kpi("Formula reorder point", f"{f_l['ROP']:.2f}", f"safety stock {f_l['SS']:.2f}", accent=T.NEON_AMBER, tip=TIPS["ss"]), unsafe_allow_html=True)
    c[3].markdown(T.kpi("Monte Carlo reorder point", f"{mc_l['ROP']:.2f}", f"safety stock {mc_l['SS']:.2f}",
                        tip="Read straight off 20,000 simulated futures: the level that covers the chosen share of them."), unsafe_allow_html=True)
    c[4].markdown(T.kpi("Class calculator", f"{NAIVE[SKU] * 4:.2f}", "sales avg × 2 days + 2 days", accent=T.CRIT), unsafe_allow_html=True)
    a, b = st.columns([1.35, 1])
    with a:
        f = fig_base(380, f"Demand during the lead time: 20,000 simulated futures (service {S.service:.0%})")
        f.add_histogram(x=ltd, nbinsx=70, marker=dict(color="rgba(8,170,144,0.55)", line=dict(color=T.SURFACE, width=1)),
                        hovertemplate="%{x:.2f} units → %{y} futures<extra></extra>")
        for x, col, lab, ypos in [(f_l["ROP"], T.NEON_AMBER, "formula", 1.0), (mc_l["ROP"], T.NEON_TEAL, "Monte Carlo", 0.9), (NAIVE[SKU] * 4, T.CRIT, "class", 0.8)]:
            f.add_vline(x=x, line=dict(color=col, width=2, dash="dash" if lab == "class" else "solid"))
            f.add_annotation(x=x, y=ypos, yref="paper", text=f"{lab} {x:.2f}", showarrow=False, xanchor="left", xshift=4,
                             font=dict(color=T.TEXT, size=11, family="JetBrains Mono"), bgcolor="rgba(14,21,40,0.85)")
        cover = (ltd <= NAIVE[SKU] * 4).mean()
        f.update_xaxes(title="units needed before the delivery arrives"); f.update_yaxes(title="futures"); f.update_layout(showlegend=False, bargap=0.02)
        show(f, "ltd")
        st.caption(f"The class calculator's reorder point would cover {'only ' if cover < S.service else ''}{cover:.0%} of these futures.")
    with b:
        dem_part = f_l["sigma_D"] ** 2 * LT_MEAN; lt_part = f_l["D_bar"] ** 2 * LT_SD ** 2; tot = dem_part + lt_part
        f = fig_base(380, "What drives the safety stock")
        f.add_bar(x=["demand surprises", "delivery-time surprises"], y=[dem_part / tot if tot else 0, lt_part / tot if tot else 0],
                  marker=dict(color=[T.TEAL, T.AMBER], cornerradius=4), text=[f"{dem_part / tot:.0%}" if tot else "–", f"{lt_part / tot:.0%}" if tot else "–"],
                  textposition="outside", textfont=dict(color=T.TEXT, family="JetBrains Mono"), hovertemplate="%{x}: %{y:.0%}<extra></extra>")
        f.update_yaxes(title="share of the uncertainty", tickformat=".0%", range=[0, 1.12]); f.update_layout(showlegend=False); show(f, "drivers")
        lead = "delivery reliability" if lt_part > dem_part else "forecast accuracy"
        st.caption(f"For this product, improving {lead} would cut safety stock the most. Formula: SS = Z·√(σ_D²·L + D_avg²·σ_L²), "
                   f"Z = {f_l['Z']:.2f}, L = {LT_MEAN:.2f} days, σ_L = {LT_SD:.2f}.")

    st.markdown(T.section("02", f"Risk scenarios for SKU {SKU}",
                          "Monte Carlo safety stock needed for each supplier risk (rows) and service level (columns). The formula's answer is in the hover."),
                unsafe_allow_html=True)
    rk = live_risk(SKU)
    z = rk[["MC 90", "MC 95", "MC 99"]].to_numpy(); fz = rk[["F 90", "F 95", "F 99"]].to_numpy()
    f = fig_base(300)
    f.add_heatmap(z=z, x=["90% service", "95% service", "99% service"], y=rk.scenario, colorscale=T.SEQ_TEAL, xgap=3, ygap=3, showscale=False,
                  text=[[f"{v:.2f}" for v in row] for row in z], texttemplate="%{text}", textfont=dict(family="JetBrains Mono", size=15),     # colour auto-contrasts with each cell
                  customdata=fz, hovertemplate="%{y} · %{x}<br>Monte Carlo %{z:.2f}<br>formula %{customdata:.2f}<extra></extra>")
    f.update_yaxes(autorange="reversed"); f.update_layout(margin=dict(t=20)); show(f, "risk")
    hi_rop, hi_ss = rk.loc[rk["ROP 95"].idxmax(), "scenario"], rk.loc[rk["MC 95"].idxmax(), "scenario"]
    why = ("" if hi_rop == hi_ss else
           f" They differ because the reorder point also covers the expected demand over the lead time, which is longest under '{hi_rop}', "
           f"while safety stock only covers the uncertainty, which is largest under '{hi_ss}'.")
    st.caption("Reorder point at 95%: " + " · ".join(f"{r.scenario} {r['ROP 95']:.2f}" for _, r in rk.iterrows()) +
               f". Highest reorder point: {hi_rop}; most safety stock: {hi_ss}." + why)

    st.markdown(T.section("03", "Where the textbook formula under-protects most",
                          "Products ranked by how much more stock Monte Carlo says is needed (notebook settings). These have the most uneven demand."),
                unsafe_allow_html=True)
    gap = PLAN["MC vs formula"].sort_values(ascending=False).head(15)
    f = fig_base(380)
    f.add_bar(x=[f"SKU {p}" for p in gap.index], y=gap.values, marker=dict(color=[T.NEON_AMBER if p == SKU else T.VIOLET for p in gap.index], cornerradius=4),
              hovertemplate="%{x}<br>Monte Carlo needs %{y:+.0%} vs formula<extra></extra>")
    f.update_yaxes(title="extra reorder point vs formula", tickformat="+.0%"); f.update_layout(showlegend=False, margin=dict(t=20)); show(f, "gap")


def page_optimiser():
    st.markdown(T.section("01", f"Bayesian optimisation for SKU {SKU}, live",
                          "A simulator copies the shelf (random demand, random deliveries, stock spoiling after the shelf life). A Gaussian-process model "
                          "learns where the cheapest policy that meets every limit lies, and tests only the most promising settings."), unsafe_allow_html=True)
    with st.spinner("Running Bayesian optimisation (40 simulations)…"):
        bo, rs = live_optimise(*ARGS)
    use_pre = is_base and SKU == HERO
    with st.spinner("Mapping the cost of every policy…"):
        if use_pre:
            G = D["grid"]; s_grid, q_grid, cost_map, ok_map = G["s"], G["q"], G["cost"], G["ok"]
        else:
            s_grid, q_grid, cost_map, ok_map = live_landscape(*ARGS)
    r = bo["result"]
    with st.spinner("Testing the policies on new random futures…"):
        oos = live_policies(*ARGS).set_index("policy").loc["AI · Bayesian"]
    c = st.columns(5)
    c[0].markdown(T.kpi("Reorder point s", f"{bo['s']:.2f}", "order when stock falls to this", tip=TIPS["s"]), unsafe_allow_html=True)
    c[1].markdown(T.kpi("Order quantity Q", f"{bo['Q']:.2f}", "units per order", accent="#8E7EF0", tip=TIPS["Q"]), unsafe_allow_html=True)
    c[2].markdown(T.kpi("Daily cost", f"{bo['cost']:.3f}", f"while tuning · on new futures {oos.cost:.3f}", accent=T.NEON_AMBER), unsafe_allow_html=True)
    c[3].markdown(T.kpi("Customers served", pct(r["fill_rate"]), f"while tuning · on new futures {pct(oos.fill_rate)} (waste {pct(oos.waste_rate)})", tip=TIPS["fill"]),
                  unsafe_allow_html=True)
    c[4].markdown(T.kpi("Status", "PASS" if bo["feasible"] else "FAIL", "meets every limit" if bo["feasible"] else "closest policy shown",
                        accent=T.NEON_TEAL if bo["feasible"] else T.CRIT,
                        chip=T.chip("WITHIN ALL LIMITS", "ok") if bo["feasible"] else T.chip("NO POLICY MEETS ALL LIMITS" if not ok_map.any() else "NONE FOUND IN 40 TRIES", "bad")), unsafe_allow_html=True)

    a, b = st.columns([1.25, 1])
    with a:
        f = fig_base(440, "Cost of every policy (red outline = meets all limits)")
        f.add_heatmap(x=s_grid, y=q_grid, z=cost_map, colorscale=[[0, "#2EE6C5"], [0.35, "#109B88"], [0.7, "#0B3940"], [1, "#0E1528"]],
                      zmax=float(np.nanpercentile(cost_map, 90)), colorbar=dict(title=dict(text="cost/day"), outlinewidth=0),
                      hovertemplate="s %{x:.2f} · Q %{y:.2f}<br>cost %{z:.3f}<extra></extra>")
        if ok_map.any():
            f.add_contour(x=s_grid, y=q_grid, z=ok_map, contours=dict(start=0.5, end=0.5, size=1, coloring="none"), line=dict(color="#FF5C7A", width=2.5),
                          showscale=False, hoverinfo="skip", name="meets all limits")
        tr = pd.DataFrame(bo["trials"])
        f.add_scatter(x=tr.s, y=tr.Q, mode="markers", name="settings the optimiser tried", text=tr.n,
                      marker=dict(size=8, color="rgba(255,255,255,0.85)", line=dict(color=T.SURFACE, width=1)), hovertemplate="try %{text}<br>s %{x:.2f} · Q %{y:.2f}<extra></extra>")
        f.add_scatter(x=[bo["s"]], y=[bo["Q"]], mode="markers", name="best policy (notebook run)" if bo.get("pinned") else "best policy", marker=dict(symbol="star", size=20, color=T.NEON_AMBER, line=dict(color=T.SURFACE, width=2)),
                      hovertemplate="best: s %{x:.2f} · Q %{y:.2f}<extra></extra>")
        f.update_xaxes(title="reorder point s"); f.update_yaxes(title="order quantity Q"); show(f, "land")
    with b:
        st.markdown(T.section("", "Why a smart search", f"Only {ok_map.mean():.1%} of possible policies meet all three limits, so blind searching wastes runs."),
                    unsafe_allow_html=True)
        grid_ok = ok_map > 0
        grid_best = float(np.nanmin(np.where(grid_ok, cost_map, np.nan))) if grid_ok.any() else None
        n_grid = "1,600" if use_pre else "576"
        methods = [("Bayesian optimisation", "40 runs", f"{bo['cost']:.3f}" if bo["feasible"] else "none valid", bo["feasible"], T.NEON_TEAL),
                   ("Random search", "40 runs", f"{rs['cost']:.3f}" if rs["feasible"] else "none valid", rs["feasible"], T.CORAL),
                   ("Brute-force grid", f"{n_grid} runs", f"{grid_best:.3f}" if grid_best is not None else "none valid", grid_best is not None, T.NEON_AMBER)]
        for name, runs, best, ok, col in methods:
            st.markdown(f'<div class="glass method" style="--accent:{col}; margin-bottom:10px"><div class="kpi-label">{name} · {runs}</div>'
                        f'<div class="big">{best}</div><div class="kpi-sub">{"best valid daily cost" if ok else "found no policy that meets every limit"}</div></div>',
                        unsafe_allow_html=True)

    st.markdown(T.section("02", "The policy in action: 4 weeks on the shelf",
                          "One example future, same demand and deliveries for both policies. Lines show stock on the shelf through each day; "
                          "▲ delivery arrives · ✕ shelf ran empty · ● food spoiled. The dashed line is the reorder point."), unsafe_allow_html=True)
    with st.spinner("Replaying four weeks…"):
        trj = live_trajectories(*ARGS)
    names = ["AI · Bayesian", "Formula + EOQ"]
    titles = []
    plural = lambda n, w: f"{n} {w}{'' if n == 1 else 's'}"
    for nme in names:
        t_ = trj[nme]
        titles.append(f"{nme}: {plural(int((t_.lost > 0.01).sum()), 'empty-shelf day')} · {t_.spoiled.sum():.1f} units spoiled · "
                      f"{plural(int((t_.ordered > 0).sum()), 'order')}")
    f = make_subplots(rows=2, cols=1, shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.12, subplot_titles=titles)
    pols, _ = policy_settings(*ARGS)
    for i, nme in enumerate(names, start=1):
        t_ = trj[nme]; col = POLICY_COLORS[nme]
        xs, ys = [], []
        for _, rw in t_.iterrows():
            xs += [rw.day, rw.day + 0.92]; ys += [rw.morning, rw.evening]
        f.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=col, width=2.5), fill="tozeroy", fillcolor="rgba(46,230,197,0.07)" if i == 1 else "rgba(234,88,122,0.07)",
                      name=nme, hovertemplate="day %{x:.0f}<br>stock %{y:.2f}<extra></extra>", row=i, col=1)
        f.add_hline(y=pols[nme][0], line=dict(color=T.MUTED, dash="dash", width=1), row=i, col=1)
        d_ = t_[t_.delivered > 0]; f.add_scatter(x=d_.day, y=d_.morning, mode="markers", marker=dict(symbol="triangle-up", size=10, color=T.NEON_TEAL),
                                                 name="delivery", showlegend=(i == 1), hovertemplate="delivery %{customdata:.2f} units<extra></extra>", customdata=d_.delivered, row=i, col=1)
        l_ = t_[t_.lost > 0.01]; f.add_scatter(x=l_.day + 0.92, y=np.zeros(len(l_)), mode="markers", marker=dict(symbol="x", size=11, color=T.CRIT),
                                               name="shelf empty", showlegend=(i == 1), customdata=l_.lost, hovertemplate="empty shelf: %{customdata:.2f} units lost<extra></extra>", row=i, col=1)
        w_ = t_[t_.spoiled > 0.01]; f.add_scatter(x=w_.day + 0.92, y=w_.evening, mode="markers", marker=dict(symbol="circle", size=10, color=T.NEON_AMBER),
                                                  name="spoiled", showlegend=(i == 1), customdata=w_.spoiled, hovertemplate="spoiled %{customdata:.2f} units<extra></extra>", row=i, col=1)
    f.update_layout(height=580, template="midnight", legend=dict(orientation="h", y=-0.13, yanchor="top"), margin=dict(t=60, b=95))
    f.update_annotations(font=dict(color=T.TEXT, size=13, family="Space Grotesk"))
    ymax = max(t_.morning.max() for t_ in trj.values()) * 1.1
    f.update_xaxes(title_text="day", row=2, col=1, dtick=2); f.update_yaxes(title_text="units on shelf", range=[0, ymax])   # same scale for both
    show(f, "saw")


def page_scenarios():
    st.markdown(T.section("01", f"How the best decision moves (SKU {SKU}, notebook settings)",
                          "Each row is a full Bayesian optimisation under different business conditions. The formula cannot react to shelf life or costs at all."),
                unsafe_allow_html=True)
    if SKU == HERO:
        se = D["sensitivity"]
    elif st.session_state.get(f"scen_{SKU}"):
        with st.spinner("Running 9 optimisations…"):
            se = live_scenarios(SKU)
    else:
        st.markdown(f'<div class="callout teal">Scenarios for SKU {HERO} come from the notebook. For SKU {SKU} they are computed on request '
                    '(9 optimisations, about 20–40 seconds).</div>', unsafe_allow_html=True)
        if st.button(f"Run the 9 scenarios for SKU {SKU}", type="primary"):
            st.session_state[f"scen_{SKU}"] = True; st.rerun()
        se = None
    if se is not None:
        okm = se["meets constraints"].astype(bool).to_numpy(); yy = list(se.scenario)
        f = fig_base(440)
        f.add_bar(y=yy, x=se["formula ROP"], orientation="h", name="formula reorder point", marker=dict(color="#3A4766", cornerradius=4),
                  hovertemplate="%{y}<br>formula %{x:.2f}<extra></extra>")
        f.add_bar(y=[y for y, o in zip(yy, okm) if o], x=se.loc[okm, "optimal s"], orientation="h", name="optimal reorder point (Bayesian)",
                  marker=dict(color=T.TEAL, cornerradius=4), hovertemplate="%{y}<br>optimal %{x:.2f}<extra></extra>")
        for y in [y for y, o in zip(yy, okm) if not o]:
            f.add_annotation(y=y, x=0.05, text="✕ no policy can meet all limits", showarrow=False, xanchor="left", yshift=-9,
                             font=dict(color="#FF8FA3", size=11, family="JetBrains Mono"))
        f.update_layout(barmode="group", bargap=0.3); f.update_yaxes(autorange="reversed"); f.update_xaxes(title="reorder point"); show(f, "sens")
        sv = se.set_index("scenario")["optimal s"]; base_s = float(sv.iloc[0])
        so = sv[[i for i in sv.index if "stock-outs" in i]]
        moves = (so.max() - so.min()) / max(base_s, 1e-9) if len(so) else 0.0
        st.caption((f"The optimal reorder point barely moves with the stock-out cost ({so.min():.2f} to {so.max():.2f}), because the service limit already "
                    "decides how much protection is needed. " if moves < 0.10 else
                    f"The optimal reorder point moves with the stock-out cost ({so.min():.2f} to {so.max():.2f}): for this product the cost of empty "
                    "shelves, not just the service limit, decides how much protection is worth holding. ")
                   + ("Where no policy can meet every limit, the business must relax one or fix the cause." if (~okm).any() else ""))
        with st.expander("Scenario table"):
            st.dataframe(se[["scenario", "optimal s", "optimal Q", "daily cost", "fill rate", "waste", "meets constraints", "formula ROP"]].round(3),
                         hide_index=True, width="stretch")

    ll = D["live_log"]
    st.markdown(T.section("02", f"The final week, replayed day by day (SKU {HERO})",
                          "Each morning yesterday's raw data arrives, the forecaster re-trains, every earlier forecast is scored and the plan is re-optimised. "
                          "This needs a full re-train per day, so it was run in the notebook for the focus product."), unsafe_allow_html=True)
    day_sel = st.select_slider("Replay day", options=list(ll.date), value=list(ll.date)[-1], key="replay")
    i = list(ll.date).index(day_sel); row = ll.iloc[i]; prev = ll.iloc[i - 1] if i > 0 else row
    c = st.columns(4)
    c[0].markdown(T.kpi("Forecast, next 7 days", f"{row['forecast next 7d (avg)']:.2f}", f"{row['forecast next 7d (avg)'] - prev['forecast next 7d (avg)']:+.2f} vs previous day"),
                  unsafe_allow_html=True)
    c[1].markdown(T.kpi("Monte Carlo reorder point", f"{row['Monte Carlo ROP']:.2f}", f"{row['Monte Carlo ROP'] - prev['Monte Carlo ROP']:+.2f} vs previous day", accent="#8E7EF0"),
                  unsafe_allow_html=True)
    c[2].markdown(T.kpi("AI reorder point s", f"{row['Bayesian s']:.2f}", f"order {row['Bayesian Q']:.2f} units", accent=T.NEON_AMBER, tip=TIPS["s"]), unsafe_allow_html=True)
    c[3].markdown(T.kpi("Forecast errors learned", f"{int(row['errors known'])}", "grows as each day is scored",
                        chip=T.chip("WITHIN ALL LIMITS", "ok") if row["meets constraints"] else T.chip("FLAGGED FOR A MANAGER", "bad")), unsafe_allow_html=True)
    f = fig_base(340, "Reorder point re-optimised every morning")
    x = np.arange(len(ll))
    f.add_bar(x=x, y=ll["actual demand that day"], name="actual demand that day", marker=dict(color="rgba(163,175,196,0.25)", cornerradius=4),
              hovertemplate="demand %{y:.2f}<extra></extra>")
    T.glow_line(f, x, ll["Bayesian s"], T.NEON_TEAL, "AI reorder point s")
    f.add_scatter(x=x, y=ll["Monte Carlo ROP"], name="Monte Carlo ROP", line=dict(color=T.VIOLET, width=2), marker=dict(size=8), mode="lines+markers",
                  hovertemplate="MC %{y:.2f}<extra></extra>")
    f.add_scatter(x=x, y=ll["formula ROP"], name="formula ROP", line=dict(color=T.AMBER, width=2), marker=dict(size=8), mode="lines+markers",
                  hovertemplate="formula %{y:.2f}<extra></extra>")
    f.add_vrect(x0=i - 0.45, x1=i + 0.45, fillcolor="rgba(46,230,197,0.08)", line_width=0)
    f.update_xaxes(tickvals=list(x), ticktext=[str(d_) for d_ in ll.date]); f.update_yaxes(title="units"); f.update_layout(hovermode="x unified"); show(f, "replay")
    st.caption("The plan stays stable from day to day: it adapts to new data without over-reacting.")


def page_plan():
    st.markdown(T.section("01", "Next week's replenishment plan, all products",
                          "Forecast, safety stock and reorder point for every product. In what-if mode the formula and Monte Carlo columns use your settings."),
                unsafe_allow_html=True)
    pl = PLAN.copy()
    if not is_base:
        with st.spinner("Recomputing the plan for every product…"):
            lv = live_plan(LT, S.service)
        pl["SS formula"], pl["ROP formula"], pl["SS Monte Carlo"], pl["ROP Monte Carlo"] = lv.SS_formula, lv.ROP_formula, lv.SS_mc, lv.ROP_mc
        st.markdown('<div class="callout">What-if mode: formula and Monte Carlo columns use your settings. The AI columns show the notebook result; '
                    're-optimise any product on the Optimiser page.</div>', unsafe_allow_html=True)
    c = st.columns(4)
    c[0].markdown(T.kpi("Products planned", f"{len(pl)}", f"store {STORE}"), unsafe_allow_html=True)
    c[1].markdown(T.kpi("Total safety stock · formula", f"{pl['SS formula'].sum():.1f}", "units (scaled)", accent=T.NEON_AMBER, tip=TIPS["ss"]), unsafe_allow_html=True)
    c[2].markdown(T.kpi("Total safety stock · Monte Carlo", f"{pl['SS Monte Carlo'].sum():.1f}", f"{pl['SS Monte Carlo'].sum() / pl['SS formula'].sum() - 1:+.0%} vs formula"),
                  unsafe_allow_html=True)
    nb = int(pl["Bayesian s"].notna().sum())
    c[3].markdown(T.kpi("AI-optimised products", f"{nb}", "the highest-volume products", accent="#8E7EF0",
                        chip=T.chip(f"{int(pl['Bayesian meets constraints'].fillna(False).astype(bool).sum())} / {nb} WITHIN LIMITS", "ok")), unsafe_allow_html=True)
    top = pl.sort_values("avg demand/day", ascending=False).head(15)
    f = fig_base(360, "Top 15 products by demand: reorder points")
    xs = [f"SKU {p}" for p in top.index]
    f.add_bar(x=xs, y=top["ROP formula"], name="formula", marker=dict(color=T.AMBER, cornerradius=4), hovertemplate="%{x}<br>formula %{y:.2f}<extra></extra>")
    f.add_bar(x=xs, y=top["ROP Monte Carlo"], name="Monte Carlo", marker=dict(color=T.VIOLET, cornerradius=4), hovertemplate="%{x}<br>MC %{y:.2f}<extra></extra>")
    f.add_bar(x=xs, y=top["Bayesian s"], name="AI · Bayesian", marker=dict(color=T.TEAL, cornerradius=4), hovertemplate="%{x}<br>AI %{y:.2f}<extra></extra>")
    f.update_layout(barmode="group", bargap=0.2, bargroupgap=0.06); f.update_yaxes(title="reorder point (units)"); show(f, "plan")
    opt_ = pl[pl["Bayesian s"].notna()]
    n_low = int(((opt_["Bayesian s"] < opt_["ROP formula"]) & (opt_["Bayesian s"] < opt_["ROP Monte Carlo"])).sum())
    st.caption(f"The AI reorder point is lower than both formulas for {n_low} of {len(opt_)} optimised products: it orders smaller amounts more often, "
               "which keeps fresh stock from spoiling." + ("" if is_base else " AI values come from the notebook settings; the formula bars use yours."))
    q = st.text_input("Search a product (SKU number)", key="plan_q", placeholder="e.g. 201")
    view = pl.reset_index()
    if q.strip():
        view = view[view["product"].astype(str).str.contains(q.strip(), regex=False)]
    view = view.assign(**{"AI s": view["Bayesian s"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—"),
                          "AI Q": view["Bayesian Q"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—"),
                          "AI within limits": view["Bayesian meets constraints"].map(lambda v: "—" if pd.isna(v) else ("✓" if bool(v) else "✕"))})
    cols = ["product", "avg demand/day", "forecast error sd", "SS formula", "ROP formula", "SS Monte Carlo", "ROP Monte Carlo", "class ROP", "AI s", "AI Q", "AI within limits"]
    st.dataframe(view[cols], hide_index=True, width="stretch", height=420, column_config={
        "product": st.column_config.NumberColumn("SKU", format="%d"),
        "avg demand/day": st.column_config.ProgressColumn("avg demand/day", min_value=0, max_value=float(pl["avg demand/day"].max()), format="%.2f"),
        **{c_: st.column_config.NumberColumn(c_, format="%.2f") for c_ in cols[2:8]}})
    out = pl.reset_index()[["product", "avg demand/day", "forecast error sd", "SS formula", "ROP formula", "SS Monte Carlo", "ROP Monte Carlo", "class ROP",
                           "Bayesian s", "Bayesian Q", "Bayesian daily cost", "Bayesian meets constraints"]]
    st.download_button("⬇  Download plan (CSV)", out.round(4).to_csv(index=False).encode(), file_name=f"replenishment_plan_store{STORE}.csv", mime="text/csv")


def page_about():
    st.markdown(T.section("01", "What this is"), unsafe_allow_html=True)
    st.markdown("""
This dashboard presents the **AI in Operations, slide 35** exercise: *enhance the reorder-point calculator and sensitivity analysis with predictive intelligence,
adaptive decision-making and real-time optimisation*. Results come from the project notebook (same code, same seeds); the control panel re-runs the safety-stock,
simulation and Bayesian-optimisation steps live.

| Step | Method |
|---|---|
| True demand | Hourly stock-out records + each product's normal hourly sales pattern |
| 1 · Forecasting | 6 models (baselines, Holt-Winters, Ridge, Gradient Boosting, LSTM), walk-forward testing, daily updates |
| 2 · Safety stock | Slide formula Z·√(σ_D²·L + D_avg²·σ_L²) and a 20,000-future Monte Carlo |
| 3 · Optimisation | Gaussian-process Bayesian optimisation (Optuna) of reorder point and order quantity under 3 limits |
""")
    st.markdown(T.section("02", "Glossary"), unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([["Reorder point (s)", TIPS["s"]], ["Order quantity (Q)", TIPS["Q"]], ["Safety stock", TIPS["ss"]],
                               ["Demand served (fill rate)", TIPS["fill"]], ["Waste", TIPS["waste"]], ["WAPE", TIPS["wape"]],
                               ["Service level (formula)", TIPS["csl"]], ["Monte Carlo", "Simulating thousands of random futures and reading the answer off the results."],
                               ["Bayesian optimisation", "A search that learns from each simulation where the best settings are likely to be, so it needs far fewer runs."],
                               ["EOQ", "Economic Order Quantity: the textbook order size balancing ordering and holding cost. It ignores spoilage."]],
                              columns=["term", "meaning"]), hide_index=True, width="stretch")
    st.markdown(T.section("03", "Assumptions (not in the data)"), unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([
        ["Supplier lead time", "1/2/3/4 days with probability " + " / ".join(f"{x:.0%}" for x in M["lt_probs"]) + f" (avg {M['lt_mean']:.2f})"],
        ["Service level for formula", f"{M['service_level']:.0%}"], ["Holding / stock-out / waste / ordering cost", f"{M['h_cost']} / {M['p_cost']} / {M['w_cost']} / {M['k_cost']}"],
        ["Shelf life", f"{M['shelf_life']} days"], ["Limits", f"served ≥ {M['min_fill']:.0%}, waste ≤ {M['max_waste']:.0%}, storage ≤ {M['capacity_days']} days of demand"],
    ], columns=["assumption", "value"]), hide_index=True, width="stretch")

    au = D["audit"].copy()
    groups = [("Loading", 0, 4), ("Days & ids", 4, 11), ("Field values", 11, 18), ("Demand recovery", 18, 24), ("No look-ahead", 24, 30),
              ("New-data path", 30, 33), ("Outputs", 33, 39), ("Simulator", 39, 44)]
    n_pass = int((au.result == "PASS").sum())
    st.markdown(T.section("04", "Data checks", f"{n_pass} of {len(au)} automatic checks pass. They run inside the notebook and stop it if anything fails."),
                unsafe_allow_html=True)
    cols = st.columns(2)
    for gi, (name, lo, hi) in enumerate(groups):
        part = au.iloc[lo:hi]; ok = int((part.result == "PASS").sum())
        with cols[gi % 2].expander(f"{'✓' if ok == len(part) else '✕'}  {name} · {ok}/{len(part)}"):
            st.markdown("".join(f'<div class="check-row"><span class="ok">{"✓" if r.result == "PASS" else "✕"}</span><span>{r.check}</span>'
                                f'<span class="d">{str(r.detail)[:40] if str(r.detail) not in ("", "nan") else ""}</span></div>' for r in part.itertuples()),
                        unsafe_allow_html=True)
    st.markdown(T.section("05", "Limits of the study"), unsafe_allow_html=True)
    st.markdown("""
- Sales are in the dataset's scaled units, not real item counts.
- 90 days of history gives no yearly seasonality.
- Future weather is not known in advance, so last week's weather is used.
- The day-by-day replay needs a full re-train per day, so it was run in the notebook for the focus product only.
""")
    st.markdown(f"<div class='foot'>Data: FreshRetailNet-50K (Dingdong-Inc, CC BY 4.0). Store {STORE}, notebook focus SKU {HERO}, seed {SEED}. "
                f"Built by {M['student_id']}.</div>", unsafe_allow_html=True)


# =============================== navigation ===============================
header()
pages = [st.Page(page_overview, title="Overview", icon=":material/space_dashboard:", default=True),
         st.Page(page_demand, title="Demand", icon=":material/trending_up:", url_path="demand"),
         st.Page(page_safety, title="Safety stock", icon=":material/shield:", url_path="safety-stock"),
         st.Page(page_optimiser, title="Optimiser", icon=":material/auto_awesome:", url_path="optimiser"),
         st.Page(page_scenarios, title="Scenarios & replay", icon=":material/tune:", url_path="scenarios"),
         st.Page(page_plan, title="Plan", icon=":material/inventory_2:", url_path="plan"),
         st.Page(page_about, title="About & checks", icon=":material/info:", url_path="about")]
st.navigation(pages, position="top").run()
