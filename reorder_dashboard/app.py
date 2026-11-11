"""Inventory Control Room: Streamlit dashboard for the AI-enhanced reorder-point project (slide 35).
Run:  streamlit run app.py"""
import json, math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import engine as E
import theme as T

st.set_page_config(page_title="Inventory Control Room · Store 308", page_icon="📦", layout="wide", initial_sidebar_state="auto")
st.markdown(T.CSS, unsafe_allow_html=True)
DATA = Path(__file__).parent / "data"


# =============================== data ===============================
@st.cache_data(show_spinner=False)
def load():
    rd = lambda n: pd.read_parquet(DATA / f"{n}.parquet")
    d = {n: rd(n) for n in ["store_daily", "hourly", "stockout_heat", "recovery_validation", "model_summary", "test_week_forecasts",
                            "forecast_errors", "next_week_forecast", "plan", "risk_scenarios", "hero_ltd", "bo_trials", "random_trials",
                            "policy_compare", "sensitivity", "live_log", "audit"]}
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
    st.selectbox("Product (SKU)", PRODUCTS, key="sku",
                 format_func=lambda p: f"SKU {p} · {PLAN.loc[p, 'avg demand/day']:.2f}/day" + ("  ★ focus" if p == HERO else ""))
    with st.expander("Supplier lead time", expanded=True):
        st.radio("Delivery pattern", list(LT_PRESETS), key="lt_preset", label_visibility="collapsed")
        lp = LT_PRESETS[st.session_state.lt_preset]
        st.caption("Chance of a delivery taking 1 / 2 / 3 / 4 days: " + " / ".join(f"{x:.0%}" for x in lp))
    with st.expander("Service target & costs", expanded=False):
        st.slider("Service level for the formula", 0.80, 0.99, key="service", step=0.01, format="%.2f")
        st.slider("Holding cost (per unit per day)", 0.01, 0.10, key="h", step=0.01)
        st.slider("Stock-out cost (per unit not served)", 0.25, 5.0, key="p", step=0.25)
        st.slider("Waste cost (per spoiled unit)", 0.0, 3.0, key="w", step=0.25)
        st.slider("Ordering cost (per order)", 0.1, 2.0, key="k", step=0.1)
        st.slider("Shelf life (days)", 2, 8, key="shelf")
    with st.expander("Business limits", expanded=False):
        st.slider("Minimum share of demand served", 0.85, 0.99, key="fill", step=0.01, format="%.2f")
        st.slider("Maximum waste", 0.01, 0.15, key="waste", step=0.01, format="%.2f")
        st.slider("Storage limit (days of demand)", 3, 10, key="cap")
    is_base = all(st.session_state[k] == v for k, v in BASE.items() if k != "sku")
    st.markdown(T.chip("NOTEBOOK BASELINE", "ok") if is_base else T.chip("WHAT-IF MODE", "warn"), unsafe_allow_html=True)
    st.button("Reset to notebook settings", on_click=reset, width="stretch")
    st.caption("Costs are in unit-cost equivalents: 1.0 = the purchase cost of one unit. Sales are in the dataset's scaled units.")

S = st.session_state
SKU, LT = S.sku, LT_PRESETS[S.lt_preset]
LT_MEAN, LT_SD = E.lead_time_stats(LT)
SIM_KW = dict(lt_probs=np.array(LT), shelf=int(S.shelf), p_cost=S.p, h_cost=S.h, w_cost=S.w, k_cost=S.k)
LIMITS = dict(min_fill=S.fill, max_waste=S.waste, cap_days=S.cap)


@st.cache_data(show_spinner=False)
def live_safety(p, lt, service):
    fc, e = product_inputs(p); lm, ls = E.lead_time_stats(lt)
    f = E.safety_stock_formula(fc, e, lm, ls, service)
    mc = E.safety_stock_mc(fc, e, np.array(lt), service, seed=SEED)
    return f, {"SS": mc["SS"], "ROP": mc["ROP"]}, mc["ltd"]


@st.cache_data(show_spinner=False)
def live_optimise(p, lt, service, h, pc, w, k, shelf, fill, waste, cap):
    fc, e = product_inputs(p)
    f, _, _ = live_safety(p, lt, service)
    kw = dict(lt_probs=np.array(lt), shelf=int(shelf), p_cost=pc, h_cost=h, w_cost=w, k_cost=k)
    lim = dict(min_fill=fill, max_waste=waste, cap_days=cap)
    bo = E.optimise(fc, e, f["D_bar"], sim_kw=kw, **lim, margin=M["opt_margin"], n_trials=40, seed=SEED)
    rs = E.optimise(fc, e, f["D_bar"], sim_kw=kw, **lim, margin=M["opt_margin"], n_trials=40, seed=SEED, sampler="random")
    return bo, rs


@st.cache_data(show_spinner=False)
def live_landscape(p, lt, service, h, pc, w, k, shelf, fill, waste, cap, n=24):
    fc, e = product_inputs(p)
    f, _, _ = live_safety(p, lt, service); d = f["D_bar"]
    kw = dict(lt_probs=np.array(lt), shelf=int(shelf), p_cost=pc, h_cost=h, w_cost=w, k_cost=k)
    s_grid = np.linspace(0, 8 * d, n); q_grid = np.linspace(0.25 * d, 6 * d, n)
    cost = np.full((n, n), np.nan); ok = np.zeros((n, n))
    for i, s_ in enumerate(s_grid):
        for j, q_ in enumerate(q_grid):
            r = E.simulate(s_, q_, fc, e, **kw); cost[j, i] = r["cost"]; ok[j, i] = E.feasible(r, d, fill, waste, cap)
    return s_grid, q_grid, cost, ok


@st.cache_data(show_spinner=False)
def live_policies(p, lt, service, h, pc, w, k, shelf, fill, waste, cap):
    fc, e = product_inputs(p)
    f, mc, _ = live_safety(p, lt, service)
    bo, _ = live_optimise(p, lt, service, h, pc, w, k, shelf, fill, waste, cap)
    kw = dict(lt_probs=np.array(lt), shelf=int(shelf), p_cost=pc, h_cost=h, w_cost=w, k_cost=k)
    d = f["D_bar"]; naive = float(NAIVE[p])
    q_text = max(min(math.sqrt(2 * k * d / h), d * shelf), 0.25 * d)     # EOQ, capped at shelf life, at least a quarter day of demand
    pols = {"Class calculator": (naive * 2 + 2 * naive, 2 * naive), "Formula ROP + EOQ (capped)": (f["ROP"], q_text),
            "Monte Carlo ROP + EOQ (capped)": (mc["ROP"], q_text), "AI · Bayesian-optimised": (bo["s"], bo["Q"])}
    rows = []
    for name, (s_, q_) in pols.items():
        r = E.simulate(s_, q_, fc, e, **kw, seed=2024)
        rows.append(dict(policy=name, s=s_, Q=q_, **r, ok=E.feasible(r, d, fill, waste, cap)))
    return pd.DataFrame(rows)


ARGS = (SKU, LT, S.service, S.h, S.p, S.w, S.k, int(S.shelf), S.fill, S.waste, S.cap)

# =============================== header ===============================
pc_base = D["policy_compare"].set_index("policy")
ai_row = pc_base.loc["Step 3 Bayesian-optimised (s, Q)"]
st.markdown(f"""
<div class="hero">
  <div>
    <div class="hero-kicker">AI in Operations · Slide 35 · Reorder point & inventory</div>
    <div class="hero-title">Inventory Control Room</div>
    <div class="hero-sub">Fresh-grocery store <b>{STORE}</b> · {M['n_products']} products · FreshRetailNet-50K (Dingdong, 2025) ·
    forecasting, dynamic safety stock and Bayesian optimisation of the reorder point. Personal data slice from seed of
    <b>{M['student_id']}</b>.</div>
  </div>
  <div style="display:flex; gap:10px; flex-wrap:wrap">
    <span class="live"><span class="dot"></span>LIVE ENGINE</span>
    <span class="live" style="border-color:rgba(255,181,71,.35);background:rgba(255,181,71,.06)"><span class="dot amber"></span>{'BASELINE' if is_base else 'WHAT-IF'}</span>
  </div>
</div>""", unsafe_allow_html=True)

tabs = st.tabs(["◉ Command deck", "◈ Demand intelligence", "◇ Safety stock lab", "✦ Optimiser", "◐ Scenario lab",
                "▶ Live replay", "▤ Replenishment plan", "✓ Data integrity", "ⓘ About"])


def fig_base(height=360, title=None):
    f = go.Figure(); f.update_layout(height=height, title=title); return f


def show(fig, key):
    st.plotly_chart(fig, width="stretch", key=f"chart_{key}", config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


# =============================== 1. Command deck ===============================
with tabs[0]:
    c = st.columns(4)
    c[0].markdown(T.kpi("AI policy · cost per day", f"{ai_row.cost:.3f}",
                        f"vs {pc_base.loc['Class calculator (sales avg, fixed SS; Q assumed)'].cost:.3f} class calculator · "
                        f"{pc_base.loc['Step 2 formula ROP + EOQ (capped)'].cost:.3f} textbook", chip=T.chip("WITHIN ALL LIMITS", "ok")), unsafe_allow_html=True)
    c[1].markdown(T.kpi("Demand served", f"{ai_row.fill_rate:.1%}", f"target ≥ {M['min_fill']:.0%}", accent=T.NEON_TEAL), unsafe_allow_html=True)
    c[2].markdown(T.kpi("Waste (spoiled stock)", f"{ai_row.waste_rate:.1%}", f"limit ≤ {M['max_waste']:.0%}", accent=T.NEON_AMBER), unsafe_allow_html=True)
    n_ok = int(pc_base["meets all constraints"].sum())
    c[3].markdown(T.kpi("Policies within all limits", f"{n_ok} / {len(pc_base)}", "tested on 300 unseen futures", accent="#8E7EF0",
                        chip=T.chip("ONLY THE AI POLICY", "ok")), unsafe_allow_html=True)

    if not is_base:
        st.markdown('<div class="callout" style="margin-top:12px">This page shows the notebook baseline. Your what-if results are on the '
                    '<b>Safety stock lab</b>, <b>Optimiser</b> and <b>Replenishment plan</b> pages.</div>', unsafe_allow_html=True)
    st.markdown(T.section("01", "The pipeline", "Each stage replaces a hand-typed number in the class calculator with something learned from data."),
                unsafe_allow_html=True)
    ms = D["model_summary"].set_index("model")
    st.markdown(f"""<div class="pipe">
      <div class="pipe-step"><div class="n">STAGE 0</div><div class="t">Recover true demand</div><div class="m">+{M['understatement']:.1%}</div>
        <div class="d">demand hidden by empty shelves, recovered from hourly stock-out records</div></div>
      <div class="pipe-step"><div class="n">STEP 1</div><div class="t">Forecast demand</div><div class="m">{ms.loc[M['best_model'], 'FINAL TEST WAPE (new week)']:.1%}</div>
        <div class="d">error of {M['best_model'].split(' (')[0]} on the unseen week (vs {ms.loc['Same weekday last week', 'FINAL TEST WAPE (new week)']:.1%} baseline)</div></div>
      <div class="pipe-step"><div class="n">STEP 2</div><div class="t">Dynamic safety stock</div><div class="m">{PLAN['MC vs formula'].mean():+.1%}</div>
        <div class="d">Monte Carlo reorder points vs the textbook formula, on average across {len(PLAN)} products</div></div>
      <div class="pipe-step"><div class="n">STEP 3</div><div class="t">Bayesian optimisation</div><div class="m">40 vs 1,600</div>
        <div class="d">simulations to find the best policy, vs a brute-force search ({M['feasible_share']:.1%} of policies are feasible)</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown(T.section("02", f"Who wins on SKU {HERO}, and why",
                          "Four policies replayed on 300 random futures they were never tuned on. The shaded box is where all three business limits are met."),
                unsafe_allow_html=True)
    pcb = D["policy_compare"].copy()
    short = {"Class calculator (sales avg, fixed SS; Q assumed)": "Class calculator", "Step 2 formula ROP + EOQ (capped)": "Formula + EOQ",
             "Step 2 Monte Carlo ROP + EOQ (capped)": "Monte Carlo + EOQ", "Step 3 Bayesian-optimised (s, Q)": "AI · Bayesian"}
    pcb["label"] = pcb.policy.map(short)
    a, b = st.columns([1.15, 1])
    with a:
        f = fig_base(380, "Constraint map: share of demand served vs waste")
        f.add_shape(type="rect", x0=0, x1=M["max_waste"], y0=M["min_fill"], y1=1.0, fillcolor="rgba(46,230,197,0.08)",
                    line=dict(color=T.NEON_TEAL, width=1, dash="dot"), layer="below")
        f.add_annotation(x=M["max_waste"], y=1.0, text="all limits met", showarrow=False, xanchor="right", yanchor="top",
                         font=dict(color=T.NEON_TEAL, size=11, family="JetBrains Mono"))
        cols = [T.AMBER, T.CORAL, T.VIOLET, T.TEAL]
        pos = {"Class calculator": "top center", "Formula + EOQ": "bottom center", "Monte Carlo + EOQ": "top center", "AI · Bayesian": "bottom center"}
        for (_, r), col in zip(pcb.iterrows(), cols):
            f.add_scatter(x=[r.waste_rate], y=[r.fill_rate], mode="markers+text", name=r.label, text=[r.label], textposition=pos.get(r.label, "top center"),
                          textfont=dict(color=T.TEXT, size=11), marker=dict(size=16 if "AI" in r.label else 12, color=col, line=dict(color=T.SURFACE, width=2)),
                          hovertemplate=f"<b>{r.label}</b><br>waste %{{x:.1%}}<br>served %{{y:.1%}}<br>cost {r.cost:.3f}/day<extra></extra>")
        f.update_xaxes(title="waste (share of stock that spoils)", tickformat=".0%", range=[0, max(0.32, pcb.waste_rate.max() * 1.1)])
        f.update_yaxes(title="share of demand served", tickformat=".0%", range=[0.92, 0.985])
        f.update_layout(showlegend=False); show(f, "cmap")
    with b:
        f = fig_base(380, "Daily cost (lower is better)")
        pcs = pcb.sort_values("cost", ascending=True)
        f.add_bar(y=pcs.label, x=pcs.cost, orientation="h", marker=dict(color=[T.TEAL if "AI" in l else "#3A4766" for l in pcs.label], cornerradius=4),
                  text=[f"{v:.3f} {'✓' if ok else '✕'}" for v, ok in zip(pcs.cost, pcs["meets all constraints"])],
                  textposition="outside", textfont=dict(color=T.TEXT, family="JetBrains Mono"), hovertemplate="%{y}: %{x:.3f}<extra></extra>")
        f.update_xaxes(title="cost per day   (✓ within limits · ✕ breaks a limit)", range=[0, pcs.cost.max() * 1.3])
        f.update_yaxes(autorange="reversed")
        f.update_layout(showlegend=False); show(f, "costbar")
    st.markdown('<div class="callout">Why the AI policy wins: the reorder-point formula never sees <b>spoilage</b> or the <b>cost of empty shelves</b>. '
                'For fresh food, stock that cannot sell in time is thrown away. The optimiser balances empty shelves, storage, waste and ordering together. '
                'The class calculator looks cheaper only because it breaks the waste limit.</div>', unsafe_allow_html=True)

# =============================== 2. Demand intelligence ===============================
with tabs[1]:
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
    f.add_annotation(x=pd.Timestamp(M["new_week"][0]), y=day.demand.max(), text="new-data week", showarrow=False, xanchor="left",
                     font=dict(color=T.TEXT_2, size=11))
    f.update_layout(hovermode="x unified"); f.update_yaxes(title="units (scaled)"); show(f, "timeline")

    a, b = st.columns(2)
    with a:
        h = D["stockout_heat"]
        f = fig_base(320, "Share of products out of stock, by weekday and hour")
        f.add_heatmap(z=h[[f"h{i}" for i in range(24)]].to_numpy(), x=list(range(24)), y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                      colorscale=T.SEQ_TEAL, colorbar=dict(tickformat=".0%", outlinewidth=0), xgap=2, ygap=2,
                      hovertemplate="%{y} %{x}:00 → %{z:.0%} out of stock<extra></extra>")
        f.update_xaxes(title="hour of day", dtick=3); f.update_yaxes(autorange="reversed"); show(f, "heat")
    with b:
        hr = D["hourly"]
        f = fig_base(320, "When customers buy: share of daily sales by hour")
        f.add_bar(x=hr.hour, y=hr.sales_share, marker=dict(color=T.TEAL, cornerradius=4), hovertemplate="%{x}:00 → %{y:.1%} of sales<extra></extra>")
        f.update_xaxes(title="hour of day", dtick=3); f.update_yaxes(tickformat=".0%"); f.update_layout(showlegend=False); show(f, "hourbar")

    v = D["recovery_validation"].set_index("method")
    c = st.columns(3)
    c[0].markdown(T.kpi("Recovery error on known days", f"{v.loc['Hourly-profile recovery', 'error (WAPE)']:.1%}",
                        f"bias {v.loc['Hourly-profile recovery', 'bias']:+.1%} (almost none)"), unsafe_allow_html=True)
    c[1].markdown(T.kpi("If sales are treated as demand", f"{v.loc['Treat sales as demand', 'error (WAPE)']:.1%}",
                        f"bias {v.loc['Treat sales as demand', 'bias']:+.1%}: always too low", accent=T.CRIT), unsafe_allow_html=True)
    c[2].markdown(T.kpi("Days too empty to estimate", f"{M['imputed_share']:.1%}", "filled from the product's previous two weeks", accent=T.NEON_AMBER),
                  unsafe_allow_html=True)

    st.markdown(T.section("02", "Step 1 · Which forecasting model wins",
                          "Six models, trained only on the past and scored on weeks they never saw. Lower error is better."), unsafe_allow_html=True)
    ms = D["model_summary"].sort_values("FINAL TEST WAPE (new week)")
    a, b = st.columns([1.1, 1])
    with a:
        f = fig_base(360, "Forecast error (WAPE): validation weeks vs final unseen week")
        f.add_bar(y=ms.model, x=ms["Validation WAPE (days 69-89)"], name="validation weeks", orientation="h", marker=dict(color=T.VIOLET, cornerradius=4),
                  hovertemplate="%{y}<br>validation %{x:.1%}<extra></extra>")
        f.add_bar(y=ms.model, x=ms["FINAL TEST WAPE (new week)"], name="final unseen week", orientation="h", marker=dict(color=T.TEAL, cornerradius=4),
                  hovertemplate="%{y}<br>final week %{x:.1%}<extra></extra>")
        f.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08); f.update_xaxes(tickformat=".0%"); f.update_yaxes(autorange="reversed")
        show(f, "models")
    with b:
        tw = D["test_week_forecasts"].query("product_id == @SKU").sort_values("t")
        models = [m for m in ms.model]
        pick = st.multiselect("Models to show", models, default=[M["best_model"], "Holt-Winters (time series)", "LSTM (deep learning)"], key="mpick")
        f = fig_base(300, f"SKU {SKU}: forecasts vs true demand on the unseen week")
        T.glow_line(f, tw.dt, tw.demand, T.NEON_TEAL, "true demand")
        f.add_scatter(x=tw.dt, y=tw.sale_amount, name="recorded sales", line=dict(color=T.MUTED, dash="dot", width=2), hovertemplate="sales %{y:.2f}<extra></extra>")
        palette = [T.AMBER, T.VIOLET, T.CORAL, T.SKY, T.LIME, "#C9D3E3"]
        for m, col in zip(pick, palette):
            f.add_scatter(x=tw.dt, y=tw[m], name=m, line=dict(color=col, width=2), marker=dict(size=8), mode="lines+markers",
                          hovertemplate=f"{m}: %{{y:.2f}}<extra></extra>")
        f.update_layout(hovermode="x unified", legend=dict(orientation="h", y=-0.3, yanchor="top"), margin=dict(b=110), height=360); show(f, "fc")
    best_row = ms.set_index("model").loc["LSTM (deep learning)"]
    st.markdown(f'<div class="callout teal">The LSTM is competitive on fully stocked days but forecast {abs(best_row["Final test bias"]):.0%} too low on the final week. '
                f'Ninety days is short for deep learning, so one Gradient Boosting model learned across 2,499 product histories wins.</div>', unsafe_allow_html=True)

# =============================== 3. Safety stock lab ===============================
with tabs[2]:
    f_l, mc_l, ltd = live_safety(SKU, LT, S.service)
    st.markdown(T.section("01", f"SKU {SKU} · safety stock, live",
                          "Recomputed from the control panel: change the supplier pattern or the service level and every number updates."), unsafe_allow_html=True)
    c = st.columns(5)
    c[0].markdown(T.kpi("Avg demand / day", f"{f_l['D_bar']:.2f}", "forecast, bias-corrected"), unsafe_allow_html=True)
    c[1].markdown(T.kpi("Forecast error sd", f"{f_l['sigma_D']:.2f}", "σ_D from real past errors", accent="#8E7EF0"), unsafe_allow_html=True)
    c[2].markdown(T.kpi("Formula reorder point", f"{f_l['ROP']:.2f}", f"safety stock {f_l['SS']:.2f}", accent=T.NEON_AMBER), unsafe_allow_html=True)
    c[3].markdown(T.kpi("Monte Carlo reorder point", f"{mc_l['ROP']:.2f}", f"safety stock {mc_l['SS']:.2f}"), unsafe_allow_html=True)
    c[4].markdown(T.kpi("Class calculator", f"{NAIVE[SKU] * 4:.2f}", "sales avg × 2 + 2 days", accent=T.CRIT), unsafe_allow_html=True)

    a, b = st.columns([1.35, 1])
    with a:
        f = fig_base(380, f"Demand during the lead time: 20,000 simulated futures (service {S.service:.0%})")
        f.add_histogram(x=ltd, nbinsx=70, marker=dict(color="rgba(8,170,144,0.55)", line=dict(color=T.SURFACE, width=1)),
                        hovertemplate="%{x:.2f} units → %{y} futures<extra></extra>", name="simulated futures")
        for x, col, lab, ypos in [(f_l["ROP"], T.NEON_AMBER, "formula", 1.0), (mc_l["ROP"], T.NEON_TEAL, "Monte Carlo", 0.9), (NAIVE[SKU] * 4, T.CRIT, "class", 0.8)]:
            f.add_vline(x=x, line=dict(color=col, width=2, dash="dash" if lab == "class" else "solid"))
            f.add_annotation(x=x, y=ypos, yref="paper", text=f"{lab} {x:.2f}", showarrow=False, xanchor="left", xshift=4,
                             font=dict(color=T.TEXT, size=11, family="JetBrains Mono"), bgcolor="rgba(14,21,40,0.85)")
        f.update_xaxes(title="units needed before the delivery arrives"); f.update_yaxes(title="futures"); f.update_layout(showlegend=False, bargap=0.02)
        show(f, "ltd")
    with b:
        z = f_l["Z"]; dem_part = f_l["sigma_D"] ** 2 * LT_MEAN; lt_part = f_l["D_bar"] ** 2 * LT_SD ** 2
        f = fig_base(380, "Where the uncertainty comes from")
        f.add_bar(x=["demand swings", "delivery-time swings"], y=[dem_part, lt_part], marker=dict(color=[T.TEAL, T.AMBER], cornerradius=4),
                  text=[f"{dem_part / (dem_part + lt_part):.0%}", f"{lt_part / (dem_part + lt_part):.0%}"], textposition="outside",
                  textfont=dict(color=T.TEXT, family="JetBrains Mono"), hovertemplate="%{x}: %{y:.3f}<extra></extra>")
        f.update_yaxes(title="variance inside the square root"); f.update_layout(showlegend=False)
        show(f, "parts")
        st.markdown(f"<div class='section-note'>SS = Z·√(σ<sub>D</sub>²·L + D<sub>avg</sub>²·σ<sub>L</sub>²) with Z = {z:.3f}, "
                    f"L = {LT_MEAN:.2f} days, σ<sub>L</sub> = {LT_SD:.2f}.</div>", unsafe_allow_html=True)

    st.markdown(T.section("02", f"Risk scenarios for SKU {HERO} (notebook baseline)",
                          "Monte Carlo safety stock needed under each supplier risk, at three service levels."), unsafe_allow_html=True)
    rk = D["risk_scenarios"]
    a, b = st.columns([1.2, 1])
    with a:
        f = fig_base(340, "Monte Carlo safety stock by risk scenario and service level")
        for sl, col in zip([90, 95, 99], ["#5FD9C2", "#14A08E", "#0A6964"]):
            f.add_bar(x=rk.scenario.str.split(" \\(").str[0], y=rk[f"MC SS {sl}"], name=f"{sl}% service", marker=dict(color=col, cornerradius=4),
                      hovertemplate=f"%{{x}}<br>{sl}%: %{{y:.2f}}<extra></extra>")
        f.update_layout(barmode="group", bargap=0.25); f.update_yaxes(title="safety stock (units)"); show(f, "risk")
    with b:
        rt = rk[["scenario", "mean lead time", "MC SS 95", "Formula SS 95", "MC ROP 95"]].rename(columns={
            "mean lead time": "avg lead time", "MC SS 95": "MC safety stock", "Formula SS 95": "formula safety stock", "MC ROP 95": "MC reorder point"})
        st.dataframe(rt.round(2), hide_index=True, width="stretch")
        st.markdown("<div class='section-note'>'Disruption' needs slightly less safety stock than 'Unreliable' because its lead time is longer but more "
                    "predictable. Its reorder point is still the highest.</div>", unsafe_allow_html=True)

    st.markdown(T.section("03", "All products: formula vs Monte Carlo",
                          "Points above the diagonal need more stock than the formula says. Those are the products with the most uneven demand."),
                unsafe_allow_html=True)
    f = fig_base(380)
    f.add_scatter(x=PLAN["ROP formula"], y=PLAN["ROP Monte Carlo"], mode="markers", name="products", text=[f"SKU {p}" for p in PLAN.index],
                  marker=dict(size=9, color=T.TEAL, opacity=0.8, line=dict(color=T.SURFACE, width=1)),
                  hovertemplate="%{text}<br>formula %{x:.2f}<br>Monte Carlo %{y:.2f}<extra></extra>")
    mx = float(PLAN[["ROP formula", "ROP Monte Carlo"]].max().max()) * 1.05
    f.add_scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=T.MUTED, dash="dot", width=1), hoverinfo="skip", showlegend=False)
    f.add_scatter(x=[PLAN.loc[SKU, "ROP formula"]], y=[PLAN.loc[SKU, "ROP Monte Carlo"]], mode="markers", name=f"SKU {SKU}",
                  marker=dict(size=16, color=T.NEON_AMBER, line=dict(color=T.SURFACE, width=2)), hovertemplate=f"SKU {SKU}<extra></extra>")
    f.update_xaxes(title="formula reorder point"); f.update_yaxes(title="Monte Carlo reorder point"); show(f, "scatter")

# =============================== 4. Optimiser ===============================
with tabs[3]:
    st.markdown(T.section("01", f"Bayesian optimisation for SKU {SKU}, live",
                          "A simulator copies the shelf (random demand, random deliveries, stock spoiling after the shelf life). "
                          "A Gaussian-process model learns where the cheapest policy that meets every limit lies, and tests only the most promising points."),
                unsafe_allow_html=True)
    with st.spinner("Running Bayesian optimisation (40 simulations) and random search…"):
        bo, rs = live_optimise(*ARGS)
    use_pre = is_base and SKU == HERO
    with st.spinner("Mapping the cost landscape…"):
        if use_pre:
            G = D["grid"]; s_grid, q_grid, cost_map, ok_map = G["s"], G["q"], G["cost"], G["ok"]
        else:
            s_grid, q_grid, cost_map, ok_map = live_landscape(*ARGS)
    c = st.columns(5)
    c[0].markdown(T.kpi("Reorder point s", f"{bo['s']:.2f}", "order when stock falls to this"), unsafe_allow_html=True)
    c[1].markdown(T.kpi("Order quantity Q", f"{bo['Q']:.2f}", "units per order", accent="#8E7EF0"), unsafe_allow_html=True)
    c[2].markdown(T.kpi("Daily cost", f"{bo['cost']:.3f}", "holding + lost sales + waste + ordering", accent=T.NEON_AMBER), unsafe_allow_html=True)
    r = bo["result"]
    c[3].markdown(T.kpi("Demand served", f"{r['fill_rate']:.1%}", f"waste {r['waste_rate']:.1%} · limits ≥{S.fill:.0%} / ≤{S.waste:.0%}"), unsafe_allow_html=True)
    c[4].markdown(T.kpi("Status", "PASS" if bo["feasible"] else "FAIL", "meets every limit" if bo["feasible"] else "closest policy shown",
                        accent=T.NEON_TEAL if bo["feasible"] else T.CRIT,
                        chip=T.chip("WITHIN ALL LIMITS", "ok") if bo["feasible"] else T.chip("NO POLICY MEETS ALL LIMITS", "bad")), unsafe_allow_html=True)
    if not bo["feasible"]:
        st.markdown('<div class="callout">No reorder policy can meet every limit under these settings. The business must relax one (accept more waste '
                    'or a lower service level) or fix the cause (e.g. a more reliable supplier).</div>', unsafe_allow_html=True)

    a, b = st.columns([1.15, 1])
    with a:
        f = fig_base(430, "Cost of every policy (red outline = meets all limits)")
        vmax = float(np.nanpercentile(cost_map, 90))
        f.add_heatmap(x=s_grid, y=q_grid, z=cost_map, colorscale=[[0, "#2EE6C5"], [0.35, "#109B88"], [0.7, "#0B3940"], [1, "#0E1528"]],
                      zmax=vmax, colorbar=dict(title=dict(text="cost/day"), outlinewidth=0), hovertemplate="s %{x:.2f} · Q %{y:.2f}<br>cost %{z:.3f}<extra></extra>")
        if ok_map.any():
            f.add_contour(x=s_grid, y=q_grid, z=ok_map, contours=dict(start=0.5, end=0.5, size=1, coloring="none"), line=dict(color="#FF5C7A", width=2.5),
                          showscale=False, hoverinfo="skip", name="feasible region")
        tr = pd.DataFrame(bo["trials"])
        f.add_scatter(x=tr.s, y=tr.Q, mode="markers", name="Bayesian trials", marker=dict(size=8, color="rgba(255,255,255,0.85)", line=dict(color=T.SURFACE, width=1)),
                      hovertemplate="trial %{text}<br>s %{x:.2f} · Q %{y:.2f}<extra></extra>", text=tr.n)
        f.add_scatter(x=[bo["s"]], y=[bo["Q"]], mode="markers", name="best policy", marker=dict(symbol="star", size=20, color=T.NEON_AMBER, line=dict(color=T.SURFACE, width=2)),
                      hovertemplate="best: s %{x:.2f} · Q %{y:.2f}<extra></extra>")
        f.update_xaxes(title="reorder point s"); f.update_yaxes(title="order quantity Q"); show(f, "land")
        st.caption(("Notebook landscape: 40×40 = 1,600 simulations." if use_pre else "Live landscape: 24×24 = 576 simulations.")
                   + f" {ok_map.mean():.1%} of tested policies meet every limit.")
    with b:
        def running(trials):
            best, out = np.inf, []
            for t in trials:
                if t["feasible"]: best = min(best, t["cost"])
                out.append(best if np.isfinite(best) else None)
            return out
        f = fig_base(430, "How fast each method finds a good policy")
        T.glow_line(f, list(range(1, 41)), running(bo["trials"]), T.NEON_TEAL, "Bayesian optimisation", hover="%{y:.4f}")
        rr = running(rs["trials"]); rs_ok = any(v is not None for v in rr)
        f.add_scatter(x=list(range(1, 41)), y=rr, mode="lines", name="random search" if rs_ok else "random search (found nothing feasible)",
                      line=dict(color=T.CORAL, width=2), hovertemplate="random: %{y:.4f}<extra></extra>")
        grid_best = np.nanmin(np.where(ok_map > 0, cost_map, np.nan)) if ok_map.any() else None
        if grid_best is not None:
            f.add_hline(y=grid_best, line=dict(color=T.NEON_AMBER, dash="dot"),
                        annotation_text=f"brute-force grid best ({'1,600' if use_pre else '576'} sims): {grid_best:.3f}", annotation_position="top left",
                        annotation_font=dict(color=T.TEXT_2, size=11))
        f.update_xaxes(title="number of simulations"); f.update_yaxes(title="best feasible cost so far"); show(f, "conv")

    st.markdown(T.section("02", "Policies compared on 300 new futures",
                          "Every policy is replayed on random futures that none of them was tuned on (same settings as the control panel)."),
                unsafe_allow_html=True)
    with st.spinner("Replaying policies…"):
        pol = live_policies(*ARGS)
    a, b = st.columns([1, 1.1])
    with a:
        f = fig_base(380, "Constraint map")
        f.add_shape(type="rect", x0=0, x1=S.waste, y0=S.fill, y1=1.0, fillcolor="rgba(46,230,197,0.08)", line=dict(color=T.NEON_TEAL, width=1, dash="dot"), layer="below")
        for (_, rr_), col, tp in zip(pol.iterrows(), [T.AMBER, T.CORAL, T.VIOLET, T.TEAL], ["top center", "bottom center", "top center", "bottom center"]):
            lbl = {"Class calculator": "Class", "Formula ROP + EOQ (capped)": "Formula", "Monte Carlo ROP + EOQ (capped)": "Monte Carlo",
                   "AI · Bayesian-optimised": "AI · Bayesian"}[rr_.policy]
            f.add_scatter(x=[rr_.waste_rate], y=[rr_.fill_rate], mode="markers+text", text=[lbl], textposition=tp,
                          textfont=dict(color=T.TEXT, size=10.5), marker=dict(size=14, color=col, line=dict(color=T.SURFACE, width=2)), name=rr_.policy,
                          hovertemplate=f"<b>{rr_.policy}</b><br>waste %{{x:.1%}} · served %{{y:.1%}}<br>cost {rr_.cost:.3f}<extra></extra>")
        f.update_xaxes(title="waste", tickformat=".0%", range=[0, max(S.waste, pol.waste_rate.max()) * 1.25])
        f.update_yaxes(title="demand served", tickformat=".0%"); f.update_layout(showlegend=False); show(f, "cmap2")
    with b:
        tbl = pol.assign(status=np.where(pol.ok, "✓ yes", "✕ no"))[
            ["policy", "status", "cost", "fill_rate", "waste_rate", "s", "Q", "orders_per_week"]]
        st.markdown("<div style='height:52px'></div>", unsafe_allow_html=True)
        st.dataframe(tbl, hide_index=True, width="stretch", column_config={
            "policy": st.column_config.TextColumn("policy", width="medium"), "status": st.column_config.TextColumn("within limits", width="small"),
            "s": st.column_config.NumberColumn("reorder point", format="%.2f"), "Q": st.column_config.NumberColumn("order qty", format="%.2f"),
            "cost": st.column_config.NumberColumn("cost/day", format="%.3f"), "fill_rate": st.column_config.NumberColumn("served", format="percent"),
            "waste_rate": st.column_config.NumberColumn("waste", format="percent"), "orders_per_week": st.column_config.NumberColumn("orders/week", format="%.1f")})

# =============================== 5. Scenario lab ===============================
with tabs[4]:
    st.markdown(T.section("01", f"How the best decision moves (SKU {HERO}, notebook baseline)",
                          "Each row is a full Bayesian optimisation under different business conditions. The formula cannot react to shelf life or costs at all."),
                unsafe_allow_html=True)
    se = D["sensitivity"]
    f = fig_base(440)
    yy = list(se.scenario)
    f.add_bar(y=yy, x=se["formula ROP"], orientation="h", name="formula reorder point", marker=dict(color="#3A4766", cornerradius=4),
              hovertemplate="%{y}<br>formula %{x:.2f}<extra></extra>")
    okm = se["meets constraints"].astype(bool)
    f.add_bar(y=[y for y, o in zip(yy, okm) if o], x=se.loc[okm, "optimal s"], orientation="h", name="optimal reorder point (Bayesian)",
              marker=dict(color=T.TEAL, cornerradius=4), hovertemplate="%{y}<br>optimal %{x:.2f}<extra></extra>")
    for y in [y for y, o in zip(yy, okm) if not o]:
        f.add_annotation(y=y, x=0.05, text="✕ no policy can meet all limits", showarrow=False, xanchor="left", yshift=-9,
                         font=dict(color="#FF8FA3", size=11, family="JetBrains Mono"))
    f.update_layout(barmode="group", bargap=0.3); f.update_yaxes(autorange="reversed"); f.update_xaxes(title="reorder point"); show(f, "sens")
    st.dataframe(se[["scenario", "optimal s", "optimal Q", "daily cost", "fill rate", "waste", "meets constraints", "formula ROP", "note"]].round(3),
                 hide_index=True, width="stretch")
    st.markdown('<div class="callout teal">The optimal reorder point barely moves with the stock-out cost, because the 95% service limit already decides how '
                'much protection is needed. To create your own scenario, use the control panel: the Optimiser and Safety stock pages recompute live.</div>',
                unsafe_allow_html=True)

# =============================== 6. Live replay ===============================
with tabs[5]:
    ll = D["live_log"]
    st.markdown(T.section("01", f"The final week, replayed day by day (SKU {HERO}, notebook baseline)",
                          "Each morning yesterday's raw data arrives, the model re-trains, every earlier forecast is scored, and the plan is re-optimised."),
                unsafe_allow_html=True)
    day_sel = st.select_slider("Replay day", options=list(ll.date), value=list(ll.date)[-1], key="replay")
    i = list(ll.date).index(day_sel); row = ll.iloc[i]; prev = ll.iloc[i - 1] if i > 0 else row
    c = st.columns(4)
    c[0].markdown(T.kpi("Forecast, next 7 days", f"{row['forecast next 7d (avg)']:.2f}", f"{row['forecast next 7d (avg)'] - prev['forecast next 7d (avg)']:+.2f} vs previous day"),
                  unsafe_allow_html=True)
    c[1].markdown(T.kpi("Monte Carlo reorder point", f"{row['Monte Carlo ROP']:.2f}", f"{row['Monte Carlo ROP'] - prev['Monte Carlo ROP']:+.2f} vs previous day",
                        accent="#8E7EF0"), unsafe_allow_html=True)
    c[2].markdown(T.kpi("AI reorder point s", f"{row['Bayesian s']:.2f}", f"order {row['Bayesian Q']:.2f} units", accent=T.NEON_AMBER), unsafe_allow_html=True)
    c[3].markdown(T.kpi("Forecast errors learned", f"{int(row['errors known'])}", "grows as each day is scored",
                        chip=T.chip("WITHIN ALL LIMITS", "ok") if row["meets constraints"] else T.chip("FLAGGED FOR A MANAGER", "bad")), unsafe_allow_html=True)
    f = fig_base(360, "Reorder point re-optimised every morning")
    f.add_bar(x=ll.date, y=ll["actual demand that day"], name="actual demand that day", marker=dict(color="rgba(163,175,196,0.25)", cornerradius=4),
              hovertemplate="demand %{y:.2f}<extra></extra>")
    T.glow_line(f, ll.date, ll["Bayesian s"], T.NEON_TEAL, "AI reorder point s")
    f.add_scatter(x=ll.date, y=ll["Monte Carlo ROP"], name="Monte Carlo ROP", line=dict(color=T.VIOLET, width=2), marker=dict(size=8), mode="lines+markers",
                  hovertemplate="MC %{y:.2f}<extra></extra>")
    f.add_scatter(x=ll.date, y=ll["formula ROP"], name="formula ROP", line=dict(color=T.AMBER, width=2), marker=dict(size=8), mode="lines+markers",
                  hovertemplate="formula %{y:.2f}<extra></extra>")
    f.add_vrect(x0=i - 0.45, x1=i + 0.45, fillcolor="rgba(46,230,197,0.08)", line_width=0)
    f.update_xaxes(type="category"); f.update_yaxes(title="units"); f.update_layout(hovermode="x unified"); show(f, "replay")
    st.dataframe(ll.drop(columns=["note"], errors="ignore").round(3), hide_index=True, width="stretch")

# =============================== 7. Replenishment plan ===============================
with tabs[6]:
    st.markdown(T.section("01", "Next week's replenishment plan, all products",
                          "Forecast, safety stock and reorder point for every product. In what-if mode the formula and Monte Carlo columns are recomputed with your settings."),
                unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def live_plan(lt, service):
        out = []
        for p in PRODUCTS:
            f_, m_, _ = live_safety(p, lt, service)
            out.append(dict(product=p, ROP_formula_live=f_["ROP"], ROP_MC_live=m_["ROP"], SS_formula_live=f_["SS"], SS_MC_live=m_["SS"]))
        return pd.DataFrame(out).set_index("product")

    pl = PLAN.copy()
    if not is_base:
        with st.spinner("Recomputing the plan for every product…"):
            lv = live_plan(LT, S.service)
        pl["SS formula"], pl["ROP formula"], pl["SS Monte Carlo"], pl["ROP Monte Carlo"] = lv.SS_formula_live, lv.ROP_formula_live, lv.SS_MC_live, lv.ROP_MC_live
        st.markdown('<div class="callout">What-if mode: formula and Monte Carlo columns use your control-panel settings. The Bayesian columns show the notebook '
                    'baseline (re-optimise a product on the Optimiser page).</div>', unsafe_allow_html=True)
    c = st.columns(4)
    c[0].markdown(T.kpi("Products planned", f"{len(pl)}", f"store {STORE}"), unsafe_allow_html=True)
    c[1].markdown(T.kpi("Total safety stock · formula", f"{pl['SS formula'].sum():.1f}", "units (scaled)", accent=T.NEON_AMBER), unsafe_allow_html=True)
    c[2].markdown(T.kpi("Total safety stock · Monte Carlo", f"{pl['SS Monte Carlo'].sum():.1f}", f"{pl['SS Monte Carlo'].sum() / pl['SS formula'].sum() - 1:+.0%} vs formula"),
                  unsafe_allow_html=True)
    nb = int(pl["Bayesian s"].notna().sum())
    c[3].markdown(T.kpi("AI-optimised products", f"{nb}", "highest-volume products", accent="#8E7EF0",
                        chip=T.chip(f"{int(pl['Bayesian meets constraints'].fillna(False).astype(bool).sum())} / {nb} WITHIN LIMITS", "ok")), unsafe_allow_html=True)
    top = pl.sort_values("avg demand/day", ascending=False).head(15)
    f = fig_base(360, "Top 15 products by demand: reorder points")
    xs = [f"SKU {p}" for p in top.index]
    f.add_bar(x=xs, y=top["ROP formula"], name="formula", marker=dict(color=T.AMBER, cornerradius=4), hovertemplate="%{x}<br>formula %{y:.2f}<extra></extra>")
    f.add_bar(x=xs, y=top["ROP Monte Carlo"], name="Monte Carlo", marker=dict(color=T.VIOLET, cornerradius=4), hovertemplate="%{x}<br>MC %{y:.2f}<extra></extra>")
    f.add_bar(x=xs, y=top["Bayesian s"], name="AI · Bayesian", marker=dict(color=T.TEAL, cornerradius=4), hovertemplate="%{x}<br>AI %{y:.2f}<extra></extra>")
    f.update_layout(barmode="group", bargap=0.2, bargroupgap=0.06); f.update_yaxes(title="reorder point (units)"); show(f, "plan")
    q = st.text_input("Search a product (SKU number)", key="plan_q", placeholder="e.g. 201")
    view = pl.reset_index()
    if q.strip():
        view = view[view["product"].astype(str).str.contains(q.strip(), regex=False)]
    cols = ["product", "avg demand/day", "forecast error sd", "SS formula", "ROP formula", "SS Monte Carlo", "ROP Monte Carlo", "class ROP",
            "Bayesian s", "Bayesian Q", "Bayesian daily cost", "Bayesian meets constraints"]
    st.dataframe(view[cols], hide_index=True, width="stretch", height=420, column_config={
        "product": st.column_config.NumberColumn("SKU", format="%d"),
        "avg demand/day": st.column_config.ProgressColumn("avg demand/day", min_value=0, max_value=float(pl["avg demand/day"].max()), format="%.2f"),
        **{c_: st.column_config.NumberColumn(c_, format="%.2f") for c_ in cols[2:11]}})
    st.download_button("⬇  Download plan (CSV)", pl.reset_index()[cols].round(4).to_csv(index=False).encode(), file_name=f"replenishment_plan_store{STORE}.csv",
                       mime="text/csv")

# =============================== 8. Data integrity ===============================
with tabs[7]:
    au = D["audit"].copy()
    groups = [("Loading", 0, 4), ("Days & ids", 4, 11), ("Field values", 11, 18), ("Demand recovery", 18, 24), ("No look-ahead", 24, 30),
              ("New-data path", 30, 33), ("Outputs", 33, 39), ("Simulator", 39, 44)]
    au["area"] = ""
    for name, lo, hi in groups:
        au.loc[lo:hi - 1, "area"] = name
    n_pass = int((au.result == "PASS").sum())
    st.markdown(T.section("01", "Data-integration checks",
                          "Every check runs inside the notebook and stops it if anything fails: the data you see here is exactly the data the models used."),
                unsafe_allow_html=True)
    a, b = st.columns([1, 2.2])
    with a:
        f = fig_base(300)
        f.add_pie(values=[n_pass, len(au) - n_pass], labels=["pass", "fail"], hole=0.78, sort=False, marker=dict(colors=[T.NEON_TEAL, "#2A3350"], line=dict(color=T.BG, width=2)),
                  textinfo="none", hovertemplate="%{label}: %{value}<extra></extra>")
        f.add_annotation(text=f"<b>{n_pass}/{len(au)}</b><br><span style='font-size:12px'>checks pass</span>", showarrow=False,
                         font=dict(size=30, color=T.TEXT, family="JetBrains Mono"))
        f.update_layout(showlegend=False, margin=dict(t=10, b=10)); show(f, "ring")
    with b:
        ga = au.groupby("area", sort=False).result.apply(lambda s: (s == "PASS").sum()).reindex([g[0] for g in groups])
        tot = au.groupby("area", sort=False).size().reindex([g[0] for g in groups])
        f = fig_base(300, "Checks by area")
        f.add_bar(y=ga.index, x=ga.values, orientation="h", marker=dict(color=T.TEAL, cornerradius=4), text=[f"{p}/{t}" for p, t in zip(ga, tot)],
                  textposition="outside", textfont=dict(color=T.TEXT, family="JetBrains Mono"), hovertemplate="%{y}: %{x} passed<extra></extra>")
        f.update_yaxes(autorange="reversed"); f.update_xaxes(range=[0, tot.max() + 1.5], title="checks passed"); f.update_layout(showlegend=False); show(f, "areas")
    st.dataframe(au[["area", "result", "check", "detail"]], hide_index=True, width="stretch", height=480)

# =============================== 9. About ===============================
with tabs[8]:
    st.markdown(T.section("01", "What this is"), unsafe_allow_html=True)
    st.markdown(f"""
This dashboard presents the **AI in Operations, slide 35** exercise: *enhance the reorder-point calculator and sensitivity analysis with predictive intelligence,
adaptive decision-making and real-time optimisation*. All results come from the project notebook (same code, same seeds). The control panel re-runs the
safety-stock, simulation and Bayesian-optimisation steps live.

| Step | Method |
|---|---|
| True demand | Hourly stock-out records + each product's normal hourly sales pattern |
| 1 · Forecasting | 6 models (baselines, Holt-Winters, Ridge, Gradient Boosting, LSTM), walk-forward testing, daily updates |
| 2 · Safety stock | Slide formula Z·√(σ_D²·L + D_avg²·σ_L²) and 20,000-future Monte Carlo |
| 3 · Optimisation | Gaussian-process Bayesian optimisation (Optuna) of reorder point and order quantity under 3 constraints |
""")
    st.markdown(T.section("02", "Assumptions (not in the data)"), unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([
        ["Supplier lead time", "1/2/3/4 days with probability " + " / ".join(f"{x:.0%}" for x in M["lt_probs"]) + f" (avg {M['lt_mean']:.2f})"],
        ["Service level for formula", f"{M['service_level']:.0%}"], ["Holding / stock-out / waste / ordering cost", f"{M['h_cost']} / {M['p_cost']} / {M['w_cost']} / {M['k_cost']}"],
        ["Shelf life", f"{M['shelf_life']} days"], ["Limits", f"served ≥ {M['min_fill']:.0%}, waste ≤ {M['max_waste']:.0%}, storage ≤ {M['capacity_days']} days of demand"],
    ], columns=["assumption", "value"]), hide_index=True, width="stretch")
    st.markdown("""
**Limits of the study.**
- Sales are in the dataset's scaled units, not real item counts.
- 90 days of history gives no yearly seasonality.
- Future weather is not known in advance, so last week's weather is used.
- The live replay and scenario pages show the notebook baseline for the focus product.
""")
    st.markdown(f"<div class='foot'>Data: FreshRetailNet-50K (Dingdong-Inc, CC BY 4.0). Store {STORE}, focus SKU {HERO}, seed {SEED}. "
                f"Built by {M['student_id']}.</div>", unsafe_allow_html=True)
