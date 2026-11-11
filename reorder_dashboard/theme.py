"""Midnight Control Room theme: CSS, Plotly template and small HTML components."""
import plotly.graph_objects as go
import plotly.io as pio

# ---- tokens ----
BG, SURFACE, SURFACE_2 = "#070B16", "#0E1528", "#131C36"
TEXT, TEXT_2, MUTED = "#E8EEF8", "#A3AFC4", "#6B7790"
NEON_TEAL, NEON_AMBER, CRIT = "#2EE6C5", "#FFB547", "#FF5C7A"     # UI glow accents (not data series)
# Data series: validated for the dark surface (lightness band, chroma, colour-blind separation, contrast all PASS)
SERIES = ["#08AA90", "#C48308", "#8E7EF0", "#EA587A", "#3E95F0", "#78A301"]
TEAL, AMBER, VIOLET, CORAL, SKY, LIME = SERIES
SEQ_TEAL = [[0, "#0E1528"], [0.25, "#0B3940"], [0.5, "#0A6964"], [0.75, "#109B88"], [1, "#2EE6C5"]]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600;700&display=swap');
:root {{ --teal:{NEON_TEAL}; --amber:{NEON_AMBER}; --crit:{CRIT}; --text:{TEXT}; --text2:{TEXT_2}; --muted:{MUTED};
        --surface:{SURFACE}; --surface2:{SURFACE_2}; }}
html, body, .stApp, [data-testid="stMarkdownContainer"], .stApp p, .stApp li, .stApp label {{ font-family:'Inter',sans-serif; }}
.stApp {{
  background:
    radial-gradient(1200px 600px at 8% -10%, rgba(46,230,197,0.10), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(142,126,240,0.10), transparent 55%),
    linear-gradient(180deg, {BG} 0%, #060914 100%);
  color:{TEXT};
}}
.stApp::before {{ content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
  background-image: linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
  background-size: 48px 48px; mask-image: radial-gradient(ellipse at top, black 30%, transparent 75%); }}
#MainMenu, footer, [data-testid="stDecoration"] {{ visibility:hidden; height:0; }}
[data-testid="stAppDeployButton"], [data-testid="stToolbarActions"] {{ display:none !important; }}
header[data-testid="stHeader"] {{ background:transparent; }}
.block-container {{ padding-top:1.4rem; padding-bottom:3rem; max-width:1480px; }}
h1,h2,h3,h4 {{ font-family:'Space Grotesk',sans-serif !important; letter-spacing:-0.01em; color:{TEXT}; }}

/* sidebar = control panel */
[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #0A1022 0%, #070B16 100%); border-right:1px solid rgba(46,230,197,0.12); }}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] summary span:not([data-testid="stIconMaterial"]) {{ font-family:'Inter',sans-serif; }}
[data-testid="stIconMaterial"], .material-symbols-rounded {{ font-family:'Material Symbols Rounded' !important; }}
[data-testid="stSidebar"] [data-testid="stExpander"] {{ background:rgba(19,28,54,0.55); border:1px solid rgba(255,255,255,0.06); border-radius:14px; }}

/* tabs = pill navigation */
.stTabs [data-baseweb="tab-list"], .stTabs [role="tablist"] {{ gap:6px; background:rgba(14,21,40,0.7); padding:6px; border-radius:16px;
  border:1px solid rgba(46,230,197,0.14); backdrop-filter:blur(10px); overflow-x:auto; }}
.stTabs [data-baseweb="tab"], .stTabs [role="tab"] {{ display:flex; align-items:center; cursor:pointer;  height:40px; border-radius:11px; padding:0 16px; color:{TEXT_2}; font-family:'Space Grotesk',sans-serif;
  font-weight:500; letter-spacing:.02em; background:transparent; white-space:nowrap; }}
.stTabs [role="tab"]:hover {{ color:{TEXT}; background:rgba(255,255,255,0.04); }}
.stTabs [aria-selected="true"] {{ background:linear-gradient(135deg, rgba(46,230,197,0.20), rgba(142,126,240,0.18)) !important;
  color:{TEXT} !important; box-shadow: inset 0 0 0 1px rgba(46,230,197,0.45), 0 0 18px rgba(46,230,197,0.18); }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display:none; }}

/* glass cards */
.glass {{ position:relative; background:linear-gradient(160deg, rgba(23,33,62,0.72), rgba(12,18,36,0.72));
  border:1px solid rgba(255,255,255,0.07); border-radius:18px; padding:18px 20px; backdrop-filter:blur(14px);
  box-shadow: 0 10px 30px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04); overflow:hidden; }}
.glass::after {{ content:""; position:absolute; left:0; top:14px; bottom:14px; width:3px; border-radius:3px; background:var(--accent, var(--teal));
  box-shadow:0 0 14px var(--accent, var(--teal)); }}
.kpi-label {{ font-family:'JetBrains Mono',monospace; font-size:10.5px; letter-spacing:.12em; line-height:1.4; text-transform:uppercase; color:{MUTED}; }}
.kpi-value {{ font-family:'JetBrains Mono',monospace; font-size:clamp(20px, 2.1vw, 32px); font-weight:700; color:{TEXT}; line-height:1.15; margin-top:6px;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
  text-shadow:0 0 22px rgba(46,230,197,0.18); }}
.kpi-sub {{ font-size:12.5px; color:{TEXT_2}; margin-top:6px; }}
.chip {{ display:inline-flex; align-items:center; gap:6px; font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:.06em;
  padding:3px 9px; border-radius:999px; border:1px solid rgba(255,255,255,0.12); color:{TEXT}; background:rgba(255,255,255,0.04); }}
.chip.ok {{ border-color:rgba(46,230,197,0.55); background:rgba(46,230,197,0.10); }}
.chip.bad {{ border-color:rgba(255,92,122,0.55); background:rgba(255,92,122,0.10); }}
.chip.warn {{ border-color:rgba(255,181,71,0.55); background:rgba(255,181,71,0.10); }}

/* hero header */
.hero {{ display:flex; justify-content:space-between; align-items:flex-end; gap:24px; flex-wrap:wrap; margin:4px 0 18px; }}
.hero-title {{ font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:40px; letter-spacing:-0.02em; line-height:1.05;
  background:linear-gradient(90deg, #FFFFFF 0%, {NEON_TEAL} 55%, #8E7EF0 100%); -webkit-background-clip:text; background-clip:text; color:transparent; }}
.hero-kicker {{ font-family:'JetBrains Mono',monospace; font-size:12px; letter-spacing:.22em; color:{NEON_TEAL}; text-transform:uppercase; }}
.hero-sub {{ color:{TEXT_2}; font-size:14px; margin-top:8px; max-width:780px; }}
.live {{ display:inline-flex; align-items:center; gap:10px; font-family:'JetBrains Mono',monospace; font-size:12px; letter-spacing:.12em; color:{TEXT};
  padding:8px 14px; border-radius:999px; border:1px solid rgba(46,230,197,0.35); background:rgba(46,230,197,0.06); }}
.dot {{ width:9px; height:9px; border-radius:50%; background:{NEON_TEAL}; box-shadow:0 0 0 0 rgba(46,230,197,0.7); animation:pulse 1.8s infinite; }}
.dot.amber {{ background:{NEON_AMBER}; box-shadow:0 0 0 0 rgba(255,181,71,0.7); animation:pulse-a 1.8s infinite; }}
@keyframes pulse {{ 0%{{box-shadow:0 0 0 0 rgba(46,230,197,.7)}} 70%{{box-shadow:0 0 0 10px rgba(46,230,197,0)}} 100%{{box-shadow:0 0 0 0 rgba(46,230,197,0)}} }}
@keyframes pulse-a {{ 0%{{box-shadow:0 0 0 0 rgba(255,181,71,.7)}} 70%{{box-shadow:0 0 0 10px rgba(255,181,71,0)}} 100%{{box-shadow:0 0 0 0 rgba(255,181,71,0)}} }}
@media (prefers-reduced-motion: reduce) {{ .dot, .dot.amber {{ animation:none; }} }}

.section {{ display:flex; align-items:baseline; gap:12px; margin:26px 0 10px; }}
.section-num {{ font-family:'JetBrains Mono',monospace; color:{NEON_TEAL}; font-size:12px; letter-spacing:.18em; }}
.section-title {{ font-family:'Space Grotesk',sans-serif; font-size:22px; font-weight:600; color:{TEXT}; }}
.section-note {{ color:{TEXT_2}; font-size:13.5px; margin:-4px 0 12px; max-width:980px; }}
.pipe {{ display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:12px; }}
.pipe-step {{ position:relative; padding:16px 16px 14px; border-radius:16px; border:1px solid rgba(255,255,255,0.07);
  background:linear-gradient(160deg, rgba(23,33,62,0.6), rgba(12,18,36,0.6)); }}
.pipe-step .n {{ font-family:'JetBrains Mono',monospace; font-size:11px; color:{NEON_TEAL}; letter-spacing:.16em; }}
.pipe-step .t {{ font-family:'Space Grotesk',sans-serif; font-size:16px; font-weight:600; margin-top:4px; }}
.pipe-step .m {{ font-family:'JetBrains Mono',monospace; font-size:22px; font-weight:700; margin-top:8px; color:{TEXT}; }}
.pipe-step .d {{ font-size:12px; color:{TEXT_2}; margin-top:4px; }}
@media (max-width: 900px) {{ .pipe {{ grid-template-columns: repeat(2, minmax(0,1fr)); }} .hero-title {{ font-size:30px; }} }}
.callout {{ border-radius:14px; padding:12px 16px; border:1px solid rgba(255,181,71,0.35); background:rgba(255,181,71,0.07); color:{TEXT}; font-size:13.5px; }}
.callout.teal {{ border-color:rgba(46,230,197,0.35); background:rgba(46,230,197,0.06); }}
[data-testid="stDataFrame"] {{ border-radius:14px; overflow:hidden; border:1px solid rgba(255,255,255,0.06); }}
.stButton > button, .stDownloadButton > button {{ border-radius:12px; border:1px solid rgba(46,230,197,0.45); background:rgba(46,230,197,0.08);
  color:{TEXT}; font-family:'Space Grotesk',sans-serif; font-weight:600; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ border-color:{NEON_TEAL}; box-shadow:0 0 18px rgba(46,230,197,0.25); color:#fff; }}
.foot {{ color:{MUTED}; font-size:12px; margin-top:28px; border-top:1px solid rgba(255,255,255,0.06); padding-top:12px; }}
</style>
"""

# ---- Plotly template ----
pio.templates["midnight"] = go.layout.Template(layout=go.Layout(
    font=dict(family="Inter, sans-serif", color=TEXT_2, size=12.5),
    title=dict(font=dict(family="Space Grotesk, sans-serif", color=TEXT, size=16), x=0.005, xanchor="left", xref="container",
               y=0.985, yanchor="top", yref="container"),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=SERIES,
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False, linecolor="rgba(255,255,255,0.14)", tickcolor="rgba(255,255,255,0.14)",
               title=dict(font=dict(color=TEXT_2))),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False, linecolor="rgba(255,255,255,0.14)", title=dict(font=dict(color=TEXT_2))),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_2, size=11.5)),
    hoverlabel=dict(bgcolor=SURFACE_2, bordercolor=NEON_TEAL, font=dict(family="JetBrains Mono, monospace", color=TEXT, size=12)),
    margin=dict(l=10, r=10, t=78, b=10), hovermode="closest",
))
pio.templates.default = "midnight"


# ---- components ----
def kpi(label, value, sub="", accent=NEON_TEAL, chip=None):
    chip_html = f'<div style="margin-top:10px">{chip}</div>' if chip else ""
    return (f'<div class="glass" style="--accent:{accent}"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div>{chip_html}</div>')


def chip(text, kind=""):
    icon = {"ok": "✓", "bad": "✕", "warn": "!"}.get(kind, "")
    return f'<span class="chip {kind}">{icon + " " if icon else ""}{text}</span>'


def section(num, title, note=""):
    n = f'<div class="section-note">{note}</div>' if note else ""
    return f'<div class="section"><span class="section-num">{num}</span><span class="section-title">{title}</span></div>{n}'


def glow_line(fig, x, y, color, name, width=2.5, dash=None, hover="%{y:.3f}"):
    """Line with a soft neon glow (an extra wide, faint copy underneath that is hidden from hover and legend)."""
    fig.add_scatter(x=x, y=y, mode="lines", line=dict(color=color, width=10), opacity=0.12, hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=x, y=y, mode="lines+markers", name=name, line=dict(color=color, width=width, dash=dash),
                    marker=dict(size=8, color=color, line=dict(color=SURFACE, width=2)), hovertemplate=f"{name}: {hover}<extra></extra>")
    return fig
