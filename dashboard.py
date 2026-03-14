"""
dashboard.py  —  CAN SLIM + VCP Agent Dashboard

Read-only view of the agent's top 3 picks per segment.
The agent runs in a background thread; this dashboard shows:
  - Live agent status
  - Market context + AI comment
  - Top 3 picks per segment with full AI analysis
  - Manual "Run Now" trigger
"""

import streamlit as st
import json
import threading
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from agent import (
    run_agent, start_scheduler, load_cached_results,
    get_state, set_state, RESULTS_FILE, CAT_LABELS,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CAN SLIM Agent · India",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg:      #060a12;
    --surf:    #0c1220;
    --surf2:   #111a2e;
    --border:  #1a2640;
    --accent:  #3b82f6;
    --green:   #10b981;
    --red:     #ef4444;
    --amber:   #f59e0b;
    --purple:  #8b5cf6;
    --cyan:    #06b6d4;
    --text:    #e2e8f0;
    --muted:   #4a6080;
}
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text);
}
.main, .block-container { background: var(--bg); padding: 1.2rem 1.8rem; }

/* Rank badge */
.rank-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 28px; height: 28px; border-radius: 50%;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; font-weight: 600;
    margin-right: 0.5rem;
}
.rank-1 { background: #f59e0b22; color: #f59e0b; border: 1.5px solid #f59e0b66; }
.rank-2 { background: #94a3b822; color: #94a3b8; border: 1.5px solid #94a3b866; }
.rank-3 { background: #92400e22; color: #b45309; border: 1.5px solid #b4530966; }

/* Pick card */
.pick-card {
    background: var(--surf);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.pick-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
}
.pick-card.rank-1-card::before { background: linear-gradient(90deg,#f59e0b,#fbbf24); }
.pick-card.rank-2-card::before { background: linear-gradient(90deg,#64748b,#94a3b8); }
.pick-card.rank-3-card::before { background: linear-gradient(90deg,#92400e,#b45309); }

/* AI analysis text */
.ai-analysis {
    background: var(--surf2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 6px;
    padding: 1rem 1.2rem;
    font-size: 0.85rem;
    line-height: 1.7;
    color: #c8d8f0;
    margin-top: 0.8rem;
}

/* Stat pill */
.stat-pill {
    display: inline-flex; align-items: center; gap: 0.3rem;
    background: var(--surf2); border: 1px solid var(--border);
    border-radius: 20px; padding: 0.2rem 0.7rem;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
    color: var(--text); margin: 0.15rem;
}

/* Status badges */
.status-running { color: #f59e0b; font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; }
.status-done    { color: #10b981; font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; }
.status-error   { color: #ef4444; font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; }
.status-idle    { color: #4a6080; font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; }

/* Market strip */
.market-strip {
    background: var(--surf); border: 1px solid var(--border);
    border-radius: 8px; padding: 0.8rem 1.2rem;
    margin-bottom: 1.2rem;
    display: flex; align-items: center; gap: 2rem; flex-wrap: wrap;
}
.mval { font-family: 'IBM Plex Mono', monospace; font-size: 0.9rem; font-weight: 600; }

/* Section title */
.seg-title {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem;
    letter-spacing: 0.12em; color: var(--muted); text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem; margin: 1.5rem 0 1rem 0;
}

/* Trend check grid */
.checks { display: flex; flex-wrap: wrap; gap: 0.3rem; margin: 0.5rem 0; }
.chk-ok  { background:#10b98120;color:#10b981;border:1px solid #10b98140;
            border-radius:4px;padding:0.15rem 0.5rem;font-size:0.68rem; }
.chk-no  { background:#ef444420;color:#ef4444;border:1px solid #ef444440;
            border-radius:4px;padding:0.15rem 0.5rem;font-size:0.68rem; }

/* Agent info bar */
.agent-bar {
    background: var(--surf); border: 1px solid var(--border);
    border-radius: 8px; padding: 0.7rem 1.2rem;
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 0.8rem; margin-bottom: 1.2rem;
    font-size: 0.8rem;
}

.stButton>button {
    background: linear-gradient(135deg,#3b82f622,#06b6d422);
    border: 1px solid #3b82f6; color: #3b82f6;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem;
    padding: 0.35rem 1rem; border-radius: 5px;
}
.stButton>button:hover { background:#3b82f6; color:#060a12; }

div[data-testid="stSidebarContent"] { background: var(--surf); }
</style>
""", unsafe_allow_html=True)


# ── Init scheduler (once per process) ────────────────────────────────────────
if "scheduler_started" not in st.session_state:
    load_cached_results()
    start_scheduler("08:00")
    st.session_state.scheduler_started = True


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;
            margin-bottom:1rem;padding-bottom:0.8rem;border-bottom:1px solid #1a2640;">
  <div>
    <span style="font-family:'IBM Plex Mono',monospace;font-size:1.25rem;
                 color:#3b82f6;letter-spacing:0.04em;">🤖 CANSLIM AGENT</span>
    <span style="font-size:0.75rem;color:#4a6080;margin-left:1rem;">
      Autonomous · India NSE · Top 3 per Segment
    </span>
  </div>
  <span style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#4a6080;">
    {}</span>
</div>
""".format(datetime.now().strftime("%d %b %Y  %H:%M")), unsafe_allow_html=True)


# ── Live state ────────────────────────────────────────────────────────────────
state   = get_state()
status  = state["status"]
results = state["results"]


# ── Agent status bar ──────────────────────────────────────────────────────────
status_html = {
    "running": '<span class="status-running">⟳ RUNNING</span>',
    "done":    '<span class="status-done">✓ COMPLETE</span>',
    "error":   '<span class="status-error">✗ ERROR</span>',
    "idle":    '<span class="status-idle">○ IDLE</span>',
}.get(status, status)

last_run_str = "Never"
if state.get("last_run"):
    try:
        lr = datetime.fromisoformat(state["last_run"])
        last_run_str = lr.strftime("%d %b  %H:%M")
    except: pass

next_run_str = "—"
if state.get("next_run"):
    try:
        nr = datetime.fromisoformat(state["next_run"])
        next_run_str = nr.strftime("%d %b  %H:%M")
    except: pass

col_status, col_btn = st.columns([5, 1])
with col_status:
    prog = f" · {state['progress']}" if state.get('progress') else ""
    st.markdown(f"""
    <div class="agent-bar">
      <span>Status: {status_html}{prog}</span>
      <span style="color:#4a6080;">Last run: <strong style="color:#94a3b8;">{last_run_str}</strong></span>
      <span style="color:#4a6080;">Next: <strong style="color:#94a3b8;">{next_run_str}</strong> (daily 08:00)</span>
      <span style="color:#4a6080;">Runs automatically every morning</span>
    </div>
    """, unsafe_allow_html=True)

with col_btn:
    if st.button("▶ Run Now", use_container_width=True,
                 disabled=(status == "running")):
        def _bg_run():
            try:
                run_agent()
            except:
                pass
        t = threading.Thread(target=_bg_run, daemon=True)
        t.start()
        st.rerun()

if status == "error" and state.get("error"):
    st.error(f"Last run failed: {state['error']}")

if status == "running":
    st.info(f"⟳  {state.get('progress','Running...')}  — Page will refresh automatically.")
    time.sleep(3)
    st.rerun()


# ── No results yet ────────────────────────────────────────────────────────────
if not results:
    st.markdown("""
    <div style="text-align:center;padding:5rem 2rem;color:#2a3a50;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:3rem;margin-bottom:1rem;">🤖</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:1rem;color:#3b82f666;">
        NO RESULTS YET
      </div>
      <div style="margin-top:0.6rem;font-size:0.85rem;">
        Click <strong>▶ Run Now</strong> to trigger the first scan.<br>
        After that, the agent runs automatically every morning at 08:00.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Market context strip ──────────────────────────────────────────────────────
mkt = results.get("market", {})
if mkt:
    trend_color = "#10b981" if mkt.get("above_200") else "#ef4444"
    trend_label = "BULL ↑" if mkt.get("above_200") else "BEAR ↓"
    chg1d_col   = "#10b981" if mkt.get("chg_1d", 0) >= 0 else "#ef4444"
    chg1m_col   = "#10b981" if mkt.get("chg_1m", 0) >= 0 else "#ef4444"

    st.markdown(f"""
    <div class="market-strip">
      <div>
        <div style="font-size:0.65rem;color:#4a6080;text-transform:uppercase;letter-spacing:0.1em;">Nifty 50</div>
        <div class="mval">₹{mkt.get('price','—'):,}</div>
      </div>
      <div>
        <div style="font-size:0.65rem;color:#4a6080;text-transform:uppercase;letter-spacing:0.1em;">200 DMA</div>
        <div class="mval">₹{mkt.get('dma200','—'):,}</div>
      </div>
      <div>
        <div style="font-size:0.65rem;color:#4a6080;text-transform:uppercase;letter-spacing:0.1em;">Market Trend</div>
        <div class="mval" style="color:{trend_color};">{trend_label}</div>
      </div>
      <div>
        <div style="font-size:0.65rem;color:#4a6080;text-transform:uppercase;letter-spacing:0.1em;">1-Day</div>
        <div class="mval" style="color:{chg1d_col};">{mkt.get('chg_1d',0):+.2f}%</div>
      </div>
      <div>
        <div style="font-size:0.65rem;color:#4a6080;text-transform:uppercase;letter-spacing:0.1em;">1-Month</div>
        <div class="mval" style="color:{chg1m_col};">{mkt.get('chg_1m',0):+.2f}%</div>
      </div>
      <div style="font-size:0.65rem;color:#4a6080;">
        Screened: <strong style="color:#94a3b8;">{results.get('run_at','')[:10]}</strong>
      </div>
    </div>
    """, unsafe_allow_html=True)

# Market AI comment
mc = results.get("market_comment", "")
if mc:
    st.markdown(f"""
    <div style="background:#0c1220;border:1px solid #1a2640;border-left:3px solid #8b5cf6;
                border-radius:6px;padding:0.8rem 1.2rem;font-size:0.82rem;
                line-height:1.7;color:#b8c8e0;margin-bottom:1.2rem;">
      <span style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
                   color:#8b5cf6;text-transform:uppercase;letter-spacing:0.1em;">
        🤖 AI Market Comment
      </span><br>{mc}
    </div>
    """, unsafe_allow_html=True)


# ── PICKS ─────────────────────────────────────────────────────────────────────
RANK_MEDALS = {1: "🥇", 2: "🥈", 3: "🥉"}

def score_color(s):
    if s >= 75: return "#10b981"
    if s >= 55: return "#f59e0b"
    return "#ef4444"

def render_pick(pick: dict, cat: str, rank: int):
    rank_cls = f"rank-{rank}"
    medal    = RANK_MEDALS.get(rank, f"#{rank}")
    sc       = pick.get('score', 0)
    rs       = pick.get('rs_rank', 0)
    sl       = pick.get('stop_loss')
    risk     = pick.get('risk_pct')
    pivot    = pick.get('pivot')
    depths   = pick.get('depths', [])
    td       = pick.get('trend_checks', {})

    # Depth arrows
    depths_str = " → ".join([f"{d}%" for d in depths]) if depths else "—"

    # Trend check pills
    check_labels = {
        'above_200': 'P>200d', 'dma200_rising': '200d↑',
        'above_150': 'P>150d', 'above_50': 'P>50d',
        'ma_stack':  'Stack',  'near_52w_high': 'Near52w',
        'above_52w_low': '25%>52wLow', 'above_20': 'P>20d',
    }
    checks_html = '<div class="checks">' + "".join([
        f'<span class="{"chk-ok" if td.get(k) else "chk-no"}">{v}</span>'
        for k, v in check_labels.items()
    ]) + '</div>'

    with st.expander(f"{medal}  {pick['name']}  ·  ₹{pick['price']}  ·  Score {sc}/100  ·  RS {rs}", expanded=(rank==1)):

        # ── Stat pills row ────────────────────────────────────────────────
        pills = f"""
        <div style="margin-bottom:0.8rem;">
          <span class="stat-pill">Score <strong style="color:{score_color(sc)};margin-left:4px;">{sc}/100</strong></span>
          <span class="stat-pill">RS <strong style="color:#f59e0b;margin-left:4px;">{rs}/100</strong></span>
          <span class="stat-pill">Vol {pick.get('vol_ratio','—')}×</span>
          <span class="stat-pill">VCP {pick.get('contractions','—')} contractions</span>
          <span class="stat-pill">Depths: {depths_str}</span>
          {"<span class='stat-pill' style='color:#10b981;'>Vol Dry ✓</span>" if pick.get('vol_dry') else ""}
          {"<span class='stat-pill' style='color:#10b981;'>Near Pivot ✓</span>" if pick.get('near_pivot') else ""}
        </div>
        """
        st.markdown(pills, unsafe_allow_html=True)

        # ── Metrics row ───────────────────────────────────────────────────
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Price",       f"₹{pick['price']}")
        c2.metric("Pivot",       f"₹{pivot}" if pivot else "—")
        c3.metric("Stop Loss",   f"₹{sl}" if sl else "—")
        c4.metric("Risk %",      f"{risk}%" if risk else "—",
                  delta=f"-{risk}%" if risk else None, delta_color="inverse")
        c5.metric("52w High",    f"₹{pick.get('high_52w','—')}")
        c6.metric("From 52w High", f"{pick.get('pct_from_52w_high','—')}%")

        # MA row
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("20 DMA",  f"₹{pick.get('dma20','—')}")
        m2.metric("50 DMA",  f"₹{pick.get('dma50','—')}")
        m3.metric("150 DMA", f"₹{pick.get('dma150','—')}")
        m4.metric("200 DMA", f"₹{pick.get('dma200','—')}")

        # Trend checks
        st.markdown(checks_html, unsafe_allow_html=True)

        # ── AI Analysis ───────────────────────────────────────────────────
        analysis = pick.get('analysis', '')
        if analysis:
            st.markdown(f"""
            <div class="ai-analysis">
              <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
                          color:#3b82f6;text-transform:uppercase;letter-spacing:0.1em;
                          margin-bottom:0.6rem;">🤖 AI Analysis</div>
              {analysis.replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)

        # ── Mini chart ────────────────────────────────────────────────────
        closes  = pick.get('recent_closes', [])
        volumes = pick.get('recent_volumes', [])
        if closes:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.72, 0.28], vertical_spacing=0.04)
            x = list(range(len(closes)))

            fig.add_trace(go.Scatter(x=x, y=closes,
                                     line=dict(color='#3b82f6', width=1.5),
                                     fill='tozeroy', fillcolor='#3b82f610',
                                     name='Close'), row=1, col=1)

            # Add DMA lines (last N points of each)
            for dma_key, col, w in [('dma20','#f59e0b',1),('dma50','#06b6d4',1),
                                     ('dma200','#ef4444',1.5)]:
                val = pick.get(dma_key)
                if val:
                    fig.add_hline(y=val, line_dash='dot', line_color=col,
                                  line_width=w, row=1, col=1)

            if sl:
                fig.add_hline(y=sl, line_dash='dash', line_color='#ef4444',
                              annotation_text=f"SL ₹{sl}",
                              annotation_font_color='#ef4444', row=1, col=1)
            if pivot:
                fig.add_hline(y=pivot, line_dash='dash', line_color='#10b981',
                              annotation_text=f"Pivot ₹{pivot}",
                              annotation_font_color='#10b981', row=1, col=1)

            if volumes:
                v_colors = ['rgba(16,185,129,0.27)' if i==0 or volumes[i]>=volumes[i-1]
                            else 'rgba(239,68,68,0.27)' for i in range(len(volumes))]
                fig.add_trace(go.Bar(x=x, y=volumes, marker_color=v_colors,
                                     showlegend=False), row=2, col=1)

            fig.update_layout(
                height=280, paper_bgcolor='#060a12', plot_bgcolor='#0c1220',
                font=dict(color='#94a3b8', size=9),
                showlegend=False, margin=dict(l=0,r=0,t=8,b=0),
                xaxis_rangeslider_visible=False,
            )
            fig.update_xaxes(gridcolor='#1a2640', showgrid=True, showticklabels=False)
            fig.update_yaxes(gridcolor='#1a2640', showgrid=True)
            st.plotly_chart(fig, use_container_width=True,
                            config={'displayModeBar': False})


# ── Render all segments ───────────────────────────────────────────────────────
picks_data = results.get("picks", {})

SEG_ICONS = {'sme':'🏭', 'small':'📊', 'mid':'📈', 'large':'🏦'}
SEG_COLOR = {'sme':'#06b6d4','small':'#10b981','mid':'#3b82f6','large':'#8b5cf6'}

for cat, label in CAT_LABELS.items():
    picks = picks_data.get(cat, [])
    icon  = SEG_ICONS[cat]
    color = SEG_COLOR[cat]

    st.markdown(f"""
    <div class="seg-title">
      <span style="color:{color};">{icon}</span>  {label}
      <span style="font-size:0.65rem;color:#4a6080;margin-left:0.8rem;">
        {len(picks)} setup{'s' if len(picks)!=1 else ''} · Top 3 by CAN SLIM Score
      </span>
    </div>
    """, unsafe_allow_html=True)

    if not picks:
        st.markdown("""
        <div style="text-align:center;padding:1.5rem;color:#2a3a50;
                    border:1px dashed #1a2640;border-radius:8px;font-size:0.82rem;">
          No setups passed screening in this segment today
        </div>
        """, unsafe_allow_html=True)
    else:
        for i, pick in enumerate(picks[:3]):
            render_pick(pick, cat, i + 1)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-top:2rem;padding-top:1rem;border-top:1px solid #1a2640;
            display:flex;justify-content:space-between;flex-wrap:wrap;
            font-size:0.72rem;color:#2a3a50;font-family:'IBM Plex Mono',monospace;">
  <span>⟡ CAN SLIM + VCP Agent · India NSE</span>
  <span>Data: Yahoo Finance · AI: Claude · Not investment advice</span>
  <span>Results from: {results.get('run_at','')[:16]}</span>
</div>
""", unsafe_allow_html=True)

# Auto-refresh every 30s while running
if status == "running":
    time.sleep(3)
    st.rerun()
