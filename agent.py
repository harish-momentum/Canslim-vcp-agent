"""
agent.py  —  CAN SLIM + VCP Agentic Screener

This agent:
1. Fetches live price data for all NSE categories
2. Runs the full CAN SLIM + VCP screening pipeline
3. Picks top 3 per segment
4. Calls Claude API to write a conviction thesis for each pick
5. Saves results to results/latest.json for the dashboard to read
6. Runs on a schedule (daily at 8:00 AM IST by default)
"""

import os
import sys
import json
import time
import logging
import schedule
import threading
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import requests
import warnings
warnings.filterwarnings('ignore')

from screener import (
    compute_indicators, check_trend_template,
    detect_vcp, compute_rs_rank,
    get_canslim_score, compute_stop_loss,
)
from universe import get_universe

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_DIR   = Path(__file__).parent / "results"
RESULTS_FILE  = RESULTS_DIR / "latest.json"
LOG_FILE      = RESULTS_DIR / "agent.log"
RESULTS_DIR.mkdir(exist_ok=True)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
TOP_N             = 3        # picks per segment
CATEGORIES        = ['sme', 'small', 'mid', 'large']
CAT_LABELS = {
    'sme':   'SME Exchange',
    'small': 'Small Cap (≤₹25k Cr)',
    'mid':   'Mid Cap (₹25–50k Cr)',
    'large': 'Large Cap (₹50k+ Cr)',
}

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-7s  %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("agent")

# shared state for dashboard
_agent_state = {
    "status":       "idle",       # idle | running | done | error
    "progress":     "",
    "last_run":     None,
    "next_run":     None,
    "results":      None,
    "error":        None,
}
_state_lock = threading.Lock()

def set_state(**kwargs):
    with _state_lock:
        _agent_state.update(kwargs)

def get_state():
    with _state_lock:
        return dict(_agent_state)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA + SCREENING
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_and_screen(category: str) -> list:
    """Fetch price data and run the full screening pipeline for one category."""
    tickers = get_universe(category)
    end     = datetime.today()
    start   = end - timedelta(days=400)
    results = []

    log.info(f"  [{category.upper()}] Scanning {len(tickers)} tickers...")

    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end,
                             progress=False, auto_adjust=True)
            if df is None or len(df) < 60:
                continue

            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df = df[['Open','High','Low','Close','Volume']].dropna()
            if len(df) < 60:
                continue

            df            = compute_indicators(df)
            trend_ok, td  = check_trend_template(df)
            vcp_ok, vd    = detect_vcp(df, min_contractions=2)

            if not (trend_ok and vcp_ok):
                continue

            price     = float(df['Close'].iloc[-1])
            vol_ratio = float(df['Volume'].iloc[-1] / df['vol_ma20'].iloc[-1]) \
                        if df['vol_ma20'].iloc[-1] > 0 else 1.0
            rs        = compute_rs_rank(df)
            score     = get_canslim_score(df, rs)
            sl        = compute_stop_loss(df, vd)
            risk_pct  = round((price - sl) / price * 100, 1) if sl else None

            # Gather chart data for AI context (last 60 days OHLCV as dict)
            chart_tail = df.tail(60)[['Open','High','Low','Close','Volume']].copy()
            chart_tail.index = chart_tail.index.strftime('%Y-%m-%d')

            results.append({
                'ticker':       ticker,
                'name':         ticker.replace('.NS','').replace('.BO',''),
                'price':        round(price, 2),
                'rs_rank':      rs,
                'score':        score,
                'vol_ratio':    round(vol_ratio, 2),
                'stop_loss':    sl,
                'risk_pct':     risk_pct,
                'pivot':        vd.get('pivot_price'),
                'contractions': vd.get('contractions'),
                'depths':       vd.get('depths', []),
                'max_depth':    vd.get('max_depth'),
                'vol_dry':      vd.get('vol_dry'),
                'near_pivot':   vd.get('near_pivot'),
                'trend_checks': td,
                'dma20':  round(float(df['dma20'].iloc[-1]), 2),
                'dma50':  round(float(df['dma50'].iloc[-1]), 2),
                'dma150': round(float(df['dma150'].iloc[-1]), 2),
                'dma200': round(float(df['dma200'].iloc[-1]), 2),
                'high_52w': round(float(df['high_52w'].iloc[-1]), 2),
                'low_52w':  round(float(df['low_52w'].iloc[-1]), 2),
                'pct_from_52w_high': round(
                    (float(df['high_52w'].iloc[-1]) - price) / float(df['high_52w'].iloc[-1]) * 100, 1),
                # last 20 close prices for AI context
                'recent_closes': [round(x,2) for x in df['Close'].tail(20).tolist()],
                'recent_volumes': [int(x) for x in df['Volume'].tail(20).tolist()],
            })
            time.sleep(0.04)

        except Exception as e:
            log.debug(f"  Skip {ticker}: {e}")
            continue

    # Sort by score, take top N
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:TOP_N]


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — CLAUDE AI ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def build_analysis_prompt(pick: dict, category: str) -> str:
    td = pick['trend_checks']
    checks_str = "\n".join([
        f"  - {'✓' if v else '✗'} {k.replace('_',' ').title()}"
        for k, v in td.items()
    ])

    depths_str = " → ".join([f"{d}%" for d in pick['depths']]) if pick['depths'] else "N/A"

    return f"""You are a senior equity analyst specialising in momentum investing using the CAN SLIM methodology and Mark Minervini's SEPA / VCP framework.

Analyse this Indian stock setup and write a concise but high-conviction trading note.

═══ STOCK DATA ═══
Ticker:        {pick['ticker']}
Category:      {CAT_LABELS.get(category, category)}
Current Price: ₹{pick['price']}
52w High:      ₹{pick['high_52w']}  ({pick['pct_from_52w_high']}% below 52w high)
52w Low:       ₹{pick['low_52w']}

Moving Averages:
  20 DMA:  ₹{pick['dma20']}
  50 DMA:  ₹{pick['dma50']}
  150 DMA: ₹{pick['dma150']}
  200 DMA: ₹{pick['dma200']}

Technical Score:    {pick['score']}/100
Relative Strength:  {pick['rs_rank']}/100
Volume Ratio:       {pick['vol_ratio']}x (vs 20d avg)

═══ VCP PATTERN ═══
Contractions detected: {pick['contractions']}
Contraction depths:    {depths_str}  (each should be tighter — confirms VCP)
Max contraction depth: {pick['max_depth']}%
Volume drying up:      {'Yes ✓' if pick['vol_dry'] else 'No'}
Near pivot:            {'Yes ✓' if pick['near_pivot'] else 'No'}
Pivot price:           ₹{pick['pivot']}

═══ TREND TEMPLATE (Minervini SEPA) ═══
{checks_str}

═══ RISK / REWARD ═══
Suggested Stop Loss: ₹{pick['stop_loss']}
Risk from current:   {pick['risk_pct']}%

═══ RECENT PRICE ACTION (last 20 sessions) ═══
Closes: {pick['recent_closes']}

═══ YOUR TASK ═══
Write a structured trading note with these exact sections:

**SETUP QUALITY** — Rate this setup: Excellent / Good / Borderline. One sentence explaining why.

**VCP THESIS** — In 2-3 sentences, explain what the VCP contractions tell you about institutional behaviour and why this stock is coiling for a breakout.

**KEY CATALYST TO WATCH** — What price action, volume event, or fundamental trigger would confirm this setup is activating? Be specific (e.g. "breakout above ₹X on 2x average volume").

**ENTRY ZONE** — Precise entry range with rationale.

**STOP LOSS LOGIC** — Explain the stop loss level in plain English. Why is this the right level?

**REWARD TARGET** — Based on the base depth and typical VCP measured move, what is a reasonable 3-6 month price target? Show your working.

**RISK FACTORS** — 2 key risks specific to this setup (not generic market risk).

Keep the total response under 350 words. Be direct. No disclaimers."""


def call_claude_api(prompt: str) -> str:
    """Call the Anthropic API and return the analysis text."""
    if not ANTHROPIC_API_KEY:
        return "_No ANTHROPIC_API_KEY set — add it to your .env file to enable AI analysis._"

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}],
                "system": "You are a senior equity analyst specialising in momentum investing. Be concise and direct.",
            },
            timeout=45,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]
    except Exception as e:
        log.error(f"Claude API error: {e}")
        return f"_AI analysis unavailable: {e}_"


def analyse_pick(pick: dict, category: str) -> str:
    """Build prompt and get Claude's analysis for one pick."""
    log.info(f"    → Calling Claude for {pick['ticker']}...")
    prompt   = build_analysis_prompt(pick, category)
    analysis = call_claude_api(prompt)
    return analysis


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — MARKET CONTEXT
# ═══════════════════════════════════════════════════════════════════════════════

def get_market_context() -> dict:
    """Fetch Nifty 50 status for market breadth context."""
    try:
        nifty = yf.download("^NSEI", period="1y", progress=False, auto_adjust=True)
        nifty.columns = [c[0] if isinstance(c, tuple) else c for c in nifty.columns]
        close  = nifty['Close'].dropna()
        price  = float(close.iloc[-1])
        dma200 = float(close.rolling(200).mean().iloc[-1])
        dma50  = float(close.rolling(50).mean().iloc[-1])
        chg1d  = float((close.iloc[-1] / close.iloc[-2] - 1) * 100)
        chg1m  = float((close.iloc[-1] / close.iloc[-22] - 1) * 100)
        return {
            'price':       round(price, 0),
            'dma200':      round(dma200, 0),
            'dma50':       round(dma50, 0),
            'above_200':   price > dma200,
            'above_50':    price > dma50,
            'chg_1d':      round(chg1d, 2),
            'chg_1m':      round(chg1m, 2),
            'trend':       'BULL' if price > dma200 else 'BEAR',
        }
    except Exception as e:
        log.warning(f"Could not fetch Nifty: {e}")
        return {}


def get_market_comment(ctx: dict) -> str:
    """Ask Claude for a one-paragraph market context comment."""
    if not ANTHROPIC_API_KEY or not ctx:
        return ""
    prompt = f"""You are a market strategist. Write a single tight paragraph (max 80 words) 
on current Indian market conditions for a momentum trader using CAN SLIM.

Nifty 50: ₹{ctx.get('price')}  
1-day change: {ctx.get('chg_1d')}%  
1-month change: {ctx.get('chg_1m')}%  
vs 200 DMA (₹{ctx.get('dma200')}): {'Above — BULL phase' if ctx.get('above_200') else 'Below — BEAR caution'}  
vs 50 DMA (₹{ctx.get('dma50')}): {'Above' if ctx.get('above_50') else 'Below'}

Comment on: trend health, risk environment, whether CAN SLIM screens should be run aggressively or selectively. 
No disclaimers. Be direct."""
    return call_claude_api(prompt)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RUN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_agent():
    """Full agent run — screen all categories, analyse top picks, save results."""
    run_start = datetime.now()
    log.info("=" * 60)
    log.info(f"AGENT RUN STARTED  {run_start.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    set_state(status="running", error=None, progress="Fetching market data...")

    try:
        # ── Market context ─────────────────────────────────────────────────
        log.info("Step 1/3 — Market context")
        set_state(progress="Analysing market conditions (Nifty 50)...")
        ctx = get_market_context()
        log.info(f"  Nifty: ₹{ctx.get('price')} | Trend: {ctx.get('trend')} | "
                 f"1d: {ctx.get('chg_1d')}%")

        market_comment = get_market_comment(ctx)

        # ── Screen each category ───────────────────────────────────────────
        log.info("Step 2/3 — Screening all segments")
        all_picks = {}
        for cat in CATEGORIES:
            set_state(progress=f"Screening {CAT_LABELS[cat]}...")
            picks = fetch_and_screen(cat)
            log.info(f"  {CAT_LABELS[cat]}: {len(picks)} setups found")
            all_picks[cat] = picks

        # ── AI analysis ───────────────────────────────────────────────────
        log.info("Step 3/3 — AI analysis for top picks")
        for cat in CATEGORIES:
            for i, pick in enumerate(all_picks[cat]):
                set_state(progress=f"AI analysing {pick['ticker']} ({cat})...")
                pick['analysis'] = analyse_pick(pick, cat)
                pick['rank']     = i + 1
                time.sleep(0.5)   # rate limit buffer

        # ── Build output ───────────────────────────────────────────────────
        run_end = datetime.now()
        output = {
            "run_at":         run_start.isoformat(),
            "completed_at":   run_end.isoformat(),
            "duration_secs":  int((run_end - run_start).total_seconds()),
            "market":         ctx,
            "market_comment": market_comment,
            "picks":          all_picks,
            "total_setups":   sum(len(v) for v in all_picks.values()),
        }

        # ── Persist ────────────────────────────────────────────────────────
        RESULTS_FILE.write_text(json.dumps(output, indent=2, default=str))
        log.info(f"Results saved → {RESULTS_FILE}")
        log.info(f"Run complete in {output['duration_secs']}s  |  "
                 f"{output['total_setups']} total setups")

        set_state(
            status="done",
            progress="",
            last_run=run_start.isoformat(),
            results=output,
        )
        return output

    except Exception as e:
        log.exception(f"Agent run failed: {e}")
        set_state(status="error", error=str(e), progress="")
        raise


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEDULER
# ═══════════════════════════════════════════════════════════════════════════════

def start_scheduler(run_time: str = "08:00"):
    """Schedule daily runs and return the scheduler thread."""
    schedule.every().day.at(run_time).do(run_agent)
    next_run = schedule.next_run()
    set_state(next_run=next_run.isoformat() if next_run else None)
    log.info(f"Scheduler active — next run at {run_time} IST daily")

    def _loop():
        while True:
            schedule.run_pending()
            next_r = schedule.next_run()
            set_state(next_run=next_r.isoformat() if next_r else None)
            time.sleep(30)

    t = threading.Thread(target=_loop, daemon=True, name="scheduler")
    t.start()
    return t


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD CACHED RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

def load_cached_results() -> dict | None:
    if RESULTS_FILE.exists():
        try:
            data = json.loads(RESULTS_FILE.read_text())
            set_state(results=data, last_run=data.get('run_at'))
            log.info(f"Loaded cached results from {data.get('run_at', 'unknown')}")
            return data
        except:
            pass
    return None


if __name__ == "__main__":
    # If run directly, do an immediate run + start scheduler
    load_cached_results()
    run_agent()
    start_scheduler("08:00")
    # Keep alive
    while True:
        time.sleep(60)
