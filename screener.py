import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators needed for CAN SLIM + VCP screening."""
    df = df.copy()

    # Moving averages
    df['dma20']  = df['Close'].rolling(20).mean()
    df['dma50']  = df['Close'].rolling(50).mean()
    df['dma150'] = df['Close'].rolling(150).mean()
    df['dma200'] = df['Close'].rolling(200).mean()

    # Volume MA
    df['vol_ma20'] = df['Volume'].rolling(20).mean()
    df['vol_ma50'] = df['Volume'].rolling(50).mean()

    # 52-week high/low
    df['high_52w'] = df['High'].rolling(252).max()
    df['low_52w']  = df['Low'].rolling(252).min()

    # ATR (for stop loss)
    high_low  = df['High'] - df['Low']
    high_prev = (df['High'] - df['Close'].shift(1)).abs()
    low_prev  = (df['Low']  - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
    df['atr14'] = tr.rolling(14).mean()

    # Price range (for VCP contraction measurement)
    df['range'] = df['High'] - df['Low']
    df['range_pct'] = df['range'] / df['Close'] * 100

    return df


# ─────────────────────────────────────────────────────────────────────────────
# MINERVINI TREND TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────

def check_trend_template(df: pd.DataFrame):
    """
    Minervini's 8-point Trend Template (SEPA).
    Returns (bool, dict_of_checks).
    """
    if len(df) < 200:
        return False, {}

    last = df.iloc[-1]
    price   = last['Close']
    dma50   = last['dma50']
    dma150  = last['dma150']
    dma200  = last['dma200']
    high52  = last['high_52w']
    low52   = last['low_52w']

    # 200 DMA trending up (compare to 4 weeks ago)
    dma200_4w = df['dma200'].iloc[-21] if len(df) >= 221 else np.nan
    dma200_rising = bool(dma200 > dma200_4w) if not np.isnan(dma200_4w) else False

    checks = {
        'above_200':     bool(price > dma200),
        'dma200_rising': dma200_rising,
        'above_150':     bool(price > dma150),
        'above_50':      bool(price > dma50),
        'ma_stack':      bool(dma50 > dma150 and dma150 > dma200),
        'near_52w_high': bool(price >= 0.70 * high52),   # within 30% of 52w high
        'above_52w_low': bool(price >= 1.25 * low52),    # 25% above 52w low
        'above_20':      bool(price > last['dma20']),
    }

    # Must pass 7/8 (near_52w_high + above_200 + dma200_rising are non-negotiable)
    must_pass = ['above_200', 'dma200_rising', 'above_150', 'ma_stack', 'near_52w_high']
    non_neg_ok = all(checks[k] for k in must_pass)
    total_ok = sum(checks.values()) >= 6

    return bool(non_neg_ok and total_ok), checks


# ─────────────────────────────────────────────────────────────────────────────
# VCP DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

def detect_vcp(df: pd.DataFrame, min_contractions: int = 2,
               lookback_days: int = 60, max_depth_pct: float = 35.0):
    """
    Detect Volatility Contraction Pattern.

    Algorithm:
    1. In the lookback window, find local swing highs and lows.
    2. Check that successive price contractions are narrowing.
    3. Check that volume is declining in each contraction.
    4. The pivot is the most recent swing high (breakout point).

    Returns: (bool, dict with details)
    """
    if len(df) < 60:
        return False, {}

    window = df.tail(lookback_days).copy()

    # ── Find swing highs / lows using a simple rolling window ───────────────
    def find_swings(series, window=5):
        highs, lows = [], []
        arr = series.values
        for i in range(window, len(arr) - window):
            if arr[i] == max(arr[i-window:i+window+1]):
                highs.append((i, arr[i]))
            if arr[i] == min(arr[i-window:i+window+1]):
                lows.append((i, arr[i]))
        return highs, lows

    swing_highs, swing_lows = find_swings(window['High'], window=4)
    vol_arr = window['Volume'].values

    if len(swing_highs) < 2 or len(swing_lows) < 1:
        return False, {}

    # ── Measure contractions between successive swing highs ──────────────────
    contractions = []
    for i in range(1, len(swing_highs)):
        prev_h_idx, prev_h = swing_highs[i-1]
        curr_h_idx, curr_h = swing_highs[i]

        # Find lowest point between the two highs
        segment = window['Low'].iloc[prev_h_idx:curr_h_idx+1]
        if len(segment) == 0:
            continue
        low_in_seg = segment.min()

        depth_pct = (prev_h - low_in_seg) / prev_h * 100
        range_pct = (curr_h - low_in_seg) / curr_h * 100

        # Volume: avg in this segment vs previous
        seg_vol = vol_arr[prev_h_idx:curr_h_idx+1].mean()

        contractions.append({
            'high': curr_h,
            'low':  low_in_seg,
            'depth_pct': depth_pct,
            'range_pct': range_pct,
            'avg_vol': seg_vol,
            'idx': curr_h_idx
        })

    if len(contractions) < min_contractions:
        return False, {}

    # ── Validate: each contraction should be tighter than previous ──────────
    depths = [c['depth_pct'] for c in contractions]
    vols   = [c['avg_vol']   for c in contractions]

    depth_contracting = all(depths[i] < depths[i-1] for i in range(1, len(depths)))
    volume_contracting = all(vols[i] < vols[i-1] * 1.05 for i in range(1, len(vols)))
    max_depth_ok = max(depths) <= max_depth_pct

    valid = depth_contracting and volume_contracting and max_depth_ok

    # ── Pivot details ─────────────────────────────────────────────────────────
    last_contraction = contractions[-1]
    pivot_price = last_contraction['high']
    current_price = float(df['Close'].iloc[-1])
    near_pivot = current_price >= pivot_price * 0.98  # within 2% of pivot

    # Current volume drying up (below 50-day avg) — ideal VCP sign
    curr_vol = float(df['Volume'].iloc[-1])
    vol_dry  = curr_vol < float(df['vol_ma50'].iloc[-1]) * 0.8

    details = {
        'contractions': len(contractions),
        'depths': [round(d, 1) for d in depths],
        'depth_contracting': depth_contracting,
        'volume_contracting': volume_contracting,
        'pivot_price': round(pivot_price, 2),
        'near_pivot': near_pivot,
        'vol_dry': vol_dry,
        'max_depth': round(max(depths), 1),
    }

    return valid, details


# ─────────────────────────────────────────────────────────────────────────────
# RELATIVE STRENGTH RANK
# ─────────────────────────────────────────────────────────────────────────────

def compute_rs_rank(df: pd.DataFrame) -> int:
    """
    Compute RS rank (0-100) as 12-month price performance.
    In live use, this would be ranked against the full universe.
    Here we compute raw 12M return and map it to 0-100 approximately.
    """
    if len(df) < 252:
        lookback = len(df) - 1
    else:
        lookback = 251

    try:
        ret_12m = (float(df['Close'].iloc[-1]) / float(df['Close'].iloc[-lookback]) - 1) * 100
        # Rough mapping: -50% → 1, 0% → 50, +100% → 90, +200% → 99
        # Using a sigmoid-like mapping
        rs = int(50 + 25 * np.tanh(ret_12m / 60))
        return max(1, min(99, rs))
    except:
        return 50


# ─────────────────────────────────────────────────────────────────────────────
# CAN SLIM COMPOSITE SCORE (Technical Proxy)
# ─────────────────────────────────────────────────────────────────────────────

def get_canslim_score(df: pd.DataFrame, rs_rank: int,
                      proximity_52w: int = 30) -> int:
    """
    Technical CAN SLIM proxy score (0-100).
    True CAN SLIM needs fundamental data (EPS, Sales) — those can be
    overlaid from a Screener.in export.

    Scoring breakdown:
    - RS Rank (35 pts)
    - MA Alignment / Trend Template (30 pts)
    - Volume behavior (20 pts)
    - 52w proximity / N factor (15 pts)
    """
    if len(df) < 50:
        return 0

    score = 0
    last = df.iloc[-1]

    # RS component (35 pts)
    score += int(rs_rank * 0.35)

    # MA alignment (30 pts)
    price   = last['Close']
    dma50   = last['dma50']
    dma150  = last['dma150']
    dma200  = last['dma200']

    if price > dma50:   score += 8
    if price > dma150:  score += 7
    if price > dma200:  score += 8
    if dma50 > dma150:  score += 4
    if dma150 > dma200: score += 3

    # Volume (20 pts)
    recent_vol  = df['Volume'].tail(10).mean()
    baseline_vol = df['vol_ma50'].iloc[-1]
    if baseline_vol > 0:
        vol_ratio = recent_vol / baseline_vol
        if vol_ratio > 1.5:   score += 20
        elif vol_ratio > 1.2: score += 14
        elif vol_ratio > 0.9: score += 8

    # 52w proximity (15 pts)
    if last['high_52w'] > 0:
        pct_from_high = (last['high_52w'] - price) / last['high_52w'] * 100
        if pct_from_high <= 5:  score += 15
        elif pct_from_high <= 10: score += 12
        elif pct_from_high <= 20: score += 8
        elif pct_from_high <= 30: score += 4

    return min(100, max(0, score))


# ─────────────────────────────────────────────────────────────────────────────
# STOP LOSS CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_stop_loss(df: pd.DataFrame, vcp_details: dict) -> float | None:
    """
    Compute Minervini-style stop loss.

    Rules:
    1. Below the last VCP pivot low
    2. Max 7-8% below current price
    3. Also check 1× ATR below the pivot low
    """
    if not vcp_details or 'pivot_price' not in vcp_details:
        return None

    try:
        current = float(df['Close'].iloc[-1])
        atr = float(df['atr14'].iloc[-1])

        # VCP pivot low: use lowest in last 10 days of base
        recent_low = float(df['Low'].tail(15).min())

        # Stop just below recent pivot low
        sl_pivot = recent_low * 0.99   # 1% buffer below pivot low

        # Hard max: 8% below entry
        sl_max = current * 0.92

        # Use the higher of the two (tighter stop)
        sl = max(sl_pivot, sl_max)

        # ATR-based sanity check (not too tight)
        sl_atr = current - (1.5 * atr)
        sl = max(sl, sl_atr * 0.995)

        return round(min(sl, current * 0.95), 2)  # never more than 5% away
    except:
        return None
