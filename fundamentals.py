"""
fundamentals.py

Parses Screener.in CSV exports and computes CAN SLIM fundamental scores.

How to export from Screener.in (free account):
1. Go to https://www.screener.in/explore/
2. Run this query (copy-paste):
   EPS growth last quarter > 20 AND
   Sales growth last quarter > 15 AND
   Return on equity > 15 AND
   Debt to equity < 1
3. Click "Export to Excel/CSV" (top right)
4. Upload that CSV here

Alternatively, for a broader universe without pre-filtering:
1. Open any stock page e.g. https://www.screener.in/company/RELIANCE/
2. Click "Export" under the financial tables
3. Repeat for your watchlist (or use Screener's bulk export for paid users)

The parser handles both formats.
"""

import pandas as pd
import numpy as np
import io
import re


# ── Expected column aliases (Screener.in varies column names) ─────────────────
COLUMN_ALIASES = {
    # Ticker / Name
    'ticker':          ['Symbol', 'NSE Symbol', 'BSE Code', 'Ticker', 'symbol', 'SYMBOL'],
    'name':            ['Name', 'Company Name', 'Company', 'name'],

    # CAN SLIM — C: Current quarterly EPS growth
    'eps_qoq':         ['EPS growth last quarter', 'EPS Growth QoQ', 'EPS Qtr Growth',
                        'Earnings growth last quarter', 'EPS growth quarter'],

    # CAN SLIM — A: Annual EPS growth (3-year CAGR proxy)
    'eps_3y':          ['EPS growth 3Years', 'EPS growth 3 years', 'EPS CAGR 3Y',
                        'Profit growth 3Years', 'Profit CAGR 3Y', 'EPS 3yr growth'],

    # Sales growth (S factor)
    'sales_qoq':       ['Sales growth last quarter', 'Revenue growth QoQ',
                        'Sales Growth QoQ', 'Revenue growth last quarter'],
    'sales_3y':        ['Sales growth 3Years', 'Revenue growth 3Years', 'Sales CAGR 3Y'],

    # Return on Equity (management quality)
    'roe':             ['Return on equity', 'ROE', 'Return On Equity', 'RoE'],

    # Debt to equity
    'debt_equity':     ['Debt to equity', 'D/E', 'Debt / Equity', 'DebtToEquity'],

    # Promoter holding (I factor proxy)
    'promoter_hold':   ['Promoter holding', 'Promoter Holding %', 'Promoter %'],

    # FII holding change
    'fii_change':      ['Change in FII Holding', 'FII Holding Change', 'FII holding change'],

    # Market cap
    'mktcap_cr':       ['Market Capitalization', 'Market Cap', 'Mcap', 'Market Cap (Cr)',
                        'Market Capitalization (Cr)'],

    # Current price (for cross-reference)
    'price':           ['Current Price', 'Price', 'CMP', 'LTP'],
}


def _find_col(df: pd.DataFrame, aliases: list) -> str | None:
    """Find first matching column from a list of aliases."""
    for alias in aliases:
        for col in df.columns:
            if alias.lower().strip() == col.lower().strip():
                return col
    # Fuzzy match
    for alias in aliases:
        for col in df.columns:
            if alias.lower() in col.lower():
                return col
    return None


def _clean_numeric(series: pd.Series) -> pd.Series:
    """Clean percentage strings and convert to float."""
    def _clean(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().replace('%', '').replace(',', '').replace('₹', '')
        s = s.replace('Cr', '').replace('cr', '').strip()
        try:
            return float(s)
        except:
            return np.nan
    return series.apply(_clean)


def parse_screener_csv(file_content: bytes | str) -> pd.DataFrame | None:
    """
    Parse a Screener.in CSV export.
    Returns a DataFrame with standardised columns, indexed by ticker symbol.
    Returns None if parsing fails.
    """
    try:
        if isinstance(file_content, bytes):
            content = file_content.decode('utf-8', errors='replace')
        else:
            content = file_content

        # Screener sometimes exports with BOM
        content = content.lstrip('\ufeff')

        df = pd.read_csv(io.StringIO(content))
        df.columns = df.columns.str.strip()

        if df.empty or len(df.columns) < 3:
            return None

        out = pd.DataFrame()

        # Map each standard field
        for std_col, aliases in COLUMN_ALIASES.items():
            src_col = _find_col(df, aliases)
            if src_col:
                out[std_col] = _clean_numeric(df[src_col])
            else:
                out[std_col] = np.nan

        # Ticker — keep as string
        ticker_col = _find_col(df, COLUMN_ALIASES['ticker'])
        if ticker_col:
            out['ticker_raw'] = df[ticker_col].astype(str).str.strip().str.upper()
        else:
            # Try to infer from Name
            name_col = _find_col(df, COLUMN_ALIASES['name'])
            if name_col:
                out['ticker_raw'] = df[name_col].astype(str).str.strip().str.upper()
            else:
                return None

        # Name
        name_col = _find_col(df, COLUMN_ALIASES['name'])
        if name_col:
            out['company_name'] = df[name_col].astype(str).str.strip()
        else:
            out['company_name'] = out['ticker_raw']

        out = out.dropna(subset=['ticker_raw'])
        out = out.set_index('ticker_raw')
        out = out[~out.index.duplicated(keep='first')]

        return out

    except Exception as e:
        return None


# ── CAN SLIM Fundamental Score ────────────────────────────────────────────────

def compute_fundamental_score(row: pd.Series) -> dict:
    """
    Score the CAN SLIM fundamental factors (0–100).

    Breakdown:
    C — Current quarterly EPS growth   (25 pts)
    A — Annual EPS growth 3Y           (20 pts)
    N — New highs proxy (from tech)    (skipped — handled technically)
    S — Sales growth QoQ               (15 pts)
    L — Leader proxy (RS rank)         (from tech layer)
    I — Institutional interest (FII)   (10 pts)
    M — Market direction               (from tech layer)
    Quality — ROE + D/E                (15 pts)
    Promoter holding stability         (15 pts)
    """
    score = 0
    breakdown = {}

    # ── C: Current EPS growth QoQ (25 pts) ───────────────────────────────────
    eps_qoq = row.get('eps_qoq', np.nan)
    if not np.isnan(eps_qoq):
        if eps_qoq >= 50:    c_pts = 25
        elif eps_qoq >= 35:  c_pts = 20
        elif eps_qoq >= 25:  c_pts = 16
        elif eps_qoq >= 15:  c_pts = 10
        elif eps_qoq >= 0:   c_pts = 5
        else:                c_pts = 0
    else:
        c_pts = 0
    score += c_pts
    breakdown['C_eps_qoq'] = {'value': eps_qoq, 'score': c_pts, 'max': 25}

    # ── A: Annual EPS growth 3Y (20 pts) ─────────────────────────────────────
    eps_3y = row.get('eps_3y', np.nan)
    if not np.isnan(eps_3y):
        if eps_3y >= 35:     a_pts = 20
        elif eps_3y >= 25:   a_pts = 16
        elif eps_3y >= 15:   a_pts = 10
        elif eps_3y >= 5:    a_pts = 5
        else:                a_pts = 0
    else:
        a_pts = 0
    score += a_pts
    breakdown['A_eps_3y'] = {'value': eps_3y, 'score': a_pts, 'max': 20}

    # ── S: Sales growth QoQ (15 pts) ─────────────────────────────────────────
    sales_qoq = row.get('sales_qoq', np.nan)
    if not np.isnan(sales_qoq):
        if sales_qoq >= 30:   s_pts = 15
        elif sales_qoq >= 20: s_pts = 12
        elif sales_qoq >= 10: s_pts = 8
        elif sales_qoq >= 0:  s_pts = 4
        else:                 s_pts = 0
    else:
        s_pts = 0
    score += s_pts
    breakdown['S_sales_qoq'] = {'value': sales_qoq, 'score': s_pts, 'max': 15}

    # ── I: FII holding change (10 pts) ────────────────────────────────────────
    fii_chg = row.get('fii_change', np.nan)
    if not np.isnan(fii_chg):
        if fii_chg >= 2:     i_pts = 10
        elif fii_chg >= 0.5: i_pts = 7
        elif fii_chg >= 0:   i_pts = 4
        else:                i_pts = 0
    else:
        i_pts = 0
    score += i_pts
    breakdown['I_fii_change'] = {'value': fii_chg, 'score': i_pts, 'max': 10}

    # ── Quality: ROE (10 pts) + D/E (5 pts) ──────────────────────────────────
    roe = row.get('roe', np.nan)
    if not np.isnan(roe):
        if roe >= 25:    roe_pts = 10
        elif roe >= 18:  roe_pts = 8
        elif roe >= 12:  roe_pts = 5
        else:            roe_pts = 2
    else:
        roe_pts = 0

    de = row.get('debt_equity', np.nan)
    if not np.isnan(de):
        if de <= 0.3:    de_pts = 5
        elif de <= 0.7:  de_pts = 3
        elif de <= 1.0:  de_pts = 1
        else:            de_pts = 0
    else:
        de_pts = 0

    score += roe_pts + de_pts
    breakdown['Quality_roe']  = {'value': roe, 'score': roe_pts, 'max': 10}
    breakdown['Quality_de']   = {'value': de,  'score': de_pts,  'max': 5}

    # ── Promoter holding (15 pts) ─────────────────────────────────────────────
    promo = row.get('promoter_hold', np.nan)
    if not np.isnan(promo):
        if promo >= 60:   p_pts = 15
        elif promo >= 45: p_pts = 10
        elif promo >= 30: p_pts = 6
        else:             p_pts = 2
    else:
        p_pts = 0
    score += p_pts
    breakdown['Promoter'] = {'value': promo, 'score': p_pts, 'max': 15}

    return {
        'fundamental_score': min(100, score),
        'breakdown': breakdown,
        'data_completeness': sum(1 for k, v in breakdown.items() if not np.isnan(v['value'])) / len(breakdown),
    }


def merge_fundamentals(results: list, fund_df: pd.DataFrame) -> list:
    """
    Merge fundamental scores into the technical screening results.
    Matches on ticker symbol (strips .NS / .BO suffix for matching).
    """
    if fund_df is None or fund_df.empty:
        return results

    # Build lookup: raw symbol → fundamental row
    fund_lookup = {}
    for idx, row in fund_df.iterrows():
        # Try multiple key formats
        clean = str(idx).upper().replace('.NS', '').replace('.BO', '').strip()
        fund_lookup[clean] = row

    for r in results:
        ticker_clean = r['ticker'].upper().replace('.NS', '').replace('.BO', '').strip()
        if ticker_clean in fund_lookup:
            frow = fund_lookup[ticker_clean]
            fdata = compute_fundamental_score(frow)
            r['fundamental_score']    = fdata['fundamental_score']
            r['fundamental_breakdown'] = fdata['breakdown']
            r['data_completeness']    = fdata['data_completeness']
            r['company_name']         = frow.get('company_name', r['name'])

            # Update eps_qoq for display
            r['eps_qoq']   = frow.get('eps_qoq', np.nan)
            r['eps_3y']    = frow.get('eps_3y', np.nan)
            r['sales_qoq'] = frow.get('sales_qoq', np.nan)
            r['roe']       = frow.get('roe', np.nan)
            r['fii_change'] = frow.get('fii_change', np.nan)

            # Blended score: 50% technical + 50% fundamental
            r['blended_score'] = int(0.5 * r['canslim_score'] + 0.5 * r['fundamental_score'])
        else:
            r['fundamental_score']    = None
            r['fundamental_breakdown'] = {}
            r['data_completeness']    = 0
            r['blended_score']        = r['canslim_score']

    return results


# ── Screener.in Query Generator ───────────────────────────────────────────────

SCREENER_QUERIES = {
    'sme': """
Market Capitalization < 500 AND
EPS growth last quarter > 15 AND
Sales growth last quarter > 10 AND
Return on equity > 12
""".strip(),

    'small': """
Market Capitalization < 25000 AND
Market Capitalization > 500 AND
EPS growth last quarter > 20 AND
Sales growth last quarter > 15 AND
Return on equity > 15 AND
Debt to equity < 1
""".strip(),

    'mid': """
Market Capitalization < 50000 AND
Market Capitalization > 25000 AND
EPS growth last quarter > 20 AND
Sales growth last quarter > 15 AND
Return on equity > 15
""".strip(),

    'large': """
Market Capitalization > 50000 AND
EPS growth last quarter > 15 AND
Sales growth last quarter > 10 AND
Return on equity > 15
""".strip(),
}
