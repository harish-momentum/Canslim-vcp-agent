"""
Universe definitions for each market cap category.

These are carefully curated NSE ticker lists. yfinance requires .NS suffix
for NSE stocks and .BO for BSE-only stocks.

To keep the free version fast, we use representative curated lists.
For production, replace with a dynamic NSE/BSE stock list fetcher.
"""


# ── NSE SME Exchange – active, liquid SME stocks ─────────────────────────────
SME_TICKERS = [
    "VLEGOV.NS", "VAISHALI.NS", "GANDHAR.NS", "SBCL.NS", "KAYNES.NS",
    "ESCONAVLOGS.NS", "EVOTECH.NS", "SAHYOG.NS", "JAIBALAJI.NS", "IDEAFORGE.NS",
    "VRAJ.NS", "NILE.NS", "MANBA.NS", "INNOKAIZ.NS", "SANSTAR.NS",
    "RNFI.NS", "YATHARTH.NS", "BFINANCE.NS", "ANYA.NS", "KALANA.NS",
    "EXICOM.NS", "CRAYONS.NS", "BROACH.NS", "DRONEACHARYA.NS", "SENCO.NS",
    "SIGNATURE.NS", "KSOLVES.NS", "SKYGOLD.NS", "LLOYDSME.NS", "RAJPUTANA.NS",
    "ECOBOARD.NS", "MITTAL.NS", "GRETEX.NS", "EFPEL.NS", "SAAKSHI.NS",
]

# ── Small Cap Main Board – up to ₹25,000 Cr ──────────────────────────────────
SMALL_CAP_TICKERS = [
    # IT/Tech
    "TANLA.NS", "NEWGEN.NS", "KPITTECH.NS", "ZENSAR.NS", "MASTEK.NS",
    "ROUTE.NS", "INTELLECT.NS", "RATEGAIN.NS", "DATAMATICS.NS", "ZAGGLE.NS",
    # Consumer / Retail
    "LATENTVIEW.NS", "RADICO.NS", "VSTIND.NS", "DOMS.NS", "WESTLIFE.NS",
    "CAMPUS.NS", "BATA.NS", "METRO.NS", "JYOTHY.NS", "CCL.NS",
    # Pharma
    "GRANULES.NS", "AARTI.NS", "GLAND.NS", "SOLARA.NS", "CAPLIN.NS",
    # Capital Goods / Infra
    "GRINDWELL.NS", "KALYANKJIL.NS", "JTEKINDO.NS", "GPIL.NS", "KEI.NS",
    "CRAFTSMAN.NS", "TEXRAIL.NS", "ROLCONS.NS", "SUPRAJIT.NS", "JKPAPER.NS",
    # Financials
    "CREDITACC.NS", "UJJIVAN.NS", "EQUITASBNK.NS", "SURYODAY.NS", "APTUS.NS",
    "KFINTECH.NS", "CAMS.NS", "ANANTRAJ.NS", "HOMEFIRST.NS", "AAVAS.NS",
    # Others
    "HAPPYFORGE.NS", "PNBHOUSING.NS", "JBMA.NS", "INDIGOPNTS.NS", "NUVAMA.NS",
]

# ── Mid Cap – ₹25,000 Cr to ₹50,000 Cr ──────────────────────────────────────
MID_CAP_TICKERS = [
    # IT
    "PERSISTENT.NS", "COFORGE.NS", "LTTS.NS", "MPHASIS.NS", "TATAELXSI.NS",
    # Consumer
    "MARICO.NS", "GODREJCP.NS", "EMAMILTD.NS", "COLPAL.NS", "TATACONSUM.NS",
    # Pharma
    "ALKEM.NS", "TORNTPHARM.NS", "AUROPHARMA.NS", "IPCA.NS", "LAURUSLABS.NS",
    # Industrials
    "CUMMINSIND.NS", "THERMAX.NS", "BHEL.NS", "AIAENG.NS", "GRINDMAST.NS",
    "POLYCAB.NS", "FINOLLEX.NS", "APAR.NS", "TDPOWER.NS", "ELGIEQUIP.NS",
    # Banks / NBFC
    "FEDERALBNK.NS", "KARURVYSYA.NS", "IDFCFIRSTB.NS", "RBLBANK.NS", "CITYUNIONB.NS",
    "CHOLAFIN.NS", "MUTHOOTFIN.NS", "MANAPPURAM.NS", "SUNDARMFIN.NS", "BAJFINANCE.NS",
    # Auto
    "MOTHERSON.NS", "SUNDRMFAST.NS", "TIINDIA.NS", "BALKRISIND.NS", "APOLLOTYRE.NS",
    # Others
    "PHOENIX.NS", "BRIGADE.NS", "SOBHA.NS", "PRESTIGE.NS", "NAUKRI.NS",
]

# ── Large Cap – above ₹50,000 Cr ─────────────────────────────────────────────
LARGE_CAP_TICKERS = [
    # Mega cap
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "WIPRO.NS", "HCLTECH.NS", "AXISBANK.NS", "KOTAKBANK.NS", "SBIN.NS",
    # Consumer
    "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "DABUR.NS", "BRITANNIA.NS",
    "ASIANPAINT.NS", "PIDILITIND.NS", "BERGEPAINT.NS",
    # Auto
    "TATAMOTORS.NS", "M&M.NS", "MARUTI.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS",
    "EICHERMOT.NS", "TVSMOTOR.NS",
    # Pharma
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS",
    # Financials
    "BAJFINANCE.NS", "HDFCLIFE.NS", "SBILIFE.NS", "ICICIPRULI.NS",
    "CHOLAFIN.NS", "MUTHOOTFIN.NS",
    # Infra / Capital Goods
    "LT.NS", "NTPC.NS", "POWERGRID.NS", "ADANIPORTS.NS", "ADANIGREEN.NS",
    "ADANIENT.NS", "SIEMENS.NS", "ABB.NS", "HAVELLS.NS", "POLYCAB.NS",
    # Tech
    "PERSISTENT.NS", "COFORGE.NS", "LTTS.NS",
    # Others
    "TITAN.NS", "ZOMATO.NS", "NYKAA.NS", "PAYTM.NS", "DMART.NS",
    "TATAPOWER.NS", "COALINDIA.NS", "ONGC.NS", "BPCL.NS", "IOC.NS",
]


def get_universe(category: str) -> list:
    """Return ticker list for a given category."""
    mapping = {
        'sme':   SME_TICKERS,
        'small': SMALL_CAP_TICKERS,
        'mid':   MID_CAP_TICKERS,
        'large': LARGE_CAP_TICKERS,
    }
    tickers = mapping.get(category, [])

    # Deduplicate
    return list(dict.fromkeys(tickers))
