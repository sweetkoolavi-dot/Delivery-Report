import io
import json
import re
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote, urljoin

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import altair as alt
from bs4 import BeautifulSoup
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

st.set_page_config(page_title='Market Intelligence Terminal v9.1', page_icon='📈', layout='wide')

NIFTY500_CSV = 'https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv'
NSE_EQUITY_LIST_CSV = 'https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv'
BENCHMARK = '^CRSLDX'
VIX = '^INDIAVIX'
NSE_BHAVCOPY_URL = 'https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv'


@dataclass
class RegimeResult:
    score: float
    warning: float
    regime: str
    posture: str
    dashboard_state: str
    summary: str


def clamp(x, lo=0.0, hi=100.0):
    return float(max(lo, min(hi, x)))


def color_for_state(state: str):
    return {
        'BULLISH AND READY': '#18a957',
        'BULLISH BUT NOT READY': '#d4a017',
        'SIDEWAYS': '#4f8cc9',
        'BEARISH': '#d46b08',
        'ULTRA BEARISH': '#c62828',
    }.get(state, '#4f8cc9')


def classify_dashboard_state(score: float, warning: float):
    if score >= 75 and warning < 30:
        return 'BULLISH AND READY', 'Market internals support breakout and momentum trades.'
    if score >= 60 and warning < 60:
        return 'BULLISH BUT NOT READY', 'Trend is positive, but internals are not fully supportive yet.'
    if score >= 45:
        return 'SIDEWAYS', 'Mixed conditions. Be selective and trade smaller.'
    if score >= 25:
        return 'BEARISH', 'Weak environment. Capital protection should dominate.'
    return 'ULTRA BEARISH', 'High-risk environment. Avoid aggressive longs.'


def derive_sector(industry):
    """Map detailed industry labels into stable broad sectors for easier comparison."""
    if industry is None or (isinstance(industry, float) and np.isnan(industry)):
        return np.nan
    x = str(industry).strip()
    if not x or x.lower() in {'nan','none','unknown'}:
        return np.nan
    u = x.upper()
    rules = [
        ('FINANCIAL SERVICES', ['BANK','FINANCE','FINANCIAL','INSURANCE','CAPITAL MARKET','ASSET MANAGEMENT','NBFC']),
        ('INFORMATION TECHNOLOGY', ['INFORMATION TECHNOLOGY','SOFTWARE','IT SERVICES','COMPUTER']),
        ('HEALTHCARE', ['PHARMA','HEALTHCARE','HOSPITAL','BIOTECH','DIAGNOSTIC']),
        ('AUTOMOBILE & AUTO COMPONENTS', ['AUTOMOBILE','AUTO COMPONENT','TYRE']),
        ('FMCG / CONSUMER', ['FMCG','FAST MOVING','FOOD PRODUCTS','BEVERAGE','HOUSEHOLD','PERSONAL PRODUCTS','CONSUMER DURABLE','CONSUMER SERVICES','RETAIL']),
        ('CAPITAL GOODS / INDUSTRIALS', ['CAPITAL GOODS','INDUSTRIAL','ELECTRICAL EQUIPMENT','AEROSPACE','DEFENCE','ENGINEERING']),
        ('CONSTRUCTION / REALTY', ['CONSTRUCTION','REALTY','REAL ESTATE','CEMENT','CONSTRUCTION MATERIAL']),
        ('METALS & MINING', ['METAL','MINING','MINERALS','IRON','STEEL','ALUMIN']),
        ('ENERGY / POWER', ['POWER','OIL','GAS','ENERGY','PETROLEUM','COAL']),
        ('CHEMICALS', ['CHEMICAL','FERTILIZER','PESTICIDE']),
        ('TELECOM', ['TELECOM']),
        ('MEDIA', ['MEDIA','ENTERTAINMENT']),
        ('TEXTILES', ['TEXTILE','APPAREL']),
        ('SERVICES', ['SERVICES','LOGISTICS','TRANSPORT','TRAVEL']),
    ]
    for sector, keys in rules:
        if any(k in u for k in keys):
            return sector
    return x


def tradingview_url(symbol):
    return f"https://www.tradingview.com/chart/?symbol=NSE%3A{quote(str(symbol).strip())}"


MONEYCONTROL_SUGGEST = 'https://www.moneycontrol.com/mccode/common/autosuggestion_solr.php'


def _clean_label(x):
    if x is None:
        return np.nan
    val = str(x).strip()
    if not val or val.lower() in {'nan','none','unknown','n/a','na','-'}:
        return np.nan
    return val


def _parse_moneycontrol_jsonp(text):
    text = (text or '').strip()
    if not text:
        return []
    if text.startswith('['):
        try:
            return json.loads(text)
        except Exception:
            return []
    m = re.search(r'\((.*)\)\s*;?$', text, flags=re.S)
    if not m:
        return []
    try:
        return json.loads(m.group(1))
    except Exception:
        return []


def _moneycontrol_profile(symbol: str):
    """Best-effort Moneycontrol fallback for classification; used only after faster sources fail."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json,text/html,*/*',
    }
    try:
        params = {'classic':'true','query':symbol,'type':'1','format':'json','callback':'suggest1'}
        r = requests.get(MONEYCONTROL_SUGGEST, params=params, headers=headers, timeout=8)
        candidates = _parse_moneycontrol_jsonp(r.text)
        if not isinstance(candidates, list) or not candidates:
            return np.nan, np.nan

        def score_item(obj):
            vals = ' '.join(str(obj.get(k,'')) for k in ['nse_code','symbol','stock_name','company_name','name']).upper()
            return 2 if symbol.upper() in vals.split() else (1 if symbol.upper() in vals else 0)
        objects = [x for x in candidates if isinstance(x,dict)]
        if not objects:
            return np.nan, np.nan
        item = sorted(objects, key=score_item, reverse=True)[0]
        industry = np.nan
        sector = np.nan
        for k,v in item.items():
            kl = str(k).lower()
            if pd.isna(industry) and 'industry' in kl:
                industry = _clean_label(v)
            if pd.isna(sector) and 'sector' in kl:
                sector = _clean_label(v)
        if pd.notna(industry) or pd.notna(sector):
            if pd.isna(sector):
                sector = derive_sector(industry)
            return industry, sector

        link = item.get('link_src') or item.get('link') or item.get('url')
        if not link:
            return np.nan, np.nan
        page_url = urljoin('https://www.moneycontrol.com', str(link))
        page = requests.get(page_url, headers=headers, timeout=10)
        soup = BeautifulSoup(page.text, 'html.parser')
        plain = ' '.join(soup.stripped_strings)
        patterns = {
            'industry': [r'Industry\s*[:\-]?\s*([A-Za-z0-9 &/().,+\-]{3,90})',
                         r'Basic Industry\s*[:\-]?\s*([A-Za-z0-9 &/().,+\-]{3,90})'],
            'sector': [r'Sector\s*[:\-]?\s*([A-Za-z0-9 &/().,+\-]{3,90})']
        }
        for rgx in patterns['industry']:
            m = re.search(rgx, plain, flags=re.I)
            if m:
                industry = _clean_label(m.group(1).split('Market Cap')[0].strip())
                break
        for rgx in patterns['sector']:
            m = re.search(rgx, plain, flags=re.I)
            if m:
                sector = _clean_label(m.group(1).split('Industry')[0].strip())
                break
        if pd.isna(sector) and pd.notna(industry):
            sector = derive_sector(industry)
        return industry, sector
    except Exception:
        return np.nan, np.nan


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def resolve_stock_profile(symbol: str):
    """Layered resolver: Yahoo structured profile, then Moneycontrol fallback."""
    industry = np.nan
    sector = np.nan
    source = np.nan
    try:
        info = yf.Ticker(f'{symbol}.NS').get_info()
        industry = _clean_label(info.get('industry'))
        sector = _clean_label(info.get('sector'))
        if pd.isna(sector) and pd.notna(industry):
            sector = derive_sector(industry)
        if pd.notna(industry) or pd.notna(sector):
            source = 'Yahoo Finance'
    except Exception:
        pass
    if pd.isna(industry) or pd.isna(sector):
        mc_industry, mc_sector = _moneycontrol_profile(symbol)
        if pd.isna(industry) and pd.notna(mc_industry):
            industry = mc_industry
        if pd.isna(sector) and pd.notna(mc_sector):
            sector = mc_sector
        if pd.notna(mc_industry) or pd.notna(mc_sector):
            source = 'Moneycontrol' if pd.isna(source) else 'Yahoo + Moneycontrol'
    return {'Symbol': symbol, 'Resolved Industry': industry, 'Resolved Sector': sector, 'Resolved Source': source}


def enrich_profiles(symbols, workers=10):
    symbols = list(dict.fromkeys(symbols))
    rows=[]
    if not symbols:
        return pd.DataFrame(columns=['Symbol','Resolved Industry','Resolved Sector','Resolved Source'])
    with ThreadPoolExecutor(max_workers=min(workers, max(1,len(symbols)))) as ex:
        futs={ex.submit(resolve_stock_profile,s):s for s in symbols}
        for fut in as_completed(futs):
            try:
                rows.append(fut.result())
            except Exception:
                rows.append({'Symbol':futs[fut],'Resolved Industry':np.nan,'Resolved Sector':np.nan,'Resolved Source':np.nan})
    return pd.DataFrame(rows)


def apply_profile_enrichment(df: pd.DataFrame, profiles: pd.DataFrame):
    if df.empty or profiles is None or profiles.empty:
        return df
    out = df.drop(columns=['Resolved Industry','Resolved Sector','Resolved Source'], errors='ignore').merge(profiles, on='Symbol', how='left')
    if 'Industry' not in out.columns:
        out['Industry'] = np.nan
    if 'Sector' not in out.columns:
        out['Sector'] = np.nan
    out['Industry'] = out['Industry'].where(out['Industry'].notna(), out['Resolved Industry'])
    out['Sector'] = out['Sector'].where(out['Sector'].notna(), out['Resolved Sector'])
    if 'Classification Source' not in out.columns:
        out['Classification Source'] = np.nan
    out['Classification Source'] = out['Classification Source'].where(out['Classification Source'].notna(), out['Resolved Source'])
    out['Sector'] = out['Sector'].where(out['Sector'].notna(), out['Industry'].map(derive_sector))
    return out.drop(columns=['Resolved Industry','Resolved Sector','Resolved Source'], errors='ignore')


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_constituents():
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Accept': 'text/csv,text/plain,*/*',
        'Referer': 'https://www.niftyindices.com/'
    }
    r = requests.get(NIFTY500_CSV, headers=headers, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip() for c in df.columns]
    if 'Symbol' not in df.columns:
        raise ValueError('NIFTY 500 constituent file did not contain a Symbol column.')
    industry_col = next((c for c in df.columns if c.lower() == 'industry'), None)
    if industry_col is None:
        df['Industry'] = 'Unknown'
    elif industry_col != 'Industry':
        df = df.rename(columns={industry_col: 'Industry'})
    df['Ticker'] = df['Symbol'].astype(str).str.strip() + '.NS'
    df['Sector'] = df['Industry'].map(derive_sector)
    df['Classification Source'] = 'Nifty Indices'
    return df[['Symbol', 'Ticker', 'Sector', 'Industry', 'Classification Source']].drop_duplicates('Ticker')


EXTENDED_CLASSIFICATION_URLS = [
    'https://www.niftyindices.com/IndexConstituent/ind_niftytotalmarket_list.csv',
    'https://www.niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv',
    'https://www.niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv',
    'https://www.niftyindices.com/IndexConstituent/ind_niftymicrocap250_list.csv',
]


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_extended_classification():
    """Best-effort official Nifty classification coverage beyond Nifty 500."""
    frames = []
    headers = {'User-Agent':'Mozilla/5.0','Accept':'text/csv,text/plain,*/*','Referer':'https://www.niftyindices.com/'}
    for url in EXTENDED_CLASSIFICATION_URLS:
        try:
            r = requests.get(url, headers=headers, timeout=12)
            r.raise_for_status()
            x = pd.read_csv(io.StringIO(r.text))
            x.columns = [c.strip() for c in x.columns]
            sym = next((c for c in x.columns if c.lower() == 'symbol'), None)
            ind = next((c for c in x.columns if c.lower() == 'industry'), None)
            if sym and ind:
                z = x[[sym,ind]].rename(columns={sym:'Symbol',ind:'Industry'}).copy()
                z['Symbol'] = z['Symbol'].astype(str).str.strip()
                z['Sector'] = z['Industry'].map(derive_sector)
                z['Classification Source'] = 'Nifty Indices'
                frames.append(z)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=['Symbol','Sector','Industry','Classification Source'])
    return pd.concat(frames, ignore_index=True).drop_duplicates('Symbol')


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_broad_nse_universe():
    """Load ordinary NSE EQ-series securities and attach Nifty 500 industry where available."""
    headers = {'User-Agent': 'Mozilla/5.0', 'Accept': 'text/csv,text/plain,*/*'}
    r = requests.get(NSE_EQUITY_LIST_CSV, headers=headers, timeout=25)
    r.raise_for_status()
    eq = pd.read_csv(io.StringIO(r.text))
    eq.columns = [c.strip() for c in eq.columns]
    symbol_col = next((c for c in eq.columns if c.upper() == 'SYMBOL'), None)
    series_col = next((c for c in eq.columns if c.upper() == 'SERIES'), None)
    if symbol_col is None or series_col is None:
        raise ValueError('NSE equity list format changed: SYMBOL/SERIES columns not found.')
    eq = eq[eq[series_col].astype(str).str.strip().eq('EQ')].copy()
    eq['Symbol'] = eq[symbol_col].astype(str).str.strip()
    eq = eq[eq['Symbol'].ne('')].drop_duplicates('Symbol')
    eq['Ticker'] = eq['Symbol'] + '.NS'
    n500 = load_constituents()[['Symbol','Sector','Industry','Classification Source']].drop_duplicates('Symbol')
    ext = load_extended_classification()
    mapping = pd.concat([n500, ext], ignore_index=True).drop_duplicates('Symbol')
    eq = eq.merge(mapping, on='Symbol', how='left')
    return eq[['Symbol','Ticker','Sector','Industry','Classification Source']].reset_index(drop=True)


@st.cache_data(ttl=20 * 60, show_spinner=False)
def download_market_frames(tickers_tuple, period='1y'):
    """Download close and volume in batches so a ~2,000-stock universe is practical."""
    tickers = list(tickers_tuple)
    close_parts, volume_parts = [], []
    batch_size = 175
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            raw = yf.download(
                tickers=batch, period=period, interval='1d', auto_adjust=True,
                progress=False, group_by='column', threads=True, timeout=20,
            )
        except Exception:
            continue
        if raw is None or raw.empty:
            continue
        def field_frame(field):
            if isinstance(raw.columns, pd.MultiIndex):
                if field in raw.columns.get_level_values(0):
                    out = raw[field].copy()
                else:
                    cols = [c for c in raw.columns if c[0] == field or c[-1] == field]
                    if not cols:
                        return pd.DataFrame(index=raw.index)
                    out = raw.loc[:, cols].copy()
                    out.columns = [c[1] if c[0] == field else c[0] for c in cols]
            else:
                if field not in raw.columns:
                    return pd.DataFrame(index=raw.index)
                out = raw[[field]].copy()
                out.columns = [batch[0]]
            return out.sort_index().dropna(how='all')
        c = field_frame('Close')
        v = field_frame('Volume')
        if not c.empty: close_parts.append(c)
        if not v.empty: volume_parts.append(v)
    close = pd.concat(close_parts, axis=1) if close_parts else pd.DataFrame()
    volume = pd.concat(volume_parts, axis=1) if volume_parts else pd.DataFrame()
    close = close.loc[:, ~close.columns.duplicated()].sort_index() if not close.empty else close
    volume = volume.loc[:, ~volume.columns.duplicated()].sort_index() if not volume.empty else volume
    return close, volume


@st.cache_data(ttl=20 * 60, show_spinner=False)
def download_ohlc(ticker, period='1y'):
    d = yf.download(ticker, period=period, interval='1d', auto_adjust=True, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d.dropna(how='all')


def pct_above_ma(close, ma_len):
    if len(close) < ma_len + 2:
        return np.nan
    ma = close.rolling(ma_len).mean()
    latest_close = close.iloc[-1]
    latest_ma = ma.iloc[-1]
    valid = latest_close.notna() & latest_ma.notna()
    if valid.sum() == 0:
        return np.nan
    return float((latest_close[valid] > latest_ma[valid]).mean() * 100)


def breadth_history(close, ma_len=50):
    ma = close.rolling(ma_len).mean()
    valid = close.notna() & ma.notna()
    above = (close > ma).where(valid)
    return above.mean(axis=1, skipna=True) * 100


def advance_decline(close):
    rets = close.pct_change(fill_method=None)
    latest = rets.iloc[-1].dropna()
    if latest.empty:
        return np.nan, np.nan
    adv = (latest > 0).sum()
    dec = (latest < 0).sum()
    ratio = adv / max(dec, 1)
    adv_pct = adv / max(len(latest), 1) * 100
    return float(ratio), float(adv_pct)


def percentile_rank(series, value=None, lookback=252):
    s = pd.Series(series).dropna().tail(lookback)
    if len(s) < 10:
        return np.nan
    if value is None:
        value = s.iloc[-1]
    return float((s <= value).mean() * 100)


def calc_regime(benchmark, vix, close, meta):
    bclose = benchmark['Close'].dropna()
    if len(bclose) < 210:
        raise ValueError('Not enough benchmark history to calculate the regime.')

    c = float(bclose.iloc[-1])
    ma20 = float(bclose.rolling(20).mean().iloc[-1])
    ma50 = float(bclose.rolling(50).mean().iloc[-1])
    ma200 = float(bclose.rolling(200).mean().iloc[-1])
    r20 = float(c / bclose.iloc[-21] - 1) if len(bclose) > 21 else 0

    trend = 0
    trend += 5 if c > ma20 else 0
    trend += 5 if ma20 > ma50 else 0
    trend += 5 if ma50 > ma200 else 0
    trend += 5 if r20 > 0 else 0
    slope20 = ma20 / float(bclose.rolling(20).mean().iloc[-6]) - 1 if len(bclose) > 25 else 0
    trend += 5 if slope20 > 0 else 0

    b20 = pct_above_ma(close, 20)
    b50 = pct_above_ma(close, 50)
    b200 = pct_above_ma(close, 200)
    breadth = sum([clamp(x, 0, 100) / 10 for x in [b20, b50, b200] if not np.isnan(x)])

    ad_ratio, adv_pct = advance_decline(close)
    bh50 = breadth_history(close, 50).dropna()
    breadth_5d = (bh50.iloc[-1] - bh50.iloc[-6]) if len(bh50) >= 6 else 0
    breadth_20d = (bh50.iloc[-1] - bh50.iloc[-21]) if len(bh50) >= 21 else 0
    participation = 0
    participation += clamp(adv_pct if not np.isnan(adv_pct) else 50, 0, 100) * 0.07
    participation += clamp(50 + breadth_5d * 4, 0, 100) * 0.04
    participation += clamp(50 + breadth_20d * 2, 0, 100) * 0.04

    returns20 = close.iloc[-1] / close.shift(20).iloc[-1] - 1 if len(close) > 20 else pd.Series(dtype=float)
    x = pd.DataFrame({'Ticker': returns20.index, 'ret20': returns20.values})
    x = x.merge(meta[['Ticker', 'Industry']], on='Ticker', how='left').dropna(subset=['ret20'])
    sector_table = x.groupby('Industry').agg(stocks=('Ticker', 'count'), median_20d=('ret20', 'median')).reset_index()
    sector_table = sector_table[sector_table['stocks'] >= 3].copy()
    if sector_table.empty:
        outperform_pct = 50.0
    else:
        sector_table['vs_benchmark'] = sector_table['median_20d'] - r20
        sector_table['Outperforming'] = sector_table['vs_benchmark'] > 0
        outperform_pct = float(sector_table['Outperforming'].mean() * 100)
    sector_score = outperform_pct * 0.20

    vclose = vix['Close'].dropna() if not vix.empty and 'Close' in vix else pd.Series(dtype=float)
    if len(vclose) >= 20:
        v_now = float(vclose.iloc[-1])
        v_pctile = percentile_rank(vclose, v_now)
        v5 = float(vclose.pct_change(5, fill_method=None).iloc[-1] * 100)
        vol_score = (100 - v_pctile) * 0.07 + clamp(50 - v5 * 3, 0, 100) * 0.03
    else:
        v_now, v_pctile, v5, vol_score = np.nan, np.nan, np.nan, 5.0

    score = clamp(trend + breadth + participation + sector_score + vol_score)

    warning = 0.0
    price_20d_position = (c - float(bclose.tail(20).min())) / max(float(bclose.tail(20).max() - bclose.tail(20).min()), 1e-9)
    if price_20d_position > 0.75 and breadth_20d < -8:
        warning += 25
    elif breadth_20d < -5:
        warning += 15
    if breadth_5d < -5:
        warning += 15
    if b50 < 45:
        warning += 15
    if outperform_pct < 35:
        warning += 15
    if not np.isnan(v5) and v5 > 12:
        warning += 15
    if c < ma20:
        warning += 10
    if ma20 < ma50:
        warning += 10
    warning = clamp(warning)

    if score >= 80 and warning < 35:
        regime, posture = 'STRONG BULL', 'Aggressive long; favour leaders and breakouts'
    elif score >= 65 and warning < 50:
        regime, posture = 'BULL', 'Normal long exposure; buy strong stocks in strong groups'
    elif score >= 50:
        regime, posture = 'SELECTIVE / MIXED', 'Trade smaller; demand stronger confirmation'
    elif score >= 35:
        regime, posture = 'DISTRIBUTION / WEAK', 'Reduce exposure; avoid mediocre breakouts'
    else:
        regime, posture = 'BEAR / RISK-OFF', 'Capital preservation; longs only exceptional'

    dashboard_state, summary = classify_dashboard_state(score, warning)

    details = {
        'trend_score': trend,
        'breadth_score': breadth,
        'participation_score': participation,
        'sector_score': sector_score,
        'vol_score': vol_score,
        'b20': b20,
        'b50': b50,
        'b200': b200,
        'ad_ratio': ad_ratio,
        'adv_pct': adv_pct,
        'breadth_5d': breadth_5d,
        'breadth_20d': breadth_20d,
        'sector_outperform_pct': outperform_pct,
        'vix': v_now,
        'vix_pctile': v_pctile,
        'vix_5d_pct': v5,
        'sector_table': sector_table.sort_values('vs_benchmark', ascending=False) if not sector_table.empty else sector_table,
        'breadth_hist': bh50,
        'benchmark_close': bclose,
        'ma20': ma20,
        'ma50': ma50,
        'ma200': ma200,
        'price_above_20': c > ma20,
        'nifty20d': r20 * 100,
    }
    return RegimeResult(score, warning, regime, posture, dashboard_state, summary), details




@st.cache_data(ttl=12 * 3600, show_spinner=False)
def download_bhavcopy_full(dt: pd.Timestamp):
    """NSE daily cash bhavcopy used for broad-universe discovery without 2,000 Yahoo downloads."""
    url = NSE_BHAVCOPY_URL.format(ddmmyyyy=dt.strftime('%d%m%Y'))
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Referer': 'https://www.nseindia.com/',
        'Accept': 'text/csv,text/plain,*/*',
    }
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip().upper().replace(' ', '_') for c in df.columns]
    if 'DELIV_%' in df.columns and 'DELIV_PER' not in df.columns:
        df = df.rename(columns={'DELIV_%':'DELIV_PER'})
    if 'SERIES' not in df.columns or 'SYMBOL' not in df.columns:
        raise ValueError('NSE bhavcopy format changed.')
    df = df[df['SERIES'].astype(str).str.strip().eq('EQ')].copy()
    aliases = {
        'CLOSE_PRICE': ['CLOSE_PRICE','CLOSE'],
        'TTL_TRD_QNTY': ['TTL_TRD_QNTY','TOTTRDQTY','TOTTRD_QTY'],
        'TURNOVER_LACS': ['TURNOVER_LACS','TOTTRDVAL'],
        'DELIV_PER': ['DELIV_PER'],
    }
    for target, candidates in aliases.items():
        if target not in df.columns:
            found = next((c for c in candidates if c in df.columns), None)
            if found and found != target:
                df = df.rename(columns={found:target})
    required = ['SYMBOL','CLOSE_PRICE','TTL_TRD_QNTY','DELIV_PER']
    if not all(c in df.columns for c in required):
        raise ValueError('Required price/volume/delivery fields missing in NSE bhavcopy.')
    keep = [c for c in ['SYMBOL','CLOSE_PRICE','TTL_TRD_QNTY','TURNOVER_LACS','DELIV_PER'] if c in df.columns]
    df = df[keep].copy()
    df['SYMBOL'] = df['SYMBOL'].astype(str).str.strip()
    for c in keep:
        if c != 'SYMBOL':
            df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'TURNOVER_LACS' not in df.columns:
        df['TURNOVER_LACS'] = df['CLOSE_PRICE'] * df['TTL_TRD_QNTY'] / 1e5
    return df.dropna(subset=['SYMBOL','CLOSE_PRICE'])


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_broad_bhav_history(symbols_tuple, sessions=30):
    symbols = set(symbols_tuple)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=max(sessions * 3, 70))
    rows = []
    got = 0
    for dt in dates[::-1]:
        try:
            d = download_bhavcopy_full(dt)
        except Exception:
            continue
        d = d[d['SYMBOL'].isin(symbols)].copy()
        if d.empty:
            continue
        d['Date'] = pd.to_datetime(dt.date())
        rows.append(d)
        got += 1
        if got >= sessions:
            break
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(['Date','SYMBOL'])


def broad_stock_summary(bhav: pd.DataFrame, meta: pd.DataFrame, benchmark_close: pd.Series, min_traded_value_cr=1):
    """Fast broad-universe stock intelligence using only cached NSE bhavcopy history.

    The function intentionally avoids per-stock web/API calls. All expensive enrichment
    (Piotroski/news) happens only after this funnel has reduced the universe.
    """
    if bhav.empty:
        return pd.DataFrame()
    b = bhav.sort_values(['SYMBOL','Date']).copy()
    g = b.groupby('SYMBOL', sort=False)
    latest = g.tail(1).set_index('SYMBOL')
    tail20 = g.tail(20).copy()
    tail10 = g.tail(10).copy()
    tail5 = g.tail(5).copy()

    avg20_delivery = tail20.groupby('SYMBOL')['DELIV_PER'].mean()
    std20_delivery = tail20.groupby('SYMBOL')['DELIV_PER'].std().replace(0, np.nan)
    avg5_delivery = tail5.groupby('SYMBOL')['DELIV_PER'].mean()
    avg20_vol = tail20.groupby('SYMBOL')['TTL_TRD_QNTY'].mean()
    std20_vol = tail20.groupby('SYMBOL')['TTL_TRD_QNTY'].std().replace(0, np.nan)
    avg5_vol = tail5.groupby('SYMBOL')['TTL_TRD_QNTY'].mean()
    avg20_turnover_cr = tail20.groupby('SYMBOL')['TURNOVER_LACS'].mean() / 100.0
    first20 = g.tail(22).groupby('SYMBOL').first()['CLOSE_PRICE']
    first5 = g.tail(6).groupby('SYMBOL').first()['CLOSE_PRICE']
    prev_px = g.tail(2).groupby('SYMBOL').first()['CLOSE_PRICE']
    last_px = latest['CLOSE_PRICE']

    out = pd.DataFrame(index=last_px.index)
    out['Latest Delivery %'] = latest['DELIV_PER']
    out['5D Avg Delivery %'] = avg5_delivery
    out['20D Avg Delivery %'] = avg20_delivery
    out['Delivery Acceleration'] = out['Latest Delivery %'] - out['20D Avg Delivery %']
    out['Delivery Z'] = (out['Latest Delivery %'] - avg20_delivery) / std20_delivery
    out['Volume Spike x'] = latest['TTL_TRD_QNTY'] / avg20_vol.replace(0,np.nan)
    out['5D Volume x'] = avg5_vol / avg20_vol.replace(0,np.nan)
    out['Volume Z'] = (latest['TTL_TRD_QNTY'] - avg20_vol) / std20_vol
    out['20D Avg Traded Value Cr'] = avg20_turnover_cr
    out['Today Traded Value Cr'] = latest['TURNOVER_LACS'] / 100.0
    out['Today % Change'] = (last_px / prev_px.replace(0,np.nan) - 1) * 100
    out['1M Price Change %'] = (last_px / first20 - 1) * 100
    out['5D Price Change %'] = (last_px / first5 - 1) * 100

    # Cheap price-location / volatility features from daily closes only.
    price_pivot = b.pivot(index='Date', columns='SYMBOL', values='CLOSE_PRICE').sort_index()
    p20 = price_pivot.tail(20)
    sma20 = p20.mean()
    high20 = p20.max()
    out['Distance to 20D High %'] = (last_px / high20.reindex(last_px.index) - 1) * 100
    out['Price Extension vs 20DMA %'] = (last_px / sma20.reindex(last_px.index) - 1) * 100
    ret = price_pivot.pct_change(fill_method=None)
    vol20 = ret.tail(20).std() * np.sqrt(252) * 100
    vol5 = ret.tail(5).std() * np.sqrt(252) * 100
    out['Volatility Contraction'] = (vol5 / vol20.replace(0,np.nan)).reindex(out.index)

    bc = benchmark_close.dropna()
    bench20 = (bc.iloc[-1]/bc.iloc[-22]-1)*100 if len(bc)>=22 else 0
    bench5 = (bc.iloc[-1]/bc.iloc[-6]-1)*100 if len(bc)>=6 else 0
    out['RS vs N500 20D %'] = out['1M Price Change %'] - bench20
    out['RS vs N500 5D %'] = out['5D Price Change %'] - bench5
    out['RS Acceleration'] = out['RS vs N500 5D %'] - (out['RS vs N500 20D %'] / 4.0)
    out['RS Improving'] = out['RS Acceleration'] > 0

    # Persistence: how many of last 10 sessions showed above-normal delivery / volume.
    mean_del_map = avg20_delivery.to_dict()
    mean_vol_map = avg20_vol.to_dict()
    tail10['del_above'] = tail10.apply(lambda r: float(r['DELIV_PER'] > mean_del_map.get(r['SYMBOL'], np.inf)), axis=1)
    tail10['vol_above'] = tail10.apply(lambda r: float(r['TTL_TRD_QNTY'] > mean_vol_map.get(r['SYMBOL'], np.inf)), axis=1)
    out['Delivery Persistence 10D'] = tail10.groupby('SYMBOL')['del_above'].sum()
    out['Volume Persistence 10D'] = tail10.groupby('SYMBOL')['vol_above'].sum()

    out = out[out['20D Avg Traded Value Cr'].fillna(0) >= float(min_traded_value_cr)]
    merge_cols = [c for c in ['Symbol','Sector','Industry','Classification Source'] if c in meta.columns]
    out = out.reset_index().rename(columns={'SYMBOL':'Symbol'}).merge(meta[merge_cols], on='Symbol', how='left')

    # Sector-relative strength where an industry mapping exists; unmapped broad-NSE names stay N/A.
    sector_ret = out.dropna(subset=['Industry']).groupby('Industry')['1M Price Change %'].median()
    out['Sector 20D Return %'] = out['Industry'].map(sector_ret)
    out['RS vs Sector 20D %'] = out['1M Price Change %'] - out['Sector 20D Return %']

    # Stage-1 accumulation score: all components are cheap and cross-sectional.
    d20p = percentile_series(out['20D Avg Delivery %'])
    dzp = percentile_series(out['Delivery Z'])
    vzp = percentile_series(out['Volume Z'])
    tvp = percentile_series(out['Today Traded Value Cr'])
    rsp = percentile_series(out['RS vs N500 20D %'])
    rap = percentile_series(out['RS Acceleration'])
    persistp = percentile_series(out['Delivery Persistence 10D'] + out['Volume Persistence 10D'])
    out['Participation Conviction'] = (0.25*dzp + 0.35*vzp + 0.25*tvp + 0.15*persistp).round(1)
    # Delivery cannot dominate the score unless actual market participation is present.
    out['Accumulation Score'] = (0.18*d20p + 0.10*dzp + 0.22*vzp + 0.15*tvp + 0.17*rsp + 0.10*rap + 0.08*persistp).round(1)

    def opportunity_type(r):
        rs20 = r.get('RS vs N500 20D %', -99)
        rsa = r.get('RS Acceleration', -99)
        dz = r.get('Delivery Z', -99)
        vz = r.get('Volume Z', -99)
        d20 = r.get('20D Avg Delivery %', 0)
        p20 = r.get('1M Price Change %', -999)
        p5 = r.get('5D Price Change %', -999)
        p0 = r.get('Today % Change', -999)
        traded = r.get('Today Traded Value Cr', 0)
        near_high = r.get('Distance to 20D High %', -99) >= -3
        contraction = r.get('Volatility Contraction', 9) < 0.85
        if p5 < -2 and vz >= 1.5 and dz >= 1 and rs20 < 0:
            return '🔴 DISTRIBUTION RISK'
        if p20 > 15 or p5 > 8 or r.get('Price Extension vs 20DMA %', 0) > 10:
            return '🟠 EXTENDED'
        if 0 <= p20 <= 7 and d20 >= 45 and dz >= 0.5 and vz >= 0.8 and rsa > 0 and traded >= 1:
            return '🟣 EARLY ACCUMULATION'
        if rs20 > 0 and rsa > 0 and near_high and contraction and d20 >= 40:
            return '🔵 SETUP READY'
        if rs20 > 0 and rsa > 0 and near_high and r.get('Volume Spike x',0) >= 1.3 and dz >= 0 and traded >= 1:
            return '🟢 MOMENTUM ENTRY'
        if rs20 > 0 and rsa > 0:
            return '🟡 WATCH'
        return 'NEUTRAL'
    out['Opportunity Type'] = out.apply(opportunity_type, axis=1)
    out['Signal'] = out['Opportunity Type']
    out = add_entry_scores(out)
    return out.sort_values(['Entry Suitability Score','Accumulation Score'], ascending=False)


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def download_bhavcopy_for_date(dt: pd.Timestamp):
    url = NSE_BHAVCOPY_URL.format(ddmmyyyy=dt.strftime('%d%m%Y'))
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Referer': 'https://www.nseindia.com/',
        'Accept': 'text/csv,text/plain,*/*',
    }
    r = requests.get(url, headers=headers, timeout=25)
    r.raise_for_status()
    text = r.text.strip()
    if not text or 'SYMBOL' not in text.upper():
        raise ValueError('Bhavcopy unavailable')
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip() for c in df.columns]
    rename_map = {}
    for col in df.columns:
        c = col.upper().replace(' ', '_')
        if c == 'DELIV_%':
            c = 'DELIV_PER'
        rename_map[col] = c
    df = df.rename(columns=rename_map)
    needed = {'SYMBOL', 'SERIES', 'DELIV_PER'}
    if not needed.issubset(df.columns):
        raise ValueError('Required delivery columns missing in bhavcopy')
    df = df[df['SERIES'].astype(str).str.strip().eq('EQ')].copy()
    df['SYMBOL'] = df['SYMBOL'].astype(str).str.strip()
    df['DELIV_PER'] = pd.to_numeric(df['DELIV_PER'], errors='coerce')
    df = df.dropna(subset=['DELIV_PER'])
    return df[['SYMBOL', 'DELIV_PER']]


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def load_delivery_data(meta, lookback_days=40):
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=max(lookback_days * 3, 80))
    meta_small = meta[['Symbol', 'Industry']].drop_duplicates('Symbol')
    records = []
    success_count = 0
    for dt in dates[::-1]:
        try:
            bhav = download_bhavcopy_for_date(dt)
        except Exception:
            continue
        merged = bhav.merge(meta_small, left_on='SYMBOL', right_on='Symbol', how='inner')
        if merged.empty:
            continue
        merged['Date'] = pd.to_datetime(dt.date())
        records.append(merged[['Date', 'SYMBOL', 'Industry', 'DELIV_PER']].rename(columns={'SYMBOL': 'Symbol', 'DELIV_PER': 'Delivery %'}))
        success_count += 1
        if success_count >= lookback_days:
            break
    if not records:
        return pd.DataFrame()
    data = pd.concat(records, ignore_index=True)
    data['Delivery %'] = pd.to_numeric(data['Delivery %'], errors='coerce')
    data = data.dropna(subset=['Delivery %'])
    return data.sort_values(['Date', 'Industry', 'Symbol'])


def summarize_sector_delivery(delivery_long: pd.DataFrame):
    if delivery_long.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    sector_pivot = delivery_long.groupby(['Date', 'Industry'])['Delivery %'].mean().unstack('Industry').sort_index()
    sector_smooth = sector_pivot.rolling(5).mean()
    latest = sector_pivot.iloc[-1].rename('Latest Delivery %').to_frame()
    latest['5D Avg Delivery %'] = sector_smooth.iloc[-1].reindex(latest.index)
    latest['5D Trend'] = latest['Latest Delivery %'] - latest['5D Avg Delivery %']
    roll20_mean = sector_pivot.rolling(20).mean()
    roll20_std = sector_pivot.rolling(20).std().replace(0, np.nan)
    latest['Delivery Z Score'] = ((sector_smooth.iloc[-1] - roll20_mean.iloc[-1]) / roll20_std.iloc[-1]).reindex(latest.index)
    if len(sector_pivot) >= 6:
        latest['5D Change'] = (sector_pivot.iloc[-1] - sector_pivot.iloc[-6]).reindex(latest.index)
    else:
        latest['5D Change'] = np.nan
    latest = latest.sort_values(['Latest Delivery %', '5D Avg Delivery %'], ascending=False)
    return sector_pivot, sector_smooth, latest


def percentile_series(s):
    x = pd.to_numeric(s, errors='coerce')
    return x.rank(pct=True) * 100


def summarize_stock_delivery(delivery_long: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame, meta: pd.DataFrame, benchmark_close: pd.Series):
    if delivery_long.empty:
        return pd.DataFrame()
    delivery_long = delivery_long.sort_values(['Symbol', 'Date']).copy()
    grouped = delivery_long.groupby('Symbol')
    latest_delivery = grouped['Delivery %'].last().rename('Latest Delivery %')
    avg5 = delivery_long.groupby('Symbol').tail(5).groupby('Symbol')['Delivery %'].mean().rename('5D Avg Delivery %')
    avg20 = delivery_long.groupby('Symbol').tail(20).groupby('Symbol')['Delivery %'].mean().rename('20D Avg Delivery %')
    count_obs = grouped['Delivery %'].count().rename('Obs')
    out = pd.concat([latest_delivery, avg5, avg20, count_obs], axis=1)
    out['Delivery Acceleration'] = out['Latest Delivery %'] - out['20D Avg Delivery %']

    ticker_index = out.index.map(lambda x: f'{x}.NS')
    if len(close) >= 22:
        chg20 = (close.iloc[-1] / close.iloc[-22] - 1) * 100
        out['1M Price Change %'] = chg20.reindex(ticker_index).values
    else:
        out['1M Price Change %'] = np.nan
    if len(close) >= 6:
        chg5 = (close.iloc[-1] / close.iloc[-6] - 1) * 100
        out['5D Price Change %'] = chg5.reindex(ticker_index).values
    else:
        out['5D Price Change %'] = np.nan

    # Volume spike = latest volume / 20-session average. 1.0 means normal, 2.0 means 2x normal.
    if not volume.empty and len(volume) >= 21:
        latest_v = volume.iloc[-1]
        avg20_v = volume.tail(20).mean()
        ratio = latest_v / avg20_v.replace(0, np.nan)
        avg5_ratio = volume.tail(5).mean() / avg20_v.replace(0, np.nan)
        out['Volume Spike x'] = ratio.reindex(ticker_index).values
        out['5D Volume x'] = avg5_ratio.reindex(ticker_index).values
    else:
        out['Volume Spike x'] = np.nan
        out['5D Volume x'] = np.nan

    # Relative strength versus the NIFTY 500 benchmark.
    bc = benchmark_close.dropna()
    if len(bc) >= 22 and len(close) >= 22:
        bench20 = (bc.iloc[-1] / bc.iloc[-22] - 1) * 100
        bench5 = (bc.iloc[-1] / bc.iloc[-6] - 1) * 100 if len(bc) >= 6 else 0
        out['RS vs N500 20D %'] = out['1M Price Change %'] - bench20
        out['RS vs N500 5D %'] = out['5D Price Change %'] - bench5
        out['RS Improving'] = out['RS vs N500 5D %'] > 0
    else:
        out['RS vs N500 20D %'] = np.nan
        out['RS vs N500 5D %'] = np.nan
        out['RS Improving'] = False

    out = out.reset_index().merge(meta[['Symbol', 'Industry']], on='Symbol', how='left')

    # Combined accumulation score. Price change is intentionally not a major weight: a quiet stock
    # with high delivery + high volume + improving RS can rank highly before a large price move.
    d20p = percentile_series(out['20D Avg Delivery %'])
    vsp = percentile_series(out['Volume Spike x'])
    rsp = percentile_series(out['RS vs N500 20D %'])
    dap = percentile_series(out['Delivery Acceleration'])
    out['Accumulation Score'] = (0.35*d20p + 0.25*vsp + 0.25*rsp + 0.15*dap).round(1)

    def setup_tag(r):
        vol_spike = r.get('Volume Spike x', np.nan)
        rs20 = r.get('RS vs N500 20D %', np.nan)
        p20 = r.get('1M Price Change %', np.nan)
        d20 = r.get('20D Avg Delivery %', np.nan)
        if pd.notna(vol_spike) and vol_spike >= 1.5 and pd.notna(d20) and d20 >= 50 and pd.notna(rs20) and rs20 > 0:
            if pd.notna(p20) and 0 <= p20 <= 6:
                return 'QUIET ACCUMULATION'
            if pd.notna(p20) and p20 > 6:
                return 'MOMENTUM ACCUMULATION'
        if pd.notna(vol_spike) and vol_spike >= 1.5 and pd.notna(p20) and p20 < 0:
            return 'HIGH ACTIVITY - CHECK'
        if pd.notna(rs20) and rs20 > 0:
            return 'RS IMPROVING'
        return 'NEUTRAL'

    out['Signal'] = out.apply(setup_tag, axis=1)
    return out.sort_values(['Accumulation Score', 'Volume Spike x'], ascending=False)


def build_sector_market_analytics(close: pd.DataFrame, volume: pd.DataFrame, meta: pd.DataFrame, benchmark_close: pd.Series):
    if close.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    ticker_to_industry = meta.set_index('Ticker')['Industry'].to_dict()
    valid_cols = [c for c in close.columns if c in ticker_to_industry]
    c = close[valid_cols].copy()
    v = volume.reindex(index=c.index, columns=valid_cols).copy() if not volume.empty else pd.DataFrame(index=c.index, columns=valid_cols)

    # Equal-weight sector index from daily stock returns.
    rets = c.pct_change(fill_method=None)
    sector_ret = {}
    for industry in sorted(set(ticker_to_industry[cx] for cx in valid_cols)):
        cols = [cx for cx in valid_cols if ticker_to_industry[cx] == industry]
        if len(cols) >= 2:
            sector_ret[industry] = rets[cols].mean(axis=1, skipna=True)
    sector_ret = pd.DataFrame(sector_ret).fillna(0)
    sector_index = (1 + sector_ret).cumprod()

    bc = benchmark_close.reindex(sector_index.index).ffill().dropna()
    sector_index = sector_index.reindex(bc.index)
    bench_index = bc / bc.iloc[0]
    rs = sector_index.div(bench_index, axis=0)
    rs = rs.div(rs.iloc[0]).mul(100)
    rs5 = rs.rolling(5).mean()

    # Sector activity uses value traded proxy (close * volume), which makes cross-stock aggregation meaningful.
    if v.empty:
        sector_turnover = pd.DataFrame()
    else:
        value_traded = c * v
        sector_turnover = {}
        for industry in sector_ret.columns:
            cols = [cx for cx in valid_cols if ticker_to_industry[cx] == industry]
            sector_turnover[industry] = value_traded[cols].sum(axis=1, min_count=1)
        sector_turnover = pd.DataFrame(sector_turnover)

    summary = pd.DataFrame(index=sector_ret.columns)
    if len(sector_index) >= 22:
        summary['Sector 20D Price %'] = (sector_index.iloc[-1] / sector_index.iloc[-22] - 1) * 100
    else:
        summary['Sector 20D Price %'] = np.nan
    if len(sector_index) >= 6:
        summary['Sector 5D Price %'] = (sector_index.iloc[-1] / sector_index.iloc[-6] - 1) * 100
    else:
        summary['Sector 5D Price %'] = np.nan
    if len(rs) >= 22:
        summary['Sector RS 20D %'] = (rs.iloc[-1] / rs.iloc[-22] - 1) * 100
    else:
        summary['Sector RS 20D %'] = np.nan
    if len(rs) >= 6:
        summary['Sector RS 5D %'] = (rs.iloc[-1] / rs.iloc[-6] - 1) * 100
    else:
        summary['Sector RS 5D %'] = np.nan
    if not sector_turnover.empty and len(sector_turnover) >= 21:
        summary['Sector Volume Spike x'] = sector_turnover.iloc[-1] / sector_turnover.tail(20).mean().replace(0, np.nan)
        summary['Sector 5D Volume x'] = sector_turnover.tail(5).mean() / sector_turnover.tail(20).mean().replace(0, np.nan)
    else:
        summary['Sector Volume Spike x'] = np.nan
        summary['Sector 5D Volume x'] = np.nan

    return rs, rs5, summary


def merge_sector_scores(sector_rank: pd.DataFrame, sector_market_summary: pd.DataFrame):
    if sector_rank.empty and sector_market_summary.empty:
        return pd.DataFrame()
    out = sector_rank.join(sector_market_summary, how='outer')
    for col in ['Latest Delivery %', '5D Avg Delivery %', 'Delivery Z Score', '5D Trend', '5D Change', 'Sector 20D Price %', 'Sector 5D Price %', 'Sector RS 20D %', 'Sector RS 5D %', 'Sector Volume Spike x', 'Sector 5D Volume x']:
        if col not in out.columns:
            out[col] = np.nan
    dp = percentile_series(out['5D Avg Delivery %'])
    dzp = percentile_series(out['Delivery Z Score'])
    vp = percentile_series(out['Sector Volume Spike x'])
    rp = percentile_series(out['Sector RS 20D %'])
    r5p = percentile_series(out['Sector RS 5D %'])
    # Delivery is useful only when participation/relative strength also confirm.
    out['Sector Opportunity Score'] = (0.22*dp + 0.13*dzp + 0.25*vp + 0.25*rp + 0.15*r5p).round(1)
    return out.sort_values(['Sector Opportunity Score', 'Sector Volume Spike x', '5D Avg Delivery %'], ascending=False)


def line_chart_with_scale(df: pd.DataFrame, title: str, log_scale: bool = False, height: int = 430):
    data = df.reset_index().melt(id_vars=df.index.name or 'index', var_name='Series', value_name='Value')
    date_col = df.index.name or 'index'
    data = data.dropna(subset=['Value'])
    if log_scale:
        data = data[data['Value'] > 0]
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X(f'{date_col}:T', title=None),
        y=alt.Y('Value:Q', title=title, scale=alt.Scale(type='log' if log_scale else 'linear')),
        color=alt.Color('Series:N', legend=alt.Legend(orient='bottom', columns=3)),
        tooltip=[alt.Tooltip(f'{date_col}:T', title='Date'), alt.Tooltip('Series:N'), alt.Tooltip('Value:Q', format='.2f')],
    ).properties(height=height).interactive()
    st.altair_chart(chart, use_container_width=True)




def render_frozen_grid(df: pd.DataFrame, pinned_left=None, height=420, score_columns=None, link_columns=None, key=None):
    """Professional scrollable grid with important columns pinned on the left."""
    if df is None or df.empty:
        st.info('No data available for this view.')
        return
    pinned_left = pinned_left or []
    score_columns = score_columns or []
    link_columns = link_columns or []
    view = df.copy()
    gb = GridOptionsBuilder.from_dataframe(view)
    gb.configure_default_column(resizable=True, sortable=True, filter=True, minWidth=105, flex=0)
    for col in pinned_left:
        if col in view.columns:
            gb.configure_column(col, pinned='left', lockPinned=True, minWidth=125)

    score_style = JsCode("""
        function(params) {
          if (params.value === null || params.value === undefined || isNaN(params.value)) return {};
          let v = Math.max(0, Math.min(100, Number(params.value)));
          if (v >= 82) return {backgroundColor:'#0f766e', color:'#ffffff', fontWeight:'700'};
          if (v >= 72) return {backgroundColor:'#d1fae5', color:'#065f46', fontWeight:'700'};
          if (v >= 62) return {backgroundColor:'#fef3c7', color:'#92400e', fontWeight:'650'};
          return {backgroundColor:'#f3f4f6', color:'#4b5563', fontWeight:'600'};
        }
    """)
    for col in score_columns:
        if col in view.columns:
            gb.configure_column(col, cellStyle=score_style, minWidth=145)

    link_renderer = JsCode("""
        class UrlCellRenderer {
          init(params) {
            this.eGui = document.createElement('a');
            this.eGui.innerText = 'Open chart';
            this.eGui.setAttribute('href', params.value || '#');
            this.eGui.setAttribute('target', '_blank');
            this.eGui.style.color = '#60a5fa';
            this.eGui.style.fontWeight = '600';
          }
          getGui() { return this.eGui; }
        }
    """)
    for col in link_columns:
        if col in view.columns:
            gb.configure_column(col, cellRenderer=link_renderer, minWidth=110)

    # Compact numeric formatting while retaining true numeric sorting/filtering.
    one_dec = JsCode("function(p){return (p.value===null||p.value===undefined||isNaN(p.value))?'':Number(p.value).toFixed(1);}")
    two_dec = JsCode("function(p){return (p.value===null||p.value===undefined||isNaN(p.value))?'':Number(p.value).toFixed(2);}")
    int_fmt = JsCode("function(p){return (p.value===null||p.value===undefined||isNaN(p.value))?'':Math.round(Number(p.value));}")
    for col in view.columns:
        if col in link_columns:
            continue
        if pd.api.types.is_numeric_dtype(view[col]):
            if 'Score' in col or 'Delivery %' in col or 'Traded Value' in col:
                gb.configure_column(col, valueFormatter=one_dec)
            elif 'Persistence' in col or 'Coverage' in col:
                gb.configure_column(col, valueFormatter=int_fmt)
            else:
                gb.configure_column(col, valueFormatter=two_dec)

    gb.configure_grid_options(rowHeight=36, headerHeight=40, suppressRowClickSelection=True, ensureDomOrder=True)
    AgGrid(
        view,
        gridOptions=gb.build(),
        height=height,
        use_container_width=True,
        theme='streamlit',
        allow_unsafe_jscode=True,
        enable_enterprise_modules=False,
        key=key,
    )


def render_sector_radar_cards(sector_df: pd.DataFrame, limit=8):
    """Clean, native Sector Radar: ranked bar chart + compact detail strip.
    Avoids raw HTML so Streamlit never exposes markup as text.
    """
    if sector_df is None or sector_df.empty:
        st.info('Sector Radar is waiting for sector data. It will remain visible here once the scan is ready.')
        return

    x = sector_df.head(limit).copy().reset_index()
    first = x.columns[0]
    x = x.rename(columns={first: 'Sector'})
    x['Sector'] = x['Sector'].astype(str)
    x['Score'] = pd.to_numeric(x.get('Sector Opportunity Score', 0), errors='coerce').fillna(0).clip(0, 100)

    def _band(v):
        if v >= 80:
            return 'High conviction'
        if v >= 68:
            return 'Positive'
        if v >= 55:
            return 'Watch'
        return 'Neutral'

    x['View'] = x['Score'].map(_band)
    order = x.sort_values('Score', ascending=False)['Sector'].tolist()

    left, right = st.columns([1.55, 1.0], gap='large')
    with left:
        # Restrained palette: only the best group is green; others are blue/amber/neutral.
        color_scale = alt.Scale(
            domain=['High conviction', 'Positive', 'Watch', 'Neutral'],
            range=['#0f766e', '#3b82f6', '#d4a72c', '#94a3b8']
        )
        bar = (
            alt.Chart(x)
            .mark_bar(cornerRadiusEnd=6, height=22)
            .encode(
                y=alt.Y('Sector:N', sort=order, title=None, axis=alt.Axis(labelLimit=180, labelFontSize=12)),
                x=alt.X('Score:Q', scale=alt.Scale(domain=[0, 100]), title='Opportunity score', axis=alt.Axis(grid=True, tickCount=5)),
                color=alt.Color('View:N', scale=color_scale, legend=alt.Legend(title=None, orient='bottom')),
                tooltip=[
                    alt.Tooltip('Sector:N', title='Sector'),
                    alt.Tooltip('Score:Q', title='Opportunity', format='.1f'),
                    alt.Tooltip('View:N', title='View'),
                    alt.Tooltip('5D Avg Delivery %:Q', title='Delivery 5DMA', format='.1f'),
                    alt.Tooltip('Delivery Z Score:Q', title='Delivery Z', format='+.2f'),
                    alt.Tooltip('Sector Volume Spike x:Q', title='RVOL', format='.2f'),
                    alt.Tooltip('Sector RS 20D %:Q', title='RS 20D', format='+.2f'),
                ],
            )
            .properties(height=max(260, 38 * len(x)))
        )
        labels = (
            alt.Chart(x)
            .mark_text(align='left', baseline='middle', dx=5, fontWeight='bold', fontSize=12)
            .encode(
                y=alt.Y('Sector:N', sort=order),
                x=alt.X('Score:Q', scale=alt.Scale(domain=[0, 100])),
                text=alt.Text('Score:Q', format='.0f'),
                color=alt.value('#334155'),
            )
        )
        st.altair_chart((bar + labels).configure_view(strokeOpacity=0), use_container_width=True)
        st.caption('Green is reserved for high-conviction sectors. Blue = positive, amber = watch, grey = neutral.')

    with right:
        st.markdown('**What is driving the leaders**')
        for _, r in x.sort_values('Score', ascending=False).head(5).iterrows():
            score = float(r['Score'])
            delivery = r.get('5D Avg Delivery %', np.nan)
            vol = r.get('Sector Volume Spike x', np.nan)
            rs = r.get('Sector RS 20D %', np.nan)
            dz = r.get('Delivery Z Score', np.nan)
            with st.container(border=True):
                a, b = st.columns([2.0, 0.65])
                a.markdown(f"**{r['Sector']}**  ·  {_band(score)}")
                b.markdown(f"### {score:.0f}")
                bits=[]
                if pd.notna(delivery): bits.append(f"Delivery **{delivery:.1f}%**")
                if pd.notna(vol): bits.append(f"RVOL **{vol:.2f}×**")
                if pd.notna(rs): bits.append(f"RS **{rs:+.1f}%**")
                if pd.notna(dz): bits.append(f"D-Z **{dz:+.1f}**")
                st.caption(' · '.join(bits))


def classification_coverage(df: pd.DataFrame):
    if df is None or df.empty:
        return 0, 0, 0.0
    mapped = int((df['Industry'].notna() & df['Sector'].notna()).sum()) if {'Industry','Sector'}.issubset(df.columns) else 0
    total = len(df)
    return mapped, total, (mapped/total*100 if total else 0)


def _statement_value(df, names, col_idx=0):
    """Return the first matching accounting row for a given statement column."""
    if df is None or df.empty or len(df.columns) <= col_idx:
        return np.nan
    # exact first
    idx_map = {str(i).strip().lower(): i for i in df.index}
    for name in names:
        key = name.strip().lower()
        if key in idx_map:
            try:
                return float(df.loc[idx_map[key]].iloc[col_idx])
            except Exception:
                pass
    # tolerant contains match
    for name in names:
        key = name.strip().lower()
        for idx in df.index:
            s = str(idx).strip().lower()
            if key in s or s in key:
                try:
                    return float(df.loc[idx].iloc[col_idx])
                except Exception:
                    pass
    return np.nan


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def calculate_piotroski(symbol: str):
    """Best-effort Piotroski F-Score (0-9) from latest annual Yahoo Finance statements."""
    try:
        t = yf.Ticker(f'{symbol}.NS')
        inc = t.financials
        bal = t.balance_sheet
        cf = t.cashflow
        if inc is None or bal is None or cf is None or inc.empty or bal.empty or cf.empty:
            return {'Symbol': symbol, 'Piotroski F-Score': np.nan, 'F-Score Coverage': 0}

        # newest columns first in yfinance annual statements
        ni0 = _statement_value(inc, ['Net Income', 'Net Income Common Stockholders'], 0)
        ni1 = _statement_value(inc, ['Net Income', 'Net Income Common Stockholders'], 1)
        rev0 = _statement_value(inc, ['Total Revenue', 'Operating Revenue'], 0)
        rev1 = _statement_value(inc, ['Total Revenue', 'Operating Revenue'], 1)
        gp0 = _statement_value(inc, ['Gross Profit'], 0)
        gp1 = _statement_value(inc, ['Gross Profit'], 1)

        a0 = _statement_value(bal, ['Total Assets'], 0)
        a1 = _statement_value(bal, ['Total Assets'], 1)
        a2 = _statement_value(bal, ['Total Assets'], 2)
        ca0 = _statement_value(bal, ['Current Assets', 'Total Current Assets'], 0)
        ca1 = _statement_value(bal, ['Current Assets', 'Total Current Assets'], 1)
        cl0 = _statement_value(bal, ['Current Liabilities', 'Total Current Liabilities'], 0)
        cl1 = _statement_value(bal, ['Current Liabilities', 'Total Current Liabilities'], 1)
        debt0 = _statement_value(bal, ['Long Term Debt', 'Long Term Debt And Capital Lease Obligation', 'Total Debt'], 0)
        debt1 = _statement_value(bal, ['Long Term Debt', 'Long Term Debt And Capital Lease Obligation', 'Total Debt'], 1)
        sh0 = _statement_value(bal, ['Ordinary Shares Number', 'Share Issued'], 0)
        sh1 = _statement_value(bal, ['Ordinary Shares Number', 'Share Issued'], 1)
        cfo0 = _statement_value(cf, ['Operating Cash Flow', 'Cash Flow From Continuing Operating Activities'], 0)

        roa0 = ni0 / a0 if np.isfinite(ni0) and np.isfinite(a0) and a0 else np.nan
        roa1 = ni1 / a1 if np.isfinite(ni1) and np.isfinite(a1) and a1 else np.nan
        cr0 = ca0 / cl0 if np.isfinite(ca0) and np.isfinite(cl0) and cl0 else np.nan
        cr1 = ca1 / cl1 if np.isfinite(ca1) and np.isfinite(cl1) and cl1 else np.nan
        gm0 = gp0 / rev0 if np.isfinite(gp0) and np.isfinite(rev0) and rev0 else np.nan
        gm1 = gp1 / rev1 if np.isfinite(gp1) and np.isfinite(rev1) and rev1 else np.nan
        at0 = rev0 / ((a0 + a1) / 2) if all(np.isfinite(x) for x in [rev0,a0,a1]) and (a0+a1) else np.nan
        if all(np.isfinite(x) for x in [rev1,a1,a2]) and (a1+a2):
            at1 = rev1 / ((a1 + a2) / 2)
        else:
            at1 = rev1 / a1 if np.isfinite(rev1) and np.isfinite(a1) and a1 else np.nan

        checks = [
            roa0 > 0 if np.isfinite(roa0) else None,                                  # positive ROA
            cfo0 > 0 if np.isfinite(cfo0) else None,                                  # positive CFO
            roa0 > roa1 if np.isfinite(roa0) and np.isfinite(roa1) else None,          # improving ROA
            cfo0 > ni0 if np.isfinite(cfo0) and np.isfinite(ni0) else None,             # cash earnings quality
            debt0 < debt1 if np.isfinite(debt0) and np.isfinite(debt1) else None,       # lower leverage
            cr0 > cr1 if np.isfinite(cr0) and np.isfinite(cr1) else None,               # improving liquidity
            sh0 <= sh1 if np.isfinite(sh0) and np.isfinite(sh1) else None,              # no dilution
            gm0 > gm1 if np.isfinite(gm0) and np.isfinite(gm1) else None,               # improving margin
            at0 > at1 if np.isfinite(at0) and np.isfinite(at1) else None,               # improving asset turnover
        ]
        coverage = sum(x is not None for x in checks)
        score = sum(bool(x) for x in checks if x is not None)
        return {
            'Symbol': symbol,
            'Piotroski F-Score': float(score) if coverage >= 6 else np.nan,
            'F-Score Coverage': int(coverage),
        }
    except Exception:
        return {'Symbol': symbol, 'Piotroski F-Score': np.nan, 'F-Score Coverage': 0}


def enrich_piotroski(symbols, workers=6):
    symbols = list(dict.fromkeys(symbols))
    rows = []
    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(symbols)))) as ex:
        futures = {ex.submit(calculate_piotroski, s): s for s in symbols}
        for fut in as_completed(futures):
            try:
                rows.append(fut.result())
            except Exception:
                rows.append({'Symbol': futures[fut], 'Piotroski F-Score': np.nan, 'F-Score Coverage': 0})
    return pd.DataFrame(rows)


def add_entry_scores(df: pd.DataFrame, piotroski_floor=8):
    """Entry score uses cheap technical/participation features; Piotroski is optional enrichment."""
    if df.empty:
        return df
    out = df.copy()

    def pct(col, neutral=50):
        if col in out.columns:
            s = percentile_series(out[col])
            return s.fillna(neutral)
        return pd.Series(neutral, index=out.index, dtype=float)

    accum = out.get('Accumulation Score', pd.Series(0,index=out.index)).fillna(0)
    rs_nifty = pct('RS vs N500 20D %')
    rs_sector = pct('RS vs Sector 20D %') if 'RS vs Sector 20D %' in out.columns else pd.Series(50,index=out.index)
    rs_accel = pct('RS Acceleration') if 'RS Acceleration' in out.columns else pct('RS vs N500 5D %')
    del_abn = pct('Delivery Z') if 'Delivery Z' in out.columns else pct('Delivery Acceleration')
    vol_abn = pct('Volume Z') if 'Volume Z' in out.columns else pct('Volume Spike x')
    persist = pct('Delivery Persistence 10D') if 'Delivery Persistence 10D' in out.columns else pct('5D Volume x')

    # Entry location: reward near-breakout / moderate extension; penalize chasing.
    dist_high = out.get('Distance to 20D High %', pd.Series(-5,index=out.index)).fillna(-5)
    ext = out.get('Price Extension vs 20DMA %', pd.Series(0,index=out.index)).fillna(0)
    p5 = out.get('5D Price Change %', pd.Series(0,index=out.index)).fillna(0)
    loc = pd.Series(np.where((dist_high >= -4) & (dist_high <= 1) & (ext <= 8) & (p5 <= 7), 100,
                    np.where((ext > 12) | (p5 > 9), 20,
                    np.where((dist_high >= -8) & (ext <= 10), 70, 45))), index=out.index)

    contraction_ratio = out.get('Volatility Contraction', pd.Series(1,index=out.index)).fillna(1)
    contraction = pd.Series(np.where(contraction_ratio <= .8, 100,
                            np.where(contraction_ratio <= 1.05, 70, 40)), index=out.index)

    base = (
        0.20*accum +
        0.12*rs_nifty +
        0.10*rs_sector +
        0.10*rs_accel +
        0.10*del_abn +
        0.10*vol_abn +
        0.08*persist +
        0.10*loc +
        0.10*contraction
    )

    # Explicit extension penalty: strong stocks can still be poor entries.
    penalty = pd.Series(0.0, index=out.index)
    penalty += np.where(ext > 12, 12, np.where(ext > 8, 5, 0))
    penalty += np.where(p5 > 10, 8, np.where(p5 > 7, 4, 0))
    out['Extension Penalty'] = penalty
    out['Base Entry Score'] = (base - penalty).clip(0,100).round(1)

    if 'Piotroski F-Score' in out.columns:
        f_norm = (out['Piotroski F-Score'].fillna(4.5) / 9.0 * 100).clip(0, 100)
        out['Entry Suitability Score'] = (0.90*out['Base Entry Score'] + 0.10*f_norm).round(1)
    else:
        out['Entry Suitability Score'] = out['Base Entry Score']

    def entry_tag(r):
        score = r.get('Entry Suitability Score', 0)
        fs = r.get('Piotroski F-Score', np.nan)
        opp = str(r.get('Opportunity Type', r.get('Signal','')))
        supportive = (
            r.get('20D Avg Delivery %', 0) >= 40 and
            r.get('RS vs N500 20D %', -99) > 0 and
            r.get('RS Acceleration', r.get('RS vs N500 5D %', -99)) > 0 and
            r.get('Extension Penalty', 99) <= 5 and
            r.get('Volume Spike x', 0) >= 1.0
        )
        fundamental_ok = (not np.isfinite(fs)) or fs >= float(piotroski_floor)
        if score >= 80 and supportive and fundamental_ok and ('EXTENDED' not in opp) and ('DISTRIBUTION' not in opp):
            return '⭐ BEST ENTRY SETUP'
        if score >= 70 and supportive and ('EXTENDED' not in opp):
            return '🟢 ENTRY READY'
        if score >= 60:
            return '🟡 WATCH / WAIT'
        if 'EXTENDED' in opp:
            return '⚠️ EXTENDED'
        return 'NEUTRAL'
    out['Entry View'] = out.apply(entry_tag, axis=1)
    return out.sort_values(['Entry Suitability Score','Accumulation Score'], ascending=False)


@st.cache_data(ttl=30 * 60, show_spinner=False)
def fetch_stock_news(symbol: str):
    try:
        ticker = yf.Ticker(f'{symbol}.NS')
        news = ticker.news or []
    except Exception:
        news = []
    rows = []
    for item in news[:8]:
        content = item.get('content', {}) if isinstance(item, dict) else {}
        title = content.get('title') or item.get('title') or ''
        pub = content.get('provider', {}).get('displayName') or item.get('publisher') or ''
        link = content.get('canonicalUrl', {}).get('url') or content.get('clickThroughUrl', {}).get('url') or item.get('link') or ''
        if title:
            rows.append({'Title': title, 'Source': pub, 'Link': link})
    return rows


st.markdown("""
<style>
    .main-title {font-size: 2.2rem; font-weight: 700; margin-bottom: 0.1rem;}
    .subtitle {color: #6b7280; margin-bottom: 1rem;}
    .status-card {border-radius: 16px; padding: 18px 20px; color: white; margin-bottom: 1rem;}
    .status-title {font-size: 1.7rem; font-weight: 800; margin-bottom: 0.25rem;}
    .status-sub {font-size: 1rem; opacity: 0.95;}
    .small-note {color:#6b7280; font-size:0.9rem;}
    div[data-testid="stMetric"] {background:rgba(148,163,184,.055); border:1px solid rgba(148,163,184,.18); padding:10px 12px; border-radius:12px;}
    div[data-testid="stMetricLabel"] {font-weight:650;}
    div[data-testid="stVerticalBlockBorderWrapper"] {border-color:rgba(148,163,184,.22) !important; border-radius:12px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Market Intelligence Terminal</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Broad NSE discovery → market condition → sector leadership → accumulation → stock opportunity → news confirmation.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header('Settings')
    universe_mode = st.selectbox('Stock universe', ['Broad NSE EQ (~2,000)', 'Nifty 500 only'], index=0,
                                help='Broad mode scans ordinary NSE EQ-series stocks. Nifty-500 names are mapped immediately; missing sector/industry labels for final candidates are resolved during the quality gate without slowing the broad scan.')
    max_stocks = st.slider('Maximum stocks to scan', 500, 2200, 2000, 100,
                           help='2,000 gives broad discovery. Broad mode uses cached NSE bhavcopy, so the scan remains light; reduce only if your connection is slow.')
    min_traded_value_cr = st.select_slider('Minimum 20D avg traded value (₹ Cr)', options=[0, 0.25, 0.5, 1, 2, 5, 10], value=1,
                                           help='Liquidity safety filter. The app still downloads the broad universe, then removes very illiquid stocks from rankings.')
    delivery_lookback = st.slider('Delivery history days', 20, 60, 30, 5,
                                  help='Used for sector and stock delivery tracking.')
    top_sector_count = st.slider('Default sectors on chart', 3, 10, 5, 1,
                                 help='Fewer sectors make the delivery chart cleaner.')
    with st.expander('Classification quality', expanded=False):
        auto_enrich_profiles = st.checkbox('Auto-fill missing sector / industry', value=True,
                                           help='Uses structured Yahoo data first and Moneycontrol as a fallback only for the most relevant unmapped stocks.')
        classification_depth = st.select_slider('Auto-enrich top unmapped candidates', options=[25, 50, 75, 100, 150], value=50,
                                                help='Higher values improve classification coverage but may add some first-run lookup time. Results are cached for 7 days.')
    refresh = st.button('Refresh all data', use_container_width=True)
    st.caption('Fast broad mode uses NSE bhavcopy for ~2,000 stocks. Classification enrichment is cached and limited to relevant names.')

if refresh:
    st.cache_data.clear()

try:
    progress = st.progress(0, text='Opening dashboard…')
    with st.spinner('Loading core market and sector data…'):
        broad_meta = load_broad_nse_universe().head(max_stocks).copy()
        n500_meta = load_constituents().copy()
        # Core regime + sector engine only needs Nifty 500, avoiding a 2,000-stock Yahoo request.
        n500_universe = tuple(n500_meta['Ticker'].tolist())
        close, volume = download_market_frames(n500_universe, '1y')
        close = close.loc[:, [c for c in close.columns if c in set(n500_universe)]]
        volume = volume.reindex(columns=close.columns) if not volume.empty else pd.DataFrame(index=close.index)
        benchmark = download_ohlc(BENCHMARK, '1y')
        vix = download_ohlc(VIX, '1y')
        result, d = calc_regime(benchmark, vix, close, n500_meta)
    progress.progress(35, text='Market regime ready. Loading NSE broad-stock activity…')

    if universe_mode.startswith('Broad'):
        meta = broad_meta
        bhav_sessions = max(25, min(int(delivery_lookback), 35))
        broad_bhav = load_broad_bhav_history(tuple(meta['Symbol'].tolist()), sessions=bhav_sessions)
        stock_rank = broad_stock_summary(broad_bhav, meta, d['benchmark_close'], min_traded_value_cr=min_traded_value_cr)
        # Sector delivery is based on mapped Nifty-500 names; broad unmapped names still remain in stock discovery.
        mapped_delivery = broad_bhav.merge(n500_meta[['Symbol','Industry']], left_on='SYMBOL', right_on='Symbol', how='inner') if not broad_bhav.empty else pd.DataFrame()
        if not mapped_delivery.empty:
            delivery_long = mapped_delivery[['Date','SYMBOL','Industry','DELIV_PER']].rename(columns={'SYMBOL':'Symbol','DELIV_PER':'Delivery %'})
        else:
            delivery_long = pd.DataFrame()
    else:
        meta = n500_meta
        bhav_sessions = max(25, min(int(delivery_lookback), 35))
        n500_bhav = load_broad_bhav_history(tuple(meta['Symbol'].tolist()), sessions=bhav_sessions)
        stock_rank = broad_stock_summary(n500_bhav, meta, d['benchmark_close'], min_traded_value_cr=min_traded_value_cr)
        if not n500_bhav.empty:
            mapped_delivery = n500_bhav.merge(n500_meta[['Symbol','Industry']], left_on='SYMBOL', right_on='Symbol', how='inner')
            delivery_long = mapped_delivery[['Date','SYMBOL','Industry','DELIV_PER']].rename(columns={'SYMBOL':'Symbol','DELIV_PER':'Delivery %'})
        else:
            delivery_long = pd.DataFrame()

    # Auto-fill missing classifications only for the most relevant broad-market names.
    if auto_enrich_profiles and not stock_rank.empty:
        missing_mask = stock_rank['Industry'].isna() | stock_rank['Sector'].isna()
        missing_candidates = stock_rank[missing_mask].sort_values(
            ['Entry Suitability Score','Accumulation Score','Participation Conviction'], ascending=False
        ).head(int(classification_depth))
        if not missing_candidates.empty:
            progress.progress(62, text=f'Filling sector / industry for {len(missing_candidates)} relevant unmapped stocks…')
            prof = enrich_profiles(missing_candidates['Symbol'].tolist())
            stock_rank = apply_profile_enrichment(stock_rank, prof)

    progress.progress(75, text='Building sector delivery and relative-strength views…')
    sector_pivot, sector_smooth, sector_rank = summarize_sector_delivery(delivery_long)
    sector_rs, sector_rs5, sector_market_summary = build_sector_market_analytics(close, volume, n500_meta, d['benchmark_close'])
    sector_opportunity = merge_sector_scores(sector_rank, sector_market_summary)
    progress.progress(100, text=f'Ready — {len(stock_rank):,} liquid stocks ranked')
    progress.empty()
    if universe_mode.startswith('Broad') and 'broad_bhav' in locals() and not broad_bhav.empty:
        latest_session = pd.to_datetime(broad_bhav['Date']).max().date()
    elif 'n500_bhav' in locals() and not n500_bhav.empty:
        latest_session = pd.to_datetime(n500_bhav['Date']).max().date()
    else:
        latest_session = None

    state_color = color_for_state(result.dashboard_state)

    # Build a compact executive summary for the front page.
    top_sector_df = sector_opportunity.head(8).copy() if not sector_opportunity.empty else pd.DataFrame()
    top_stock_df = stock_rank.head(12).copy() if not stock_rank.empty else pd.DataFrame()

    positive_rs_stocks = int((stock_rank['RS vs N500 20D %'] > 0).sum()) if not stock_rank.empty else 0
    delivery_stocks = int((stock_rank['20D Avg Delivery %'] >= 50).sum()) if not stock_rank.empty else 0
    volume_stocks = int((stock_rank['Volume Spike x'] >= 1.3).sum()) if not stock_rank.empty else 0
    accumulation_stocks = int((stock_rank['Accumulation Score'] >= 70).sum()) if not stock_rank.empty else 0

    st.markdown(
        f'<div class="status-card" style="background:linear-gradient(135deg,{state_color},#111827);">'
        f'<div class="status-title">{result.dashboard_state}</div>'
        f'<div class="status-sub">{result.summary} &nbsp;•&nbsp; {result.posture}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if latest_session is not None:
        st.caption(f'Latest NSE session used for current-day % change: {latest_session.strftime("%d %b %Y")}')

    tabs = st.tabs(['Overview', 'Breadth', 'Industry Gain / Loss', 'Sector Delivery + Volume', 'Sector Relative Strength', 'Sector Stocks', 'Accumulation Stocks', 'Stock News'])

    with tabs[0]:
        # MARKET COMMAND CENTER
        st.markdown('### Market Command Center')
        st.caption('One-screen summary of regime, sector leadership, delivery/volume activity, relative strength, stock accumulation and news support.')

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric('Market Regime', result.dashboard_state.title())
        k2.metric('Regime Score', f'{result.score:.0f}/100')
        k3.metric('Warning', f'{result.warning:.0f}/100')
        k4.metric('India VIX', 'N/A' if np.isnan(d['vix']) else f"{d['vix']:.2f}",
                  'N/A' if np.isnan(d['vix_5d_pct']) else f"{d['vix_5d_pct']:+.1f}% / 5d")
        k5.metric('Actionable Candidates', f'{accumulation_stocks}', 'Score ≥ 70')

        st.markdown('#### Opportunity Funnel')
        mapped_count, mapped_total, mapped_pct = classification_coverage(stock_rank)
        st.caption(f"Scanning {len(meta):,} NSE EQ-series stocks. {mapped_count:,}/{mapped_total:,} ranked stocks ({mapped_pct:.0f}%) currently have both sector and industry labels. Official Nifty classification is used first; relevant missing labels are enriched from Yahoo and Moneycontrol fallback and cached.")
        f1, f2, f3, f4, f5 = st.columns(5)
        f1.metric('Universe', f'{len(stock_rank):,}' if not stock_rank.empty else '0')
        f2.metric('RS Positive', f'{positive_rs_stocks:,}')
        f3.metric('High Delivery', f'{delivery_stocks:,}')
        f4.metric('Volume Expanding', f'{volume_stocks:,}')
        f5.metric('High Accumulation', f'{accumulation_stocks:,}')

        # SECTOR SNAPSHOT — always visible on the front page.
        st.markdown('#### Sector Radar')
        render_sector_radar_cards(top_sector_df, limit=8)
        st.caption('For the full sortable sector table, open **Sector Delivery + Volume**. The Overview intentionally shows only the decision-useful summary.')

        # ROTATION SNAPSHOT
        st.markdown('#### Sector Rotation Snapshot')
        if not sector_opportunity.empty:
            rot = sector_opportunity.copy()
            def rotation_state(r):
                rs20 = r.get('Sector RS 20D %', np.nan)
                rs5 = r.get('Sector RS 5D %', np.nan)
                if pd.isna(rs20) or pd.isna(rs5):
                    return 'Unclear'
                if rs20 > 0 and rs5 > 0:
                    return 'LEADING'
                if rs20 <= 0 and rs5 > 0:
                    return 'EMERGING'
                if rs20 > 0 and rs5 <= 0:
                    return 'WEAKENING'
                return 'LAGGING'
            rot['Rotation'] = rot.apply(rotation_state, axis=1)
            leading = rot[rot['Rotation'] == 'LEADING'].head(4).index.tolist()
            emerging = rot[rot['Rotation'] == 'EMERGING'].sort_values('Sector RS 5D %', ascending=False).head(4).index.tolist()
            weakening = rot[rot['Rotation'] == 'WEAKENING'].head(4).index.tolist()
            lagging = rot[rot['Rotation'] == 'LAGGING'].head(4).index.tolist()
            rotation_items = [
                ('🟢 Leading', leading, 'Established leadership'),
                ('🔵 Emerging', emerging, 'Improving relative strength'),
                ('🟠 Weakening', weakening, 'Leadership losing momentum'),
                ('⚪ Lagging', lagging, 'Relative underperformance'),
            ]
            rot_cols = st.columns(4)
            for col, (title, names, note) in zip(rot_cols, rotation_items):
                with col:
                    with st.container(border=True):
                        st.markdown(f'**{title}**')
                        st.caption(note)
                        st.write(' · '.join(names) if names else 'None')

        # STOCK RADAR
        st.markdown('#### Stock Opportunity Radar')
        st.caption('Shortlist first: pinned columns show the setup, entry quality and accumulation score. Scroll right only when you want the supporting evidence.')
        if not top_stock_df.empty:
            stock_show = top_stock_df.copy().head(10)
            stock_show['TradingView'] = stock_show['Symbol'].map(tradingview_url)
            # Overview stays compact; deeper evidence remains in Sector Stocks / Accumulation tabs.
            radar_cols = ['Symbol','Signal','Entry Suitability Score','Accumulation Score','Sector','Today % Change','Latest Delivery %','Volume Spike x','RS vs N500 20D %','TradingView']
            radar_cols = [c for c in radar_cols if c in stock_show.columns]
            render_frozen_grid(
                stock_show[radar_cols],
                pinned_left=['Symbol','Signal','Entry Suitability Score','Accumulation Score'],
                score_columns=['Entry Suitability Score','Accumulation Score'],
                link_columns=['TradingView'],
                height=385,
                key='overview_stock_radar_grid',
            )
        else:
            st.info('Stock opportunity data is not available yet.')

        # NEWS-SUPPORTED ACTIVITY
        st.markdown('#### News-Supported Activity')
        st.caption('Recent news is used as confirmation after delivery, volume and RS identify the stock — not as the starting signal.')
        if not top_stock_df.empty:
            news_candidates = top_stock_df.head(5)['Symbol'].tolist()
            news_found = []
            for sym in news_candidates:
                rows = fetch_stock_news(sym)
                if rows:
                    news_found.append((sym, rows[0]))
            if news_found:
                for sym, n in news_found:
                    title = n['Title']
                    src = n['Source'] or 'Source unavailable'
                    link = n['Link']
                    if link:
                        st.markdown(f"**{sym}** — [{title}]({link}) · *{src}*")
                    else:
                        st.markdown(f"**{sym}** — {title} · *{src}*")
            else:
                st.info('No recent supporting headlines were found for the current top accumulation candidates.')

        # LIGHTWEIGHT MARKET INTERNALS, intentionally not the centre of the front page.
        with st.expander('Market internals snapshot'):
            q1, q2, q3, q4 = st.columns(4)
            q1.metric('% > 50 DMA', f"{d['b50']:.1f}%", f"{d['breadth_20d']:+.1f} pts / 20d")
            q2.metric('Advance / Decline', f"{d['ad_ratio']:.2f}", f"Advancers {d['adv_pct']:.1f}%")
            q3.metric('Sectors outperforming', f"{d['sector_outperform_pct']:.0f}%")
            q4.metric('NIFTY 500 20D', f"{d['nifty20d']:+.2f}%")
            st.caption('Detailed 20/50/200-DMA breadth remains in the Breadth tab.')

    with tabs[1]:
        st.subheader('Breadth structure')
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('% > 20 DMA', f"{d['b20']:.1f}%")
        c2.metric('% > 50 DMA', f"{d['b50']:.1f}%", f"{d['breadth_5d']:+.1f} pts / 5d")
        c3.metric('% > 200 DMA', f"{d['b200']:.1f}%")
        c4.metric('Advance / Decline', f"{d['ad_ratio']:.2f}", f"Advancers {d['adv_pct']:.1f}%")
        chart = pd.concat([d['benchmark_close'].rename('NIFTY 500'), d['breadth_hist'].rename('% above 50 DMA')], axis=1).dropna().tail(140)
        norm = chart / chart.iloc[0] * 100
        st.line_chart(norm, use_container_width=True)
        st.caption('Both lines are rebased to 100. If the index rises while breadth weakens, internals are deteriorating.')

    with tabs[2]:
        st.subheader('Industry-wise gain / loss')
        st.caption('Median current-session price change by industry. Median is used so one very large stock does not distort the whole industry.')
        mapped_now = stock_rank.dropna(subset=['Industry']).copy() if not stock_rank.empty else pd.DataFrame()
        if mapped_now.empty or 'Today % Change' not in mapped_now.columns:
            st.warning('Industry performance could not be prepared from the current scan.')
        else:
            industry_perf = mapped_now.groupby('Industry').agg(
                Stocks=('Symbol','count'),
                Today_Median=('Today % Change','median'),
                Advancers=('Today % Change', lambda x: (x > 0).mean()*100),
                Traded_Value_Cr=('Today Traded Value Cr','sum'),
                Avg_Delivery=('Latest Delivery %','mean'),
                Avg_Volume_Spike=('Volume Spike x','mean'),
            ).reset_index()
            industry_perf = industry_perf[industry_perf['Stocks'] >= 2].sort_values('Today_Median', ascending=False)
            top_count = st.slider('Industries to display', 10, min(40, max(10,len(industry_perf))), min(25,max(10,len(industry_perf))), 5, key='ind_count') if len(industry_perf) >= 10 else len(industry_perf)
            chart_df = pd.concat([industry_perf.head(max(1,top_count//2)), industry_perf.tail(max(1,top_count//2))]).drop_duplicates('Industry')
            chart = alt.Chart(chart_df).mark_bar(cornerRadiusEnd=4).encode(
                y=alt.Y('Industry:N', sort='-x', title=None),
                x=alt.X('Today_Median:Q', title='Median current-session change (%)'),
                color=alt.condition(alt.datum.Today_Median >= 0, alt.value('#22c55e'), alt.value('#ef4444')),
                tooltip=['Industry:N', alt.Tooltip('Today_Median:Q', format='+.2f', title='Today %'), alt.Tooltip('Advancers:Q', format='.0f', title='Advancers %'), alt.Tooltip('Traded_Value_Cr:Q', format='.1f', title='Traded value ₹Cr')]
            ).properties(height=max(420, 25*len(chart_df)))
            st.altair_chart(chart, use_container_width=True)
            show = industry_perf.rename(columns={'Today_Median':'Today Median %','Advancers':'Advancers %','Traded_Value_Cr':'Today Traded Value ₹Cr','Avg_Delivery':'Avg Delivery %','Avg_Volume_Spike':'Avg Volume Spike x'})
            render_frozen_grid(show, pinned_left=['Industry'], height=460, key='industry_gain_loss_grid')

    with tabs[3]:
        st.subheader('Sector delivery + volume activity')
        st.caption('Delivery is smoothed with a 5 DMA. Volume uses a value-traded proxy, so a sector with only a small price move can still stand out when activity surges.')
        if sector_smooth.empty:
            st.warning('Sector delivery data could not be loaded right now.')
        else:
            default_sectors = sector_opportunity.head(top_sector_count).index.tolist() if not sector_opportunity.empty else sector_rank.head(top_sector_count).index.tolist()
            selected = st.multiselect('Select sectors to display', options=sector_smooth.columns.tolist(), default=default_sectors, key='delivery_sectors')
            delivery_measure = st.radio('Delivery measurement', ['Excess vs 20D average (pp)', '5 DMA Delivery %', 'Delivery Z-score'], horizontal=True, key='delivery_measure')
            if selected:
                raw = sector_pivot[selected].copy()
                if delivery_measure == 'Excess vs 20D average (pp)':
                    frame = raw.rolling(5).mean() - raw.rolling(20).mean()
                    line_chart_with_scale(frame.dropna(how='all'), 'Delivery excess vs 20D average (percentage points)', log_scale=False)
                    st.caption('Above zero = sector delivery is running above its own 20-day normal. This removes the 40–60% compression seen in raw delivery lines.')
                elif delivery_measure == 'Delivery Z-score':
                    mu = raw.rolling(20).mean(); sd = raw.rolling(20).std().replace(0,np.nan)
                    frame = (raw.rolling(5).mean() - mu) / sd
                    line_chart_with_scale(frame.dropna(how='all'), 'Delivery Z-score', log_scale=False)
                    st.caption('Positive Z-score = unusually high delivery for that sector relative to its own recent history.')
                else:
                    scale_choice = st.radio('Raw delivery scale', ['Linear', 'Logarithmic'], horizontal=True, key='delivery_scale')
                    line_chart_with_scale(sector_smooth[selected].dropna(how='all'), '5 DMA Delivery %', log_scale=(scale_choice == 'Logarithmic'))
            else:
                st.info('Choose one or more sectors to display the delivery chart.')

            st.markdown('**Sector opportunity ranking — delivery + volume + relative strength**')
            rank_show = sector_opportunity.reset_index().rename(columns={'index': 'Industry'})
            show_cols = ['Industry', 'Sector Opportunity Score', 'Delivery Z Score', 'Latest Delivery %', '5D Avg Delivery %', 'Sector Volume Spike x', 'Sector 5D Volume x', 'Sector 20D Price %', 'Sector 5D Price %', 'Sector RS 20D %', 'Sector RS 5D %', '5D Change']
            show_cols = [c for c in show_cols if c in rank_show.columns]
            render_frozen_grid(
                rank_show[show_cols],
                pinned_left=['Industry','Sector Opportunity Score'],
                score_columns=['Sector Opportunity Score'],
                height=470,
                key='sector_delivery_volume_grid',
            )

            st.caption('Pinned columns stay visible while you scroll horizontally. Delivery Z above 0 means sector delivery is above its own recent normal; volume and RS must confirm before the opportunity score becomes strong.')

    with tabs[4]:
        st.subheader('Sector relative strength vs NIFTY 500')
        st.caption('Each line is an equal-weight sector index divided by NIFTY 500 and rebased to 100. Rising line = sector is strengthening relative to the market.')
        if sector_rs5.empty:
            st.warning('Sector relative-strength history could not be prepared.')
        else:
            rs_ranked = sector_opportunity.sort_values('Sector RS 20D %', ascending=False).index.tolist() if not sector_opportunity.empty else sector_rs5.columns.tolist()
            default_rs = [x for x in rs_ranked if x in sector_rs5.columns][:top_sector_count]
            show_all_rs = st.checkbox('Show all sectors on RS chart', value=False, key='all_rs')
            if show_all_rs:
                rs_selected = sector_rs5.columns.tolist()
                st.caption('All sectors selected. Use zoom/pan on the chart if the lines overlap.')
            else:
                rs_selected = st.multiselect('Select sectors for RS chart', options=sector_rs5.columns.tolist(), default=default_rs, key='rs_sectors')
            rs_window = st.radio('RS comparison window', [20, 60, 130], horizontal=True, index=1, key='rs_window')
            if rs_selected:
                raw_rs = sector_rs5[rs_selected].tail(int(rs_window)).dropna(how='all')
                if not raw_rs.empty:
                    base = raw_rs.apply(lambda c: c.dropna().iloc[0] if c.notna().any() else np.nan)
                    rs_perf = (raw_rs.divide(base, axis=1) - 1.0) * 100.0
                    line_chart_with_scale(rs_perf, 'Relative performance vs NIFTY 500 (%)', log_scale=False)
                    st.caption('0% is the start of the selected window. +3% means the sector has outperformed NIFTY 500 by about 3 percentage points over that window; negative values mean underperformance.')

            rs_table = sector_opportunity.reset_index().rename(columns={'index': 'Industry'})
            rs_cols = ['Industry', 'Sector RS 20D %', 'Sector RS 5D %', 'Sector Volume Spike x', '5D Avg Delivery %', 'Sector Opportunity Score']
            rs_cols = [c for c in rs_cols if c in rs_table.columns]
            st.dataframe(rs_table[rs_cols].sort_values('Sector RS 20D %', ascending=False).style.format({
                'Sector RS 20D %': '{:+.2f}',
                'Sector RS 5D %': '{:+.2f}',
                'Sector Volume Spike x': '{:.2f}x',
                '5D Avg Delivery %': '{:.1f}',
                'Sector Opportunity Score': '{:.1f}',
            }), use_container_width=True, hide_index=True)

    with tabs[5]:
        st.subheader('Stocks inside a selected sector')
        st.caption('Use the sector RS chart first, then drill into the stocks whose delivery, volume and stock-level RS are strengthening.')
        if stock_rank.empty:
            st.warning('Stock delivery data could not be loaded right now.')
        else:
            ranked_sectors = sector_opportunity.index.tolist() if not sector_opportunity.empty else []
            all_mapped = sorted(stock_rank['Industry'].dropna().astype(str).unique().tolist())
            sector_options = ranked_sectors + [x for x in all_mapped if x not in ranked_sectors]
            chosen_sector = st.selectbox('Choose sector / industry', options=sector_options, key='sector_drilldown')
            sector_stocks = stock_rank[stock_rank['Industry'] == chosen_sector].copy()
            sector_stocks = sector_stocks.sort_values(['Accumulation Score', 'Volume Spike x'], ascending=False)
            sector_stocks['TradingView'] = sector_stocks['Symbol'].map(tradingview_url)
            show_cols = ['Symbol', 'Signal', 'Accumulation Score', 'Entry Suitability Score', 'Sector', 'Industry', 'Today % Change', 'Today Traded Value Cr', 'Latest Delivery %', 'Delivery Z', '20D Avg Delivery %', 'Volume Spike x', 'Volume Z', '5D Volume x', 'RS vs N500 20D %', 'RS vs N500 5D %', '1M Price Change %', '5D Price Change %', 'TradingView']
            show_cols = [c for c in show_cols if c in sector_stocks.columns]
            render_frozen_grid(
                sector_stocks[show_cols],
                pinned_left=['Symbol','Signal','Accumulation Score'],
                score_columns=['Accumulation Score','Entry Suitability Score'],
                link_columns=['TradingView'],
                height=500,
                key='sector_stock_grid',
            )

            if not sector_stocks.empty:
                stock_choice = st.selectbox('View stock RS chart', options=sector_stocks['Symbol'].head(50).tolist(), key='stock_rs_choice')
                st.link_button('Open TradingView chart', tradingview_url(stock_choice), use_container_width=False)
                ticker = f'{stock_choice}.NS'
                if ticker in close.columns:
                    stock_px = close[ticker].dropna()
                    bench_px = d['benchmark_close'].reindex(stock_px.index).ffill().dropna()
                    stock_px = stock_px.reindex(bench_px.index)
                    rs_line = (stock_px / stock_px.iloc[0]) / (bench_px / bench_px.iloc[0]) * 100
                    rs_line = rs_line.rolling(5).mean().rename(stock_choice).to_frame().tail(130)
                    line_chart_with_scale(rs_line, 'Stock RS vs NIFTY 500', log_scale=False, height=320)

    with tabs[6]:
        st.subheader('Opportunity Funnel — accumulation to entry')
        st.caption('Fast scan first, deeper checks later. The expensive Piotroski/news layer only runs on the final shortlist so broad NSE speed is preserved.')
        if stock_rank.empty:
            st.warning('Accumulation stock data could not be loaded right now.')
        else:
            # Three-stage funnel overview.
            stage1 = stock_rank.copy()
            stage2 = stage1[
                (stage1['20D Avg Delivery %'] >= 40) &
                (stage1['Volume Spike x'] >= 1.0) &
                (stage1['Today Traded Value Cr'] >= float(min_traded_value_cr)) &
                (stage1['RS vs N500 20D %'] > 0)
            ].copy()
            stage3 = stage2[
                (stage2['RS Acceleration'].fillna(-99) > 0) &
                (stage2['Entry Suitability Score'] >= 60) &
                (~stage2['Opportunity Type'].astype(str).str.contains('EXTENDED|DISTRIBUTION', regex=True))
            ].copy()
            fc1, fc2, fc3, fc4 = st.columns(4)
            fc1.metric('Stage 1 · Liquid universe', f'{len(stage1):,}')
            fc2.metric('Stage 2 · Participation + RS', f'{len(stage2):,}')
            fc3.metric('Stage 3 · Quality setups', f'{len(stage3):,}')
            fc4.metric('Technical finalists', f'{len(stage3):,}')

            f1, f2, f3, f4 = st.columns(4)
            min_delivery = f1.slider('Minimum 20D delivery %', 20, 80, 40, 5)
            min_volume = f2.slider('Minimum volume spike', 0.5, 5.0, 1.0, 0.1)
            rs_only = f3.checkbox('Only positive 20D RS', value=True)
            only_actionable = f4.checkbox('Hide extended/distribution', value=True)

            filtered = stock_rank[(stock_rank['20D Avg Delivery %'] >= min_delivery) & (stock_rank['Volume Spike x'] >= min_volume) & (stock_rank['Today Traded Value Cr'] >= float(min_traded_value_cr))].copy()
            if rs_only:
                filtered = filtered[filtered['RS vs N500 20D %'] > 0]
            if only_actionable:
                filtered = filtered[~filtered['Opportunity Type'].astype(str).str.contains('EXTENDED|DISTRIBUTION', regex=True)]
            filtered = add_entry_scores(filtered)

            st.markdown('#### Technical pre-screen')
            st.caption('These are technical/participation candidates. You decide how strict the Piotroski fundamental gate should be below.')
            st.metric('Technical candidates awaiting fundamental gate', f'{len(filtered):,}')

            type_counts = filtered['Opportunity Type'].value_counts().rename_axis('Opportunity Type').reset_index(name='Stocks')
            st.markdown('#### Opportunity mix')
            st.dataframe(type_counts, use_container_width=True, hide_index=True)

            st.markdown('#### Final quality gate — choose your Piotroski threshold')
            st.caption('Fundamentals are fetched only for the final shortlist, so broad-NSE speed is preserved. Higher F-Score = stricter quality filter; lower it when you want a wider opportunity set.')
            cpi1, cpi2, cpi3 = st.columns([1.2, 1.4, 1])
            pi_min_score = cpi1.slider('Minimum Piotroski F-Score', 0, 9, 8, 1, key='pi_min_score')
            pi_count = cpi2.slider('Candidates to run through final quality gate', 5, 40, 15, 5)
            load_pi = cpi3.button('Run final quality gate', use_container_width=True)
            if 'final_quality_data' not in st.session_state:
                st.session_state['final_quality_data'] = pd.DataFrame()
            if load_pi and not filtered.empty:
                symbols = filtered.head(pi_count)['Symbol'].tolist()
                with st.spinner(f'Checking Piotroski + sector/industry for top {len(symbols)} candidates...'):
                    pio = enrich_piotroski(symbols)
                    prof = enrich_profiles(symbols)
                    quality = pio.merge(prof, on='Symbol', how='outer')
                    st.session_state['final_quality_data'] = quality

            quality = st.session_state.get('final_quality_data', pd.DataFrame())
            if isinstance(quality, pd.DataFrame) and not quality.empty:
                pio_cols = [c for c in ['Symbol','Piotroski F-Score','F-Score Coverage'] if c in quality.columns]
                prof_cols = [c for c in ['Symbol','Resolved Industry','Resolved Sector','Resolved Source'] if c in quality.columns]
                filtered = filtered.drop(columns=['Piotroski F-Score','F-Score Coverage'], errors='ignore').merge(quality[pio_cols], on='Symbol', how='inner')
                if len(prof_cols) > 1:
                    filtered = apply_profile_enrichment(filtered, quality[prof_cols])
                filtered = filtered[filtered['Piotroski F-Score'].notna() & (filtered['Piotroski F-Score'] >= float(pi_min_score))].copy()
                filtered = add_entry_scores(filtered, piotroski_floor=pi_min_score)
                filtered['TradingView'] = filtered['Symbol'].map(tradingview_url)
            else:
                filtered = pd.DataFrame()
                st.info(f'Run the final quality gate to see stocks meeting your selected Piotroski threshold (currently ≥ {pi_min_score}).')

            if not filtered.empty:
                st.success(f"{len(filtered)} candidate(s) passed Piotroski ≥ {pi_min_score} and the technical/participation filters.")
                top_n = st.slider('Rows to display', 5, 50, 20, 5, key='accum_rows')
                preferred = [
                    'Entry View','Opportunity Type','Symbol','Entry Suitability Score','Piotroski F-Score','Accumulation Score',
                    'Sector','Industry','Classification Source','Participation Conviction','Today % Change','Today Traded Value Cr',
                    'Latest Delivery %','20D Avg Delivery %','Delivery Z','Delivery Acceleration',
                    'Volume Spike x','Volume Z','5D Volume x','Delivery Persistence 10D','Volume Persistence 10D',
                    'RS vs N500 20D %','RS vs N500 5D %','RS Acceleration','RS vs Sector 20D %',
                    'Distance to 20D High %','Price Extension vs 20DMA %','Volatility Contraction',
                    '1M Price Change %','5D Price Change %','Extension Penalty','F-Score Coverage','TradingView'
                ]
                show_cols = [c for c in preferred if c in filtered.columns]
                display_df = filtered[show_cols].head(top_n).copy()
                render_frozen_grid(
                    display_df,
                    pinned_left=['Entry View','Opportunity Type','Symbol','Entry Suitability Score','Piotroski F-Score','Accumulation Score'],
                    score_columns=['Entry Suitability Score','Accumulation Score'],
                    link_columns=['TradingView'],
                    height=560,
                    key='accumulation_quality_grid',
                )
                chart_pick = st.selectbox('TradingView / detailed chart stock', options=filtered['Symbol'].head(30).tolist(), key='accum_chart_pick')
                st.link_button('Open selected stock in TradingView', tradingview_url(chart_pick))

                best = filtered[filtered['Entry View'] == '⭐ BEST ENTRY SETUP'].head(5)
                early = filtered[filtered['Opportunity Type'] == '🟣 EARLY ACCUMULATION'].head(10)
                setup = filtered[filtered['Opportunity Type'] == '🔵 SETUP READY'].head(10)
                if not best.empty:
                    st.success('Best entry setups: ' + ', '.join(best['Symbol'].tolist()))
                if not early.empty:
                    st.info('Early accumulation: ' + ', '.join(early['Symbol'].tolist()))
                if not setup.empty:
                    st.info('Setup ready: ' + ', '.join(setup['Symbol'].tolist()))

            st.caption('Key interpretation: positive Delivery/Volume Z means activity is unusually high for that stock; RS Acceleration identifies improving leadership; values below 1.0 in Volatility Contraction indicate short-term volatility is contracting versus its 20-day norm. Piotroski is a selectable quality gate, not the trigger.')



    with tabs[7]:
        st.subheader('Stock news support')
        st.caption('Use news as confirmation, not as a substitute for the delivery + volume + RS signal.')
        if stock_rank.empty:
            st.warning('No stock list available for news lookup.')
        else:
            top_symbols = stock_rank.sort_values('Accumulation Score', ascending=False)['Symbol'].head(100).tolist()
            chosen_symbol = st.selectbox('Choose stock', options=top_symbols, key='news_stock')
            chosen_row = stock_rank[stock_rank['Symbol'] == chosen_symbol].head(1)
            if not chosen_row.empty:
                info_cols = st.columns(5)
                info_cols[0].metric('Sector / Industry', chosen_row['Sector'].iloc[0] if ('Sector' in chosen_row.columns and pd.notna(chosen_row['Sector'].iloc[0])) else chosen_row['Industry'].iloc[0])
                info_cols[1].metric('Delivery', f"{chosen_row['Latest Delivery %'].iloc[0]:.1f}%")
                info_cols[2].metric('Volume Spike', f"{chosen_row['Volume Spike x'].iloc[0]:.2f}x")
                info_cols[3].metric('RS 20D', f"{chosen_row['RS vs N500 20D %'].iloc[0]:+.2f}%")
                info_cols[4].metric('Accumulation', f"{chosen_row['Accumulation Score'].iloc[0]:.1f}")
                st.write(f"**Signal:** {chosen_row['Signal'].iloc[0]}")
                st.link_button('Open TradingView chart', tradingview_url(chosen_symbol))
            with st.spinner('Loading news...'):
                news_rows = fetch_stock_news(chosen_symbol)
            if not news_rows:
                st.info('No recent news found from the current news source for this stock.')
            else:
                for n in news_rows:
                    title = n['Title']
                    src = n['Source'] or 'Source unavailable'
                    link = n['Link']
                    if link:
                        st.markdown(f"- [{title}]({link}) — *{src}*")
                    else:
                        st.markdown(f"- {title} — *{src}*")

    st.markdown('<div class="small-note">Main idea: rank sectors using delivery + volume expansion + relative strength, drill into stocks with the same combination, and use news only as supporting confirmation.</div>', unsafe_allow_html=True)

except Exception as e:
    st.error(f'Unable to complete the market scan: {e}')
    st.markdown('''
**Try these fixes:**
1. Press **Refresh all data** once.
2. Broad mode now uses NSE bhavcopy. If NSE is temporarily unavailable, switch to **Nifty 500 only** and retry.
3. Retry after a few minutes if Yahoo or NSE temporarily blocks downloads.
4. If the issue continues, send me a screenshot of this error.
''')
