# ============================================================
#  LSTM STOCK PREDICTOR — Dark Mode Edition
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta
import requests
import warnings
import sys
import time as _time

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    TF_OK = True
except Exception as e:
    TF_OK = False
    st.error(f"TensorFlow failed: {e}")

import plotly.io as pio
pio.renderers.default = 'iframe'

# ============================================================
#  PAGE
# ============================================================
st.set_page_config(page_title="LSTM Stock Predictor",
                   page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

# ============================================================
#  HARDCODED DARK PALETTE
# ============================================================
P = dict(
    bg="#0b0f17", surface="#111721", surface2="#161d2a",
    border="#1f2937", text="#e5e7eb", text_dim="#9ca3af",
    accent="#5b8def", accent_dim="#3b5a99",
    good="#4ade80", bad="#f87171", warn="#fbbf24",
    purple="#a78bfa", teal="#2dd4bf",
    grid="rgba(255,255,255,0.06)", chart_bg="#111721",
    plotly_template="plotly_dark"
)

# ============================================================
#  GLOBAL CSS
# ============================================================
st.markdown(f"""
<style>
    .stApp {{ background-color: {P['bg']}; color: {P['text']}; }}
    section[data-testid="stSidebar"] > div {{
        background-color: {P['surface']};
    }}
    .main .block-container {{ padding-top: 1.2rem; max-width: 1400px; }}

    html, body, [class*="css"] {{
        color: {P['text']};
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    h1, h2, h3, h4, h5, h6 {{ color: {P['text']}; }}
    p, span, div, label {{ color: {P['text']}; }}

    .top-bar {{
        display:flex; justify-content: space-between; align-items: center;
        padding: 8px 0 16px 0; border-bottom: 1px solid {P['border']};
        margin-bottom: 16px;
    }}
    .top-bar .title {{
        font-size: 1.35rem; font-weight: 700;
        color: {P['text']}; letter-spacing: -0.3px;
    }}
    .top-bar .title span {{
        color: {P['text_dim']}; font-weight: 400; margin-left: 10px;
        font-size: 0.95rem;
    }}

    .stat {{
        background: {P['surface']};
        border: 1px solid {P['border']};
        border-radius: 10px; padding: 14px 16px;
    }}
    .stat .lbl {{
        color: {P['text_dim']}; font-size: 0.72rem;
        font-weight: 600; letter-spacing: 0.6px;
        text-transform: uppercase;
    }}
    .stat .val {{
        color: {P['text']}; font-size: 1.4rem;
        font-weight: 700; margin-top: 6px;
        letter-spacing: -0.4px;
    }}
    .stat .dlt {{ font-size: 0.82rem; font-weight: 600; margin-top: 2px; }}

    .sec {{
        font-size: 0.95rem; font-weight: 700;
        color: {P['text']}; letter-spacing: 0.2px;
        margin: 20px 0 10px 0;
        text-transform: uppercase;
    }}

    .stButton > button {{
        background: {P['accent']}; color: white;
        border: none; border-radius: 8px;
        font-weight: 600; letter-spacing: 0.2px;
        transition: background 0.15s;
    }}
    .stButton > button:hover {{
        background: {P['accent_dim']}; color: white;
    }}

    div[data-testid="stMetricValue"] {{ color: {P['text']}; font-size: 1.5rem; }}
    div[data-testid="stMetricLabel"] {{ color: {P['text_dim']}; }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px; border-bottom: 1px solid {P['border']};
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent; color: {P['text_dim']};
        font-weight: 600; border-radius: 6px 6px 0 0;
        padding: 8px 16px;
    }}
    .stTabs [aria-selected="true"] {{
        background: {P['surface']}; color: {P['accent']};
    }}

    .streamlit-expanderHeader {{ color: {P['text']}; font-weight: 600; }}
    .stSlider label, .stCheckbox label, .stRadio label {{
        color: {P['text']} !important;
    }}
    div[data-testid="stDataFrame"] {{ color: {P['text']}; }}

    /* Custom split bar */
    .split-bar {{
        display: flex; height: 56px;
        border-radius: 10px; overflow: hidden;
        border: 1px solid {P['border']};
        margin: 8px 0 16px 0;
    }}
    .split-bar .train {{
        background: {P['accent']};
        display: flex; align-items: center; justify-content: center;
        color: white; font-weight: 700; font-size: 0.9rem;
        transition: width 0.3s ease;
    }}
    .split-bar .test {{
        background: {P['good']};
        display: flex; align-items: center; justify-content: center;
        color: #052e16; font-weight: 700; font-size: 0.9rem;
        transition: width 0.3s ease;
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================
#  SIDEBAR
# ============================================================
# ============================================================
#  SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Controls")

    ticker_symbol = st.text_input(
        "Ticker", value="TSLA",
        help="e.g. AAPL, MSFT, GOOG, NVDA, SPY"
    ).upper().strip()

    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()

    # ------- Author / Connect section -------
    st.markdown("### 👤 Connect")

    st.markdown(f"""
    <style>
        .profile-link {{
            display: flex; align-items: center;
            padding: 10px 14px; margin-bottom: 8px;
            border-radius: 8px;
            background: {P['surface2']};
            border: 1px solid {P['border']};
            text-decoration: none !important;
            color: {P['text']} !important;
            transition: all 0.15s ease;
        }}
        .profile-link:hover {{
            background: {P['accent']};
            border-color: {P['accent']};
            transform: translateX(3px);
        }}
        .profile-link:hover span {{
            color: white !important;
        }}
        .profile-link .icon {{
            width: 22px; height: 22px;
            margin-right: 12px;
            display: inline-flex;
            align-items: center; justify-content: center;
            font-weight: 700;
            font-size: 13px;
            color: white;
            border-radius: 5px;
        }}
        .profile-link .label {{
            font-weight: 600;
            font-size: 0.88rem;
            color: {P['text']};
        }}
        .profile-link .handle {{
            font-size: 0.72rem;
            color: {P['text_dim']};
            margin-left: auto;
        }}
        .gh-icon  {{ background: #24292e; }}
        .li-icon  {{ background: #0a66c2; }}
        .kg-icon  {{ background: #20beff; }}
    </style>

    <a class="profile-link" href="https://github.com/BlamerX" target="_blank">
        <span class="icon gh-icon">GH</span>
        <span class="label">GitHub</span>
        <span class="handle">@BlamerX</span>
    </a>

    <a class="profile-link" href="https://www.linkedin.com/in/adarsh-kumar-374150171/" target="_blank">
        <span class="icon li-icon">in</span>
        <span class="label">LinkedIn</span>
        <span class="handle">Adarsh Kumar</span>
    </a>

    <a class="profile-link" href="https://www.kaggle.com/blamerx" target="_blank">
        <span class="icon kg-icon">K</span>
        <span class="label">Kaggle</span>
        <span class="handle">@blamerx</span>
    </a>
    """, unsafe_allow_html=True)

    st.divider()
    st.caption(f"Python {sys.version.split()[0]}")
    st.caption(f"yfinance {'✓' if YF_OK else '✗'}  ·  TF {'✓' if TF_OK else '✗'}")
    st.divider()
    st.caption("⚠️ Educational use only. Not financial advice.")

# ============================================================
#  DATA LOADERS
# ============================================================
@st.cache_data(ttl=1800, show_spinner=False)
def source_yahoo_v8(ticker):
    p1 = int(_time.mktime(pd.Timestamp("2005-01-01").timetuple()))
    p2 = int(_time.mktime(pd.Timestamp.today().timetuple()))
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"period1": p1, "period2": p2, "interval": "1d",
              "events": "div,split", "includeAdjustedClose": "true"}
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36"),
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://finance.yahoo.com/"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=25)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"
        data = r.json()
        result = data.get("chart", {}).get("result")
        if not result:
            return None, "no result"
        result = result[0]
        ts = result.get("timestamp")
        if not ts:
            return None, "no timestamps"
        q = result["indicators"]["quote"][0]
        df = pd.DataFrame({
            "Open": q.get("open"), "High": q.get("high"),
            "Low": q.get("low"), "Close": q.get("close"),
            "Volume": q.get("volume"),
        }, index=pd.to_datetime(ts, unit="s", utc=True).tz_localize(None))
        df = df.dropna(subset=["Close"])
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["Close"])
        df["Volume"] = df["Volume"].fillna(0)
        return (df, None) if len(df) >= 100 else (None, f"only {len(df)}")
    except Exception as e:
        return None, f"v8: {e}"


@st.cache_data(ttl=1800, show_spinner=False)
def source_yfinance(ticker):
    if not YF_OK:
        return None, "not installed"
    try:
        df = yf.download(ticker, start="2005-01-01", auto_adjust=True,
                         progress=False, threads=False)
        if df is None or df.empty:
            return None, "0 rows"
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["Close"])
        df["Volume"] = df["Volume"].fillna(0)
        return (df, None) if len(df) >= 100 else (None, f"only {len(df)}")
    except Exception as e:
        return None, f"yf: {e}"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_company_info(ticker):
    if not YF_OK:
        return {}
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


def load_data(ticker):
    errors = []
    for name, fn in [("Yahoo v8", source_yahoo_v8), ("yfinance", source_yfinance)]:
        df, err = fn(ticker)
        if df is not None:
            return df, name, None
        errors.append(f"{name}: {err}")
    return None, None, " | ".join(errors)


with st.spinner(f"Loading {ticker_symbol}..."):
    raw_data, source, err = load_data(ticker_symbol)

if raw_data is None or len(raw_data) < 100:
    st.error(f"Could not load **{ticker_symbol}**.")
    if err:
        with st.expander("Details"):
            st.code(err)
    st.stop()

info = fetch_company_info(ticker_symbol)

# ============================================================
#  TOP BAR
# ============================================================
name = info.get("longName") or info.get("shortName") or ticker_symbol
sector = info.get("sector", "—")
country = info.get("country", "—")

st.markdown(f"""
<div class="top-bar">
    <div class="title">{name} <span>{ticker_symbol} · {sector} · {country}</span></div>
    <div style="color:{P['text_dim']}; font-size:0.8rem;">
        {len(raw_data):,} rows · {raw_data.index.min().date()} → {raw_data.index.max().date()}
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
#  DATE CONTROLS
# ============================================================
with st.expander("🗓️  Date range & data controls", expanded=False):
    min_date = raw_data.index.min().date()
    max_date = raw_data.index.max().date()
    c1, c2, c3 = st.columns([2, 2, 1])

    presets = {"6M": 180, "1Y": 365, "2Y": 730, "5Y": 1825, "Max": None}
    with c3:
        st.markdown("**Presets**")
        preset = st.radio("Preset", list(presets.keys()),
                          index=3, horizontal=False, label_visibility="collapsed")

    default_days = presets[preset]
    default_start = max(min_date, max_date - timedelta(days=default_days)) if default_days else min_date

    with c1:
        start_date = st.date_input("Start", value=default_start,
                                   min_value=min_date, max_value=max_date)
    with c2:
        end_date = st.date_input("End", value=max_date,
                                 min_value=min_date, max_value=max_date)

if start_date >= end_date:
    st.error("Start must be before end.")
    st.stop()

data = raw_data.loc[str(start_date):str(end_date)].copy()
if len(data) < 100:
    st.error(f"Only {len(data)} rows — widen the range.")
    st.stop()

# ============================================================
#  STATS STRIP
# ============================================================
latest = float(data['Close'].iloc[-1])
prev = float(data['Close'].iloc[-2])
delta_pct = (latest - prev) / prev * 100
returns = data['Close'].pct_change().dropna()
annual_ret = returns.mean() * 252 * 100
annual_vol = returns.std() * np.sqrt(252) * 100
sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
high_52 = float(data['Close'].tail(252).max())
low_52 = float(data['Close'].tail(252).min())


def stat(col, lbl, val, dlt=None, color=None):
    dc = ""
    if dlt is not None:
        c = P['good'] if dlt >= 0 else P['bad']
        dc = f'<div class="dlt" style="color:{c}">{dlt:+.2f}%</div>'
    col.markdown(f"""
        <div class="stat">
            <div class="lbl">{lbl}</div>
            <div class="val" style="{f'color:{color}' if color else ''}">{val}</div>
            {dc}
        </div>""", unsafe_allow_html=True)


k1, k2, k3, k4, k5 = st.columns(5)
with k1: stat(st, "Price", f"${latest:,.2f}", delta_pct)
with k2: stat(st, "Ann. Return", f"{annual_ret:+.1f}%",
              color=P['good'] if annual_ret >= 0 else P['bad'])
with k3: stat(st, "Ann. Vol", f"{annual_vol:.1f}%")
with k4: stat(st, "Sharpe", f"{sharpe:.2f}")
with k5: stat(st, "52w Range", f"${low_52:,.0f}–${high_52:,.0f}")

# ============================================================
#  SHARED CHART LAYOUT HELPER
# ============================================================
def chart_layout(height=340, title=None, show_legend=False):
    return dict(
        template=P['plotly_template'],
        paper_bgcolor=P['chart_bg'], plot_bgcolor=P['chart_bg'],
        font=dict(color=P['text'], size=11),
        height=height,
        margin=dict(l=40, r=20, t=30 if title else 15, b=30),
        title=dict(text=title, x=0.01, xanchor='left',
                   font=dict(size=13, color=P['text'])) if title else None,
        showlegend=show_legend,
        xaxis=dict(gridcolor=P['grid'], zerolinecolor=P['grid']),
        yaxis=dict(gridcolor=P['grid'], zerolinecolor=P['grid'])
    )


# ============================================================
#  TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["📊  Charts", "📋  Statistics", "🤖  LSTM Predictor"])

# ------------------------------------------------------------
#  TAB 1 — CHARTS
# ------------------------------------------------------------
with tab1:
    # ============ PRICE ACTION ============
    st.markdown('<div class="sec">Price action</div>', unsafe_allow_html=True)

    tcols = st.columns(5)
    with tcols[0]: show_ma = st.checkbox("MA 50/200", value=True)
    with tcols[1]: show_bb = st.checkbox("Bollinger", value=False)
    with tcols[2]: show_vol = st.checkbox("Volume", value=True)
    with tcols[3]: log_sc = st.checkbox("Log scale", value=False)
    with tcols[4]: show_range = st.checkbox("Range slider", value=False)

    fig = make_subplots(
        rows=2 if show_vol else 1, cols=1, shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25] if show_vol else [1.0])

    fig.add_trace(go.Scatter(
        x=data.index, y=data['Close'], mode='lines', name='Close',
        line=dict(color=P['accent'], width=1.8),
        fill='tozeroy', fillcolor="rgba(91,141,239,0.08)",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>$%{y:.2f}<extra></extra>"
    ), row=1, col=1)

    if show_ma:
        if len(data) > 50:
            fig.add_trace(go.Scatter(
                x=data.index, y=data['Close'].rolling(50).mean(), name='MA 50',
                line=dict(color=P['warn'], width=1.2, dash='dot')), row=1, col=1)
        if len(data) > 200:
            fig.add_trace(go.Scatter(
                x=data.index, y=data['Close'].rolling(200).mean(), name='MA 200',
                line=dict(color=P['text_dim'], width=1.2, dash='dot')), row=1, col=1)

    if show_bb and len(data) > 20:
        ma20 = data['Close'].rolling(20).mean()
        sd20 = data['Close'].rolling(20).std()
        fig.add_trace(go.Scatter(
            x=data.index, y=ma20 + 2 * sd20, name='BB↑',
            line=dict(color=P['text_dim'], width=0.8, dash='dot')),
            row=1, col=1)
        fig.add_trace(go.Scatter(
            x=data.index, y=ma20 - 2 * sd20, name='BB↓',
            line=dict(color=P['text_dim'], width=0.8, dash='dot'),
            fill='tonexty', fillcolor="rgba(148,163,184,0.08)"),
            row=1, col=1)

    if show_vol:
        colors = [P['good'] if c >= o else P['bad']
                  for o, c in zip(data['Open'], data['Close'])]
        fig.add_trace(go.Bar(x=data.index, y=data['Volume'],
                             marker=dict(color=colors, opacity=0.5),
                             showlegend=False), row=2, col=1)

    fig.update_layout(**chart_layout(height=560 if show_vol else 440,
                                     show_legend=True))
    fig.update_layout(hovermode='x unified',
                      xaxis_rangeslider_visible=show_range,
                      legend=dict(orientation='h', y=1.08, x=1,
                                  xanchor='right', bgcolor='rgba(0,0,0,0)'))
    fig.update_yaxes(title="Price", type='log' if log_sc else 'linear',
                     row=1, col=1)
    if show_vol:
        fig.update_yaxes(title="Volume", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True, key="price_chart")

    # ============ RETURNS ROW ============
    st.markdown('<div class="sec">Returns</div>', unsafe_allow_html=True)
    L, R = st.columns(2)

    with L:
        fr = go.Figure(go.Histogram(
            x=returns, nbinsx=80,
            marker=dict(color=P['accent'], opacity=0.75, line=dict(width=0))))
        fr.update_layout(**chart_layout(height=280,
                                        title="Daily return distribution"))
        fr.update_layout(bargap=0.03)
        fr.update_xaxes(tickformat='.1%', title="Return")
        fr.update_yaxes(title="Frequency")
        st.plotly_chart(fr, use_container_width=True, key="ret_hist")

    with R:
        cum = (1 + returns).cumprod() - 1
        fc = go.Figure(go.Scatter(
            x=cum.index, y=cum * 100, mode='lines',
            line=dict(color=P['good'], width=1.8),
            fill='tozeroy', fillcolor="rgba(74,222,128,0.08)"))
        fc.add_hline(y=0, line_dash='dot', line_color=P['text_dim'])
        fc.update_layout(**chart_layout(height=280, title="Cumulative return"))
        fc.update_xaxes(title="Date")
        fc.update_yaxes(title="Return %")
        st.plotly_chart(fc, use_container_width=True, key="cum_ret")

    # ============ RSI + MACD ============
    st.markdown('<div class="sec">Technical indicators</div>',
                unsafe_allow_html=True)
    L2, R2 = st.columns(2)

    with L2:
        # RSI calculation
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        frsi = go.Figure()
        frsi.add_trace(go.Scatter(
            x=data.index, y=rsi, mode='lines',
            line=dict(color=P['purple'], width=1.8),
            name='RSI (14)',
            hovertemplate="<b>%{x|%d %b %Y}</b><br>RSI: %{y:.1f}<extra></extra>"))
        frsi.add_hline(y=70, line_dash='dash', line_color=P['bad'],
                       annotation_text="Overbought (70)",
                       annotation_position="top left",
                       annotation_font_color=P['bad'])
        frsi.add_hline(y=30, line_dash='dash', line_color=P['good'],
                       annotation_text="Oversold (30)",
                       annotation_position="bottom left",
                       annotation_font_color=P['good'])
        frsi.update_layout(**chart_layout(height=300, title="RSI (14-day)"))
        frsi.update_xaxes(title="Date")
        frsi.update_yaxes(title="RSI", range=[0, 100])
        st.plotly_chart(frsi, use_container_width=True, key="rsi")

    with R2:
        # MACD
        ema12 = data['Close'].ewm(span=12, adjust=False).mean()
        ema26 = data['Close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal

        fmacd = go.Figure()
        fmacd.add_trace(go.Bar(
            x=data.index, y=hist,
            marker=dict(color=[P['good'] if h >= 0 else P['bad'] for h in hist],
                        opacity=0.55),
            name='Histogram'))
        fmacd.add_trace(go.Scatter(
            x=data.index, y=macd, mode='lines',
            line=dict(color=P['accent'], width=1.6), name='MACD'))
        fmacd.add_trace(go.Scatter(
            x=data.index, y=signal, mode='lines',
            line=dict(color=P['warn'], width=1.6), name='Signal'))
        fmacd.update_layout(**chart_layout(height=300, title="MACD (12, 26, 9)",
                                          show_legend=True))
        fmacd.update_layout(legend=dict(orientation='h', y=1.08, x=1,
                                        xanchor='right', bgcolor='rgba(0,0,0,0)'))
        fmacd.update_xaxes(title="Date")
        fmacd.update_yaxes(title="Value")
        st.plotly_chart(fmacd, use_container_width=True, key="macd")

    # ============ DRAWDOWN ============
    st.markdown('<div class="sec">Drawdown</div>', unsafe_allow_html=True)
    cummax = data['Close'].cummax()
    dd = (data['Close'] - cummax) / cummax * 100

    fdd = go.Figure(go.Scatter(
        x=data.index, y=dd, mode='lines',
        line=dict(color=P['bad'], width=1.5),
        fill='tozeroy', fillcolor="rgba(248,113,113,0.12)",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Drawdown: %{y:.1f}%<extra></extra>"))
    fdd.update_layout(**chart_layout(height=280, title="Underwater chart (% from peak)"))
    fdd.update_xaxes(title="Date")
    fdd.update_yaxes(title="Drawdown (%)")
    st.plotly_chart(fdd, use_container_width=True, key="drawdown")

    # ============ YEARLY RETURNS ============
    st.markdown('<div class="sec">Yearly returns</div>', unsafe_allow_html=True)
    yearly = data['Close'].resample('YE').last().pct_change().dropna() * 100
    yearly.index = yearly.index.year

    if len(yearly) > 0:
        fyr = go.Figure(go.Bar(
            x=[str(y) for y in yearly.index],
            y=yearly.values,
            marker=dict(color=[P['good'] if v >= 0 else P['bad']
                                for v in yearly.values], opacity=0.85),
            text=[f"{v:+.1f}%" for v in yearly.values],
            textposition='outside',
            textfont=dict(color=P['text'], size=10),
            hovertemplate="<b>%{x}</b><br>Return: %{y:.2f}%<extra></extra>"))
        fyr.update_layout(**chart_layout(height=300, title="Calendar year returns"))
        fyr.update_xaxes(title="Year")
        fyr.update_yaxes(title="Return (%)")
        st.plotly_chart(fyr, use_container_width=True, key="yearly")

    # ============ MONTHLY HEATMAP ============
    st.markdown('<div class="sec">Monthly returns heatmap</div>',
                unsafe_allow_html=True)
    monthly = data['Close'].resample('ME').last().pct_change().dropna() * 100
    if len(monthly) > 6:
        mdf = pd.DataFrame({
            'Year': monthly.index.year,
            'Month': monthly.index.month,
            'Return': monthly.values
        })
        pivot = mdf.pivot(index='Year', columns='Month', values='Return')
        pivot = pivot.rename(columns={1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',
                                       6:'Jun',7:'Jul',8:'Aug',9:'Sep',
                                       10:'Oct',11:'Nov',12:'Dec'})
        heat = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=[str(y) for y in pivot.index],
            colorscale=[[0, '#7f1d1d'], [0.5, '#111721'], [1, '#14532d']],
            zmid=0,
            text=np.round(pivot.values, 1),
            texttemplate='%{text}%',
            textfont=dict(size=10, color='white'),
            hovertemplate="<b>%{y} %{x}</b><br>%{z:.2f}%<extra></extra>",
            colorbar=dict(
                title="%",
                tickfont=dict(color=P['text']),
                titlefont=dict(color=P['text']))
        ))
        heat.update_layout(
            template=P['plotly_template'],
            paper_bgcolor=P['chart_bg'], plot_bgcolor=P['chart_bg'],
            font=dict(color=P['text']),
            height=max(280, 32 * len(pivot)),
            margin=dict(l=50, r=20, t=20, b=30),
            xaxis=dict(side='top', color=P['text']),
            yaxis=dict(color=P['text'], autorange='reversed'))
        st.plotly_chart(heat, use_container_width=True, key="heatmap")

    # ============ ROLLING SHARPE ============
    st.markdown('<div class="sec">Rolling Sharpe (90-day)</div>',
                unsafe_allow_html=True)
    roll_mean = returns.rolling(90).mean() * 252
    roll_std = returns.rolling(90).std() * np.sqrt(252)
    roll_sharpe = roll_mean / roll_std

    fsh = go.Figure(go.Scatter(
        x=roll_sharpe.index, y=roll_sharpe.values, mode='lines',
        line=dict(color=P['teal'], width=1.6),
        fill='tozeroy', fillcolor="rgba(45,212,191,0.08)"))
    fsh.add_hline(y=0, line_dash='dot', line_color=P['text_dim'])
    fsh.add_hline(y=1, line_dash='dash', line_color=P['good'],
                  annotation_text="Good (1.0)",
                  annotation_position="top right",
                  annotation_font_color=P['good'])
    fsh.update_layout(**chart_layout(height=280, title="Rolling 90-day Sharpe ratio"))
    fsh.update_xaxes(title="Date")
    fsh.update_yaxes(title="Sharpe")
    st.plotly_chart(fsh, use_container_width=True, key="rolling_sharpe")

# ------------------------------------------------------------
#  TAB 2 — STATISTICS
# ------------------------------------------------------------
with tab2:
    st.markdown('<div class="sec">Key statistics</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)

    with a:
        st.caption("**Returns**")
        st.dataframe(pd.DataFrame({
            "Metric": ["Days", "Avg daily", "σ daily", "Ann. return",
                       "Ann. σ", "Best day", "Worst day", "Sharpe"],
            "Value": [f"{len(data):,}",
                      f"{returns.mean()*100:+.3f}%",
                      f"{returns.std()*100:.2f}%",
                      f"{annual_ret:+.1f}%",
                      f"{annual_vol:.1f}%",
                      f"{returns.max()*100:+.1f}%",
                      f"{returns.min()*100:+.1f}%",
                      f"{sharpe:.2f}"]
        }), use_container_width=True, hide_index=True, height=300)

    with b:
        st.caption("**Price**")
        st.dataframe(pd.DataFrame({
            "Metric": ["Last", "High", "Low", "Mean", "Median", "Std"],
            "Value": [f"${latest:,.2f}",
                      f"${data['Close'].max():,.2f}",
                      f"${data['Close'].min():,.2f}",
                      f"${data['Close'].mean():,.2f}",
                      f"${data['Close'].median():,.2f}",
                      f"${data['Close'].std():,.2f}"]
        }), use_container_width=True, hide_index=True, height=300)

    with c:
        st.caption("**Company**")
        mc = info.get("marketCap")
        mc_s = f"${mc/1e9:.1f}B" if mc else "—"
        pe = info.get("trailingPE")
        st.dataframe(pd.DataFrame({
            "Metric": ["Sector", "Industry", "Country", "Employees",
                       "Market cap", "P/E", "Beta"],
            "Value": [sector,
                      info.get("industry", "—"),
                      country,
                      f"{info.get('fullTimeEmployees', 0):,}" if info.get("fullTimeEmployees") else "—",
                      mc_s,
                      f"{pe:.2f}" if pe else "—",
                      f"{info.get('beta', 0):.2f}" if info.get("beta") else "—"]
        }), use_container_width=True, hide_index=True, height=300)

# ------------------------------------------------------------
#  TAB 3 — LSTM
# ------------------------------------------------------------
with tab3:
    st.markdown('<div class="sec">Model configuration</div>',
                unsafe_allow_html=True)

    g = st.columns(6)
    with g[0]: split_pct = st.slider("Train %", 50, 95, 80, 5)
    with g[1]: lookback = st.slider("Lookback", 20, 120, 60, 5)
    with g[2]: units = st.slider("LSTM units", 16, 96, 32, 8)
    with g[3]: dropout = st.slider("Dropout", 0.0, 0.5, 0.2, 0.05)
    with g[4]: batch = st.select_slider("Batch", [16, 32, 64, 128], value=32)
    with g[5]: lr = st.select_slider("LR", [0.0005, 0.001, 0.005, 0.01], value=0.001)

    g2 = st.columns(3)
    with g2[0]: epochs = st.slider("Max epochs", 10, 200, 50, 10)
    with g2[1]: n_future = st.slider("Forecast days", 1, 30, 5, 1)
    with g2[2]: patience = st.slider("Early stop patience", 3, 20, 7, 1)

    # ============================================
    #  SPLIT BAR — PURE HTML/CSS (no Plotly)
    # ============================================
    total_rows = len(data) - 1
    tr_live = int(total_rows * split_pct / 100)
    te_live = total_rows - tr_live
    tr_width = split_pct
    te_width = 100 - split_pct

    st.markdown(f"""
    <div class="split-bar">
        <div class="train" style="width: {tr_width}%;">
            Train · {tr_live:,} rows ({tr_width}%)
        </div>
        <div class="test" style="width: {te_width}%;">
            Test · {te_live:,} rows ({te_width}%)
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Predict button
    predict_btn = st.button("🚀  Train & Forecast",
                            type="primary", use_container_width=True)

    if predict_btn:
        if not TF_OK:
            st.error("TensorFlow unavailable.")
            st.stop()

        # ---- Prepare data ----
        close_prices = data['Close'].values.reshape(-1, 1)
        log_returns = np.log(close_prices[1:] / close_prices[:-1]).reshape(-1, 1)
        n = len(log_returns)
        split_row = int(n * split_pct / 100)

        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(log_returns[:split_row])
        scaled = scaler.transform(log_returns)

        X, y = [], []
        for i in range(lookback, len(scaled)):
            X.append(scaled[i - lookback:i, 0])
            y.append(scaled[i, 0])
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        split_xy = split_row - lookback
        if split_xy <= 0 or split_xy >= len(X):
            st.error("Split invalid for this lookback — try a wider range.")
            st.stop()

        X_train, X_test = X[:split_xy], X[split_xy:]
        y_train, y_test = y[:split_xy], y[split_xy:]

        # ============================================
        #  TRAINING UI
        # ============================================
        st.markdown('<div class="sec">🧠 Training</div>', unsafe_allow_html=True)

        progress_bar = st.progress(0, text="Preparing...")

        mrow = st.columns(4)
        m_epoch = mrow[0].empty()
        m_train = mrow[1].empty()
        m_val = mrow[2].empty()
        m_gap = mrow[3].empty()

        live_chart_ph = st.empty()

        model = Sequential([
            Input(shape=(lookback, 1)),
            LSTM(units),
            Dropout(dropout),
            Dense(max(8, units // 2), activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=lr),
                      loss='mean_squared_error')
        es = EarlyStopping(monitor='val_loss', patience=patience,
                           restore_best_weights=True)

        train_losses, val_losses = [], []

        class LiveCB(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                train_losses.append(logs.get('loss', 0))
                val_losses.append(logs.get('val_loss', 0))

                pct = min((epoch + 1) / epochs, 1.0)
                progress_bar.progress(
                    pct,
                    text=f"Epoch {epoch + 1} of {epochs}  ·  "
                         f"progress {pct * 100:.0f}%")

                m_epoch.markdown(
                    f'<div class="stat"><div class="lbl">Epoch</div>'
                    f'<div class="val">{epoch + 1}</div></div>',
                    unsafe_allow_html=True)
                m_train.markdown(
                    f'<div class="stat"><div class="lbl">Train loss</div>'
                    f'<div class="val" style="color:{P["accent"]}">'
                    f'{logs.get("loss", 0):.5f}</div></div>',
                    unsafe_allow_html=True)
                m_val.markdown(
                    f'<div class="stat"><div class="lbl">Val loss</div>'
                    f'<div class="val" style="color:{P["good"]}">'
                    f'{logs.get("val_loss", 0):.5f}</div></div>',
                    unsafe_allow_html=True)
                gap = logs.get("val_loss", 0) - logs.get("loss", 0)
                gc = P['good'] if gap >= 0 else P['bad']
                m_gap.markdown(
                    f'<div class="stat"><div class="lbl">Gap (val−train)</div>'
                    f'<div class="val" style="color:{gc}">{gap:+.5f}</div></div>',
                    unsafe_allow_html=True)

                lc = go.Figure()
                lc.add_trace(go.Scatter(
                    y=train_losses, name='Train',
                    mode='lines+markers',
                    line=dict(color=P['accent'], width=2),
                    marker=dict(size=5)))
                lc.add_trace(go.Scatter(
                    y=val_losses, name='Validation',
                    mode='lines+markers',
                    line=dict(color=P['good'], width=2),
                    marker=dict(size=5)))
                lc.update_layout(**chart_layout(height=260, show_legend=True))
                lc.update_layout(
                    legend=dict(orientation='h', y=1.15, x=1,
                                xanchor='right', bgcolor='rgba(0,0,0,0)'))
                lc.update_xaxes(title="Epoch")
                lc.update_yaxes(title="MSE Loss")
                live_chart_ph.plotly_chart(lc, use_container_width=True,
                                          key=f"live_{epoch}")

        hist = model.fit(
            X_train, y_train,
            epochs=epochs, batch_size=batch,
            validation_split=0.1,
            callbacks=[es, LiveCB()],
            verbose=0)

        progress_bar.progress(1.0, text=f"✅ Done — {len(train_losses)} epochs")

        # ============================================
        #  EVALUATION
        # ============================================
        ps = model.predict(X_test, verbose=0)
        pr = scaler.inverse_transform(ps).flatten()
        ar = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        prev_test = close_prices[split_row:-1, 0]
        actual_prices = prev_test * np.exp(ar)
        pred_prices = prev_test * np.exp(pr)

        rmse = np.sqrt(mean_squared_error(actual_prices, pred_prices))
        mae = mean_absolute_error(actual_prices, pred_prices)
        mape = np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100
        dir_acc = np.mean(np.sign(np.diff(actual_prices)) ==
                          np.sign(np.diff(pred_prices))) * 100
        naive_rmse = np.sqrt(mean_squared_error(actual_prices[1:],
                                                actual_prices[:-1]))
        beats = rmse < naive_rmse

        st.markdown('<div class="sec">Results</div>', unsafe_allow_html=True)

        r1, r2, r3, r4 = st.columns(4)
        with r1: stat(st, "RMSE", f"${rmse:.2f}",
                      color=P['good'] if beats else P['bad'])
        with r2: stat(st, "MAE", f"${mae:.2f}")
        with r3: stat(st, "MAPE", f"{mape:.2f}%")
        with r4: stat(st, "Dir. Accuracy", f"{dir_acc:.1f}%",
                      color=P['good'] if dir_acc > 52 else P['bad'])

        st.markdown('<div class="sec">vs Naive baseline</div>',
                    unsafe_allow_html=True)
        cmp_df = pd.DataFrame({
            "Model": ["LSTM", "Naive (tomorrow = today)"],
            "RMSE": [f"${rmse:.2f}", f"${naive_rmse:.2f}"],
            "Verdict": ["✅ Better" if beats else "❌ Worse", "—"]
        })
        st.dataframe(cmp_df, use_container_width=True, hide_index=True)

        st.markdown('<div class="sec">Actual vs Predicted</div>',
                    unsafe_allow_html=True)
        test_dates = data.index[split_row:split_row + len(actual_prices)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=test_dates, y=actual_prices, mode='lines', name='Actual',
            line=dict(color=P['accent'], width=2)))
        fig.add_trace(go.Scatter(
            x=test_dates, y=pred_prices, mode='lines', name='Predicted',
            line=dict(color=P['warn'], width=2, dash='dot')))
        fig.update_layout(**chart_layout(height=380, show_legend=True))
        fig.update_layout(
            hovermode='x unified',
            legend=dict(orientation='h', y=1.05, x=1, xanchor='right',
                        bgcolor='rgba(0,0,0,0)'))
        fig.update_xaxes(title="Date")
        fig.update_yaxes(title="Price ($)")
        st.plotly_chart(fig, use_container_width=True, key="avp")

        st.markdown('<div class="sec">Error analysis</div>',
                    unsafe_allow_html=True)
        e1, e2 = st.columns(2)
        with e1:
            residuals = actual_prices - pred_prices
            eh = go.Figure(go.Histogram(
                x=residuals, nbinsx=40,
                marker=dict(color=P['warn'], opacity=0.75)))
            eh.add_vline(x=0, line_dash='dot', line_color=P['text_dim'])
            eh.update_layout(**chart_layout(height=280, title="Prediction errors"))
            eh.update_xaxes(title="Error ($)")
            eh.update_yaxes(title="Count")
            st.plotly_chart(eh, use_container_width=True, key="errh")
        with e2:
            sc = go.Figure()
            sc.add_trace(go.Scatter(
                x=actual_prices, y=pred_prices, mode='markers',
                marker=dict(color=P['accent'], size=5, opacity=0.5),
                showlegend=False))
            lo = min(actual_prices.min(), pred_prices.min())
            hi = max(actual_prices.max(), pred_prices.max())
            sc.add_trace(go.Scatter(
                x=[lo, hi], y=[lo, hi], mode='lines',
                line=dict(color=P['bad'], dash='dash', width=1.5),
                name='Perfect'))
            sc.update_layout(**chart_layout(height=280,
                                            title="Actual vs Predicted scatter",
                                            show_legend=False))
            sc.update_xaxes(title="Actual ($)")
            sc.update_yaxes(title="Predicted ($)")
            st.plotly_chart(sc, use_container_width=True, key="scatter")

        # ============================================
        #  FORECAST
        # ============================================
        st.markdown('<div class="sec">Forecast</div>', unsafe_allow_html=True)

        last_w = scaled[-lookback:].reshape(1, lookback, 1)
        last_p = float(close_prices[-1, 0])
        futs = []
        cw = last_w.copy()
        for _ in range(int(n_future)):
            ns = model.predict(cw, verbose=0)[0, 0]
            nr = scaler.inverse_transform([[ns]])[0, 0]
            np_ = last_p * np.exp(nr)
            futs.append(np_)
            cw = np.append(cw[:, 1:, :], [[[ns]]], axis=1)
            last_p = np_

        sigma = float(np.std(residuals))
        fd_dates = pd.date_range(
            start=data.index[-1] + pd.Timedelta(days=1),
            periods=int(n_future), freq='B')
        upper = [p + sigma * np.sqrt(i + 1) for i, p in enumerate(futs)]
        lower = [p - sigma * np.sqrt(i + 1) for i, p in enumerate(futs)]

        ff = go.Figure()
        tail = data['Close'].tail(60)
        ff.add_trace(go.Scatter(
            x=tail.index, y=tail.values, mode='lines',
            name='Historical',
            line=dict(color=P['accent'], width=2)))
        ff.add_trace(go.Scatter(
            x=list(fd_dates) + list(fd_dates)[::-1],
            y=upper + lower[::-1],
            fill='toself', fillcolor="rgba(74,222,128,0.12)",
            line=dict(color='rgba(0,0,0,0)'),
            name='1σ band', showlegend=True, hoverinfo='skip'))
        ff.add_trace(go.Scatter(
            x=fd_dates, y=futs, mode='lines+markers',
            name='Forecast',
            line=dict(color=P['good'], width=2.5, dash='dash'),
            marker=dict(size=8, line=dict(color=P['chart_bg'], width=2))))
        ff.update_layout(**chart_layout(height=380, show_legend=True))
        ff.update_layout(
            hovermode='x unified',
            legend=dict(orientation='h', y=1.05, x=1, xanchor='right',
                        bgcolor='rgba(0,0,0,0)'))
        ff.update_xaxes(title="Date")
        ff.update_yaxes(title="Price ($)")
        st.plotly_chart(ff, use_container_width=True, key="forecast")

        forecast_df = pd.DataFrame({
            "Date": [d.strftime('%a, %d %b') for d in fd_dates],
            "Predicted": [f"${p:,.2f}" for p in futs],
            "Low (1σ)": [f"${l:,.2f}" for l in lower],
            "High (1σ)": [f"${h:,.2f}" for h in upper],
            "Δ": [f"{((p / close_prices[-1,0]) - 1) * 100:+.2f}%" for p in futs]
        })
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

        st.caption("⚠️ Educational demo only. Not financial advice.")