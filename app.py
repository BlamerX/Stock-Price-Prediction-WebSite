# ============================================================
#  LSTM STOCK PREDICTOR — Live data, multi-source fallback
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta
import requests
import io
import warnings
import sys
import time

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False

try:
    from curl_cffi import requests as curl_requests
    CURL_OK = True
except Exception:
    CURL_OK = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
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
st.set_page_config(page_title="LSTM Stock Predictor", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem; font-weight: 700;
        background: linear-gradient(135deg,#1e3c72 0%,#2a5298 60%,#6b8dd6 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header { color:#666; font-size:1rem; margin-top:-8px; margin-bottom:20px; }
    .stat-card {
        background: linear-gradient(135deg,#f5f7fa 0%,#e8ecf3 100%);
        padding:16px; border-radius:12px; border-left:5px solid #2a5298;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .stat-card.green{border-left-color:#43a047;}
    .stat-card.red{border-left-color:#e53935;}
    .stat-card.orange{border-left-color:#fb8c00;}
    .stat-label{color:#666;font-size:.78rem;font-weight:600;
                text-transform:uppercase;letter-spacing:.5px;}
    .stat-value{color:#1a1a1a;font-size:1.5rem;font-weight:700;margin-top:4px;}
    .stat-delta{font-size:.88rem;font-weight:600;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">📈 LSTM Stock Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Interactive time-series forecasting with deep learning</p>',
            unsafe_allow_html=True)

# ============================================================
#  SIDEBAR
# ============================================================
st.sidebar.header("⚙️ Configuration")

ticker_symbol = st.sidebar.text_input(
    "Stock Ticker", value="TSLA",
    help="Examples: AAPL, MSFT, GOOG, NVDA, SPY"
).upper().strip()

if st.sidebar.button("🔄 Refresh / Clear Cache"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()
st.sidebar.warning("⚠️ **Educational Use Only.** Not for real trading.")
st.sidebar.divider()
st.sidebar.markdown("**Environment**")
st.sidebar.write(f"• yfinance: {'✅' if YF_OK else '❌'}")
st.sidebar.write(f"• curl_cffi: {'✅' if CURL_OK else '❌'}")
st.sidebar.write(f"• TensorFlow: {'✅' if TF_OK else '❌'}")
st.sidebar.write(f"• Python: {sys.version.split()[0]}")

if not ticker_symbol or not ticker_symbol.isalpha():
    st.info("👈 Enter a valid ticker (letters only).")
    st.stop()

# ============================================================
#  DATA LOADERS — three live sources, tried in order
# ============================================================
@st.cache_data(ttl=1800, show_spinner=False)
def source_yahoo_v8(ticker):
    """
    Yahoo Finance v8 chart JSON API.
    Uses period1/period2 timestamps instead of range=max —
    range=max silently downgrades to monthly data for long windows.
    """
    import time as _time

    # Unix timestamps for 2005-01-01 → today (daily bars available)
    period1 = int(_time.mktime(pd.Timestamp("2005-01-01").timetuple()))
    period2 = int(_time.mktime(pd.Timestamp.today().timetuple()))

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "period1": period1,
        "period2": period2,
        "interval": "1d",          # force daily bars
        "events": "div,split",
        "includeAdjustedClose": "true",
    }
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36"),
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://finance.yahoo.com/",
    }
    try:
        r = requests.get(url, params=params, headers=headers, timeout=25)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"

        data = r.json()
        result = data.get("chart", {}).get("result")
        if not result:
            err_msg = data.get("chart", {}).get("error") or "no result"
            return None, f"Yahoo error: {err_msg}"

        result = result[0]
        ts = result.get("timestamp")
        if not ts:
            return None, "no timestamps"

        q = result["indicators"]["quote"][0]

        df = pd.DataFrame({
            "Open":   q.get("open"),
            "High":   q.get("high"),
            "Low":    q.get("low"),
            "Close":  q.get("close"),
            "Volume": q.get("volume"),
        }, index=pd.to_datetime(ts, unit="s", utc=True).tz_localize(None))

        # Drop rows with missing Close
        df = df.dropna(subset=["Close"])

        # Coerce to numeric
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["Close"])

        if "Volume" not in df.columns:
            df["Volume"] = 0
        df["Volume"] = df["Volume"].fillna(0)

        # Sanity check — reject if we got suspiciously few rows
        expected_min = 500   # if we requested 20 years, we should get thousands
        if len(df) < 100:
            return None, f"only {len(df)} rows returned"

        return df, None

    except Exception as e:
        return None, f"v8 error: {e}"

@st.cache_data(ttl=1800, show_spinner=False)
def source_yfinance(ticker):
    """Standard yfinance — matches your Kaggle notebook exactly."""
    if not YF_OK:
        return None, "yfinance not installed"
    try:
        kwargs = dict(start="2010-01-01", auto_adjust=True,
                      progress=False, threads=False)
        if CURL_OK:
            kwargs["session"] = curl_requests.Session(impersonate="chrome")
        df = yf.download(ticker, **kwargs)
        if df is None or df.empty:
            return None, "0 rows"
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "Close" not in df.columns:
            return None, "no Close"
        df = df.dropna(subset=["Close"])
        if "Volume" not in df.columns:
            df["Volume"] = 0
        df["Volume"] = df["Volume"].fillna(0)
        if len(df) < 100:
            return None, f"only {len(df)} rows"
        return df, None
    except Exception as e:
        return None, f"yfinance error: {e}"


@st.cache_data(ttl=1800, show_spinner=False)
def source_stooq(ticker):
    """Stooq CSV endpoint — a completely different provider."""
    try:
        sym = ticker.lower()
        if not sym.endswith(".us") and "." not in sym:
            sym = f"{sym}.us"
        url = "https://stooq.com/q/d/l/"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, params={"s": sym, "i": "d"},
                         headers=headers, timeout=20)
        if r.status_code != 200 or len(r.text) < 50:
            return None, f"HTTP {r.status_code}"
        if "No data" in r.text[:100] or "Exceeded" in r.text[:100]:
            return None, "no data / limited"
        df = pd.read_csv(io.StringIO(r.text))
        if df.empty or "Close" not in df.columns:
            return None, "empty CSV"
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["Close"])
        if "Volume" not in df.columns:
            df["Volume"] = 0
        df["Volume"] = df["Volume"].fillna(0)
        if len(df) < 100:
            return None, f"only {len(df)} rows"
        return df, None
    except Exception as e:
        return None, f"stooq error: {e}"


def load_data(ticker):
    """Try live sources in order of reliability from cloud IPs."""
    attempts = [
        ("Yahoo v8 API", source_yahoo_v8),
        ("yfinance", source_yfinance),
        ("Stooq", source_stooq),
    ]
    errors = []
    for name, fn in attempts:
        df, err = fn(ticker)
        if df is not None:
            return df, name, None
        errors.append(f"{name}: {err}")
    return None, None, " | ".join(errors)


with st.spinner(f"📥 Fetching **{ticker_symbol}**..."):
    raw_data, source, err = load_data(ticker_symbol)

if raw_data is None or len(raw_data) < 100:
    st.error(f"❌ Could not load **`{ticker_symbol}`**.")
    if err:
        with st.expander("🔧 What each source returned"):
            st.code(err)
    st.markdown(f"""
    ### 🔎 What's happening

    All three live sources failed:
    1. **Yahoo v8 API** — direct JSON endpoint
    2. **yfinance** — Python wrapper
    3. **Stooq** — different provider

    This usually means the app is running on a **cloud IP that Yahoo
    rate-limits**. Kaggle's VM is not rate-limited, which is why the
    same code works there.

    ### ✅ Try

    - **Click "🔄 Refresh / Clear Cache"** in the sidebar
    - **Wait 5–10 minutes** — Yahoo blocks are temporary
    - **Try a different ticker** (AAPL, MSFT, GOOG)
    - If it works on your laptop but not on Streamlit → it's the IP block
    """)
    st.stop()

st.caption(f"✅ Source: **{source}** · {len(raw_data):,} rows · "
           f"{raw_data.index.min().date()} → {raw_data.index.max().date()}")

# ============================================================
#  DATE RANGE
# ============================================================
min_date = raw_data.index.min().date()
max_date = raw_data.index.max().date()

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    start_date = st.date_input(
        "📅 Start Date",
        value=max(min_date, max_date - timedelta(days=365 * 5)),
        min_value=min_date, max_value=max_date)
with c2:
    end_date = st.date_input("📅 End Date", value=max_date,
                             min_value=min_date, max_value=max_date)
with c3:
    st.write(""); st.write("")
    st.caption(f"Full: {min_date} → {max_date}")

if start_date >= end_date:
    st.error("Start must be before end.")
    st.stop()

data = raw_data.loc[str(start_date):str(end_date)].copy()
if len(data) < 100:
    st.error(f"Only {len(data)} rows — widen the range.")
    st.stop()

# ============================================================
#  STAT CARDS
# ============================================================
latest = float(data['Close'].iloc[-1])
prev = float(data['Close'].iloc[-2])
delta_pct = (latest - prev) / prev * 100
returns = data['Close'].pct_change().dropna()
annual_ret = returns.mean() * 252 * 100
annual_vol = returns.std() * np.sqrt(252) * 100
high_52 = float(data['Close'].tail(252).max())
low_52 = float(data['Close'].tail(252).min())

def stat_card(col, label, value, delta=None, color=""):
    dh = ""
    if delta is not None:
        c = "#43a047" if delta >= 0 else "#e53935"
        a = "▲" if delta >= 0 else "▼"
        dh = f'<div class="stat-delta" style="color:{c}">{a} {delta:+.2f}%</div>'
    col.markdown(f"""<div class="stat-card {color}">
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>{dh}</div>""", unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
with k1: stat_card(st, "Current Price", f"${latest:,.2f}", delta_pct,
                   "green" if delta_pct >= 0 else "red")
with k2: stat_card(st, "Annual Return", f"{annual_ret:+.1f}%", None,
                   "green" if annual_ret >= 0 else "red")
with k3: stat_card(st, "Annual Volatility", f"{annual_vol:.1f}%", None, "orange")
with k4: stat_card(st, "52w High", f"${high_52:,.2f}")
with k5: stat_card(st, "52w Low", f"${low_52:,.2f}")

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
#  TABS
# ============================================================
tab_charts, tab_stats, tab_predict = st.tabs(
    ["📊 Charts", "📋 Statistics", "🤖 LSTM Predictor"])

# ------------------------------------------------------------
#  CHARTS
# ------------------------------------------------------------
with tab_charts:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.06, row_heights=[0.75, 0.25])
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines',
                             name='Close', line=dict(color='#1e88e5', width=2),
                             fill='tozeroy', fillcolor='rgba(30,136,229,0.08)',
                             hovertemplate="<b>%{x|%d %b %Y}</b><br>$%{y:.2f}<extra></extra>"),
                  row=1, col=1)
    if len(data) > 50:
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(50).mean(),
                                 name='MA 50', line=dict(color='#fb8c00', width=1.3)),
                      row=1, col=1)
    if len(data) > 200:
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(200).mean(),
                                 name='MA 200', line=dict(color='#8e24aa', width=1.3)),
                      row=1, col=1)
    vol_colors = ['#26a69a' if c >= o else '#ef5350'
                  for o, c in zip(data['Open'], data['Close'])]
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume',
                         marker=dict(color=vol_colors), showlegend=False),
                  row=2, col=1)
    fig.update_layout(title=f'{ticker_symbol} — Price & Volume',
                      template='plotly_white', height=620, hovermode='x unified',
                      xaxis_rangeslider_visible=False,
                      margin=dict(l=50, r=30, t=60, b=40),
                      legend=dict(orientation='h', y=1.02, x=1, xanchor='right'))
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    L, R = st.columns(2)
    with L:
        fr = go.Figure(go.Histogram(x=returns, nbinsx=100,
                                    marker=dict(color='#26a69a',
                                                line=dict(color='white', width=0.5)),
                                    opacity=0.85))
        fr.update_layout(title='Distribution of Daily Returns',
                         xaxis_title='Daily Return', yaxis_title='Frequency',
                         xaxis_tickformat='.1%', template='plotly_white',
                         height=350, bargap=0.02,
                         margin=dict(l=50, r=30, t=60, b=40))
        st.plotly_chart(fr, use_container_width=True)
    with R:
        fd = go.Figure()
        fd.add_trace(go.Scatter(x=returns.index, y=returns, mode='lines',
                                line=dict(color='#43a047', width=0.7)))
        fd.add_hline(y=0, line_dash='dot', line_color='#888')
        fd.update_layout(title='Daily Returns Over Time',
                         xaxis_title='Date', yaxis_title='Return',
                         yaxis_tickformat='.1%', template='plotly_white',
                         height=350, margin=dict(l=50, r=30, t=60, b=40))
        st.plotly_chart(fd, use_container_width=True)

# ------------------------------------------------------------
#  STATISTICS
# ------------------------------------------------------------
with tab_stats:
    st.subheader(f"📋 Full Statistics — {ticker_symbol}")
    a, b = st.columns(2)
    with a:
        st.markdown("**Return Statistics**")
        st.dataframe(pd.DataFrame({
            "Metric": ["Trading days", "Avg daily return", "Daily σ",
                       "Annualized return", "Annualized σ",
                       "Best day", "Worst day", "Sharpe (approx.)"],
            "Value": [f"{len(data):,}",
                      f"{returns.mean()*100:+.4f}%",
                      f"{returns.std()*100:.4f}%",
                      f"{annual_ret:+.2f}%",
                      f"{annual_vol:.2f}%",
                      f"{returns.max()*100:+.2f}%",
                      f"{returns.min()*100:+.2f}%",
                      f"{(returns.mean()/returns.std())*np.sqrt(252):.2f}"]
        }), use_container_width=True, hide_index=True)
    with b:
        st.markdown("**Price Statistics**")
        st.dataframe(pd.DataFrame({
            "Metric": ["Max close", "Min close", "Mean close",
                       "Median close", "Std. dev."],
            "Value": [f"${data['Close'].max():,.2f}",
                      f"${data['Close'].min():,.2f}",
                      f"${data['Close'].mean():,.2f}",
                      f"${data['Close'].median():,.2f}",
                      f"${data['Close'].std():,.2f}"]
        }), use_container_width=True, hide_index=True)
    with st.expander("📄 Raw OHLCV data (last 500 rows)"):
        st.dataframe(data.tail(500), use_container_width=True)

# ------------------------------------------------------------
#  LSTM
# ------------------------------------------------------------
with tab_predict:
    st.subheader("🤖 LSTM Price Prediction")
    st.markdown("""
    Trained on **log returns** with a **60-day lookback window**.
    We compare honestly against the **naive baseline** (*tomorrow = today*).
    """)

    ctrl1, ctrl2 = st.columns([2, 1])
    with ctrl1:
        split_pct = st.slider("Train / Test Split", 50, 95, 80, 5, format="%d%%")
    total_rows = len(data) - 1
    tr_live = int(total_rows * split_pct / 100)
    te_live = total_rows - tr_live
    with ctrl2:
        st.markdown("#### Live Split")
        st.metric("🟢 Train rows", f"{tr_live:,}")
        st.metric("🔵 Test rows", f"{te_live:,}")

    bar = go.Figure(go.Bar(
        x=[tr_live, te_live], y=['Split'], orientation='h',
        marker=dict(color=['#43a047', '#1e88e5']),
        text=[f"Train {tr_live:,}", f"Test {te_live:,}"],
        textposition='inside', insidetextanchor='middle',
        hovertemplate="%{text}<extra></extra>"))
    bar.update_layout(barmode='stack', height=120, template='plotly_white',
                      showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(showticklabels=False, showgrid=False),
                      yaxis=dict(showticklabels=False))
    st.plotly_chart(bar, use_container_width=True)

    st.divider()
    with st.expander("🔧 Advanced Parameters"):
        a1, a2, a3 = st.columns(3)
        with a1: LOOKBACK = st.number_input("Lookback (days)", 20, 120, 60, 5)
        with a2: n_future = st.number_input("Forecast days", 1, 30, 5, 1)
        with a3: epochs = st.number_input("Max epochs", 10, 200, 50, 5)

    st.divider()
    predict_btn = st.button("🚀 Train LSTM & Generate Predictions",
                            type="primary", use_container_width=True)

    if predict_btn:
        if not TF_OK:
            st.error("TensorFlow unavailable."); st.stop()

        close_prices = data['Close'].values.reshape(-1, 1)
        log_returns = np.log(close_prices[1:] / close_prices[:-1]).reshape(-1, 1)
        n = len(log_returns)
        split_row = int(n * split_pct / 100)

        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(log_returns[:split_row])
        scaled = scaler.transform(log_returns)

        X, y = [], []
        for i in range(LOOKBACK, len(scaled)):
            X.append(scaled[i - LOOKBACK:i, 0])
            y.append(scaled[i, 0])
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        split_xy = split_row - LOOKBACK
        if split_xy <= 0 or split_xy >= len(X):
            st.error("Split invalid — try a wider range.")
            st.stop()

        X_train, X_test = X[:split_xy], X[split_xy:]
        y_train, y_test = y[:split_xy], y[split_xy:]

        with st.spinner("🧠 Training LSTM..."):
            model = Sequential([
                Input(shape=(LOOKBACK, 1)),
                LSTM(32), Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1)])
            model.compile(optimizer='adam', loss='mean_squared_error')
            es = EarlyStopping(monitor='val_loss', patience=7,
                               restore_best_weights=True)
            pbar = st.progress(0, text="Training...")

            class CB(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    pbar.progress(min((epoch + 1) / epochs, 1.0),
                                  text=f"Epoch {epoch+1}/{epochs} — loss {logs.get('loss',0):.5f}")

            hist = model.fit(X_train, y_train, epochs=int(epochs),
                             batch_size=32, validation_split=0.1,
                             callbacks=[es, CB()], verbose=0)
            pbar.empty()

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
        naive_rmse = np.sqrt(mean_squared_error(actual_prices[1:], actual_prices[:-1]))

        st.success("✅ Training complete!")
        st.markdown("### 📊 Model Performance")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RMSE", f"${rmse:.2f}")
        m2.metric("MAE", f"${mae:.2f}")
        m3.metric("MAPE", f"{mape:.2f}%")
        m4.metric("Directional Accuracy", f"{dir_acc:.2f}%")

        st.dataframe(pd.DataFrame({
            "Model": ["LSTM", "Naive (t = t-1)"],
            "RMSE": [f"${rmse:.2f}", f"${naive_rmse:.2f}"],
            "Beats naive?": ["✅" if rmse < naive_rmse else "❌", "—"]
        }), use_container_width=True, hide_index=True)

        if dir_acc < 52:
            st.warning(f"⚠️ Directional accuracy **{dir_acc:.1f}%** near random.")
        elif dir_acc < 55:
            st.info(f"ℹ️ Directional accuracy **{dir_acc:.1f}%** — small edge.")
        else:
            st.success(f"🎯 Directional accuracy **{dir_acc:.1f}%**!")

        st.divider()
        st.markdown("### 📉 Actual vs Predicted (Test Set)")
        test_dates = data.index[split_row:split_row + len(actual_prices)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_dates, y=actual_prices, mode='lines',
                                 name='Actual', line=dict(color='#1e88e5', width=2)))
        fig.add_trace(go.Scatter(x=test_dates, y=pred_prices, mode='lines',
                                 name='LSTM Prediction',
                                 line=dict(color='#e53935', width=2, dash='dot')))
        fig.update_layout(title=f'{ticker_symbol} — Actual vs LSTM Predicted',
                          xaxis_title='Date', yaxis_title='Price (USD)',
                          template='plotly_white', height=450,
                          hovermode='x unified', margin=dict(l=50, r=30, t=60, b=40))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📈 Training Diagnostics")
        dL, dR = st.columns(2)
        with dL:
            fh = go.Figure()
            fh.add_trace(go.Scatter(y=hist.history['loss'], name='Train Loss',
                                    line=dict(color='#1e88e5', width=2)))
            fh.add_trace(go.Scatter(y=hist.history['val_loss'], name='Val Loss',
                                    line=dict(color='#e53935', width=2)))
            fh.update_layout(title='Training Curve (MSE)',
                             xaxis_title='Epoch', yaxis_title='Loss',
                             template='plotly_white', height=350,
                             margin=dict(l=50, r=30, t=60, b=40))
            st.plotly_chart(fh, use_container_width=True)
        with dR:
            res = actual_prices - pred_prices
            fr = go.Figure()
            fr.add_trace(go.Scatter(x=test_dates, y=res, mode='lines',
                                    line=dict(color='#fb8c00', width=1.2),
                                    fill='tozeroy', fillcolor='rgba(251,140,0,0.1)'))
            fr.add_hline(y=0, line_dash='dot', line_color='#888')
            fr.update_layout(title='Prediction Residuals',
                             xaxis_title='Date', yaxis_title='Error ($)',
                             template='plotly_white', height=350,
                             margin=dict(l=50, r=30, t=60, b=40))
            st.plotly_chart(fr, use_container_width=True)

        st.markdown(f"### 🔮 Next {int(n_future)}-Day Forecast")
        last_w = scaled[-LOOKBACK:].reshape(1, LOOKBACK, 1)
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
        fd = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1),
                           periods=int(n_future), freq='B')

        ff = go.Figure()
        tail = data['Close'].tail(60)
        ff.add_trace(go.Scatter(x=tail.index, y=tail.values, mode='lines',
                                name='Historical (last 60d)',
                                line=dict(color='#1e88e5', width=2)))
        ff.add_trace(go.Scatter(x=fd, y=futs, mode='lines+markers',
                                name='Forecast',
                                line=dict(color='#43a047', width=2, dash='dash'),
                                marker=dict(size=10)))
        ff.update_layout(title=f'{ticker_symbol} — Next {int(n_future)} Business Days',
                         xaxis_title='Date', yaxis_title='Price (USD)',
                         template='plotly_white', height=430,
                         hovermode='x unified', margin=dict(l=50, r=30, t=60, b=40))
        st.plotly_chart(ff, use_container_width=True)

        st.dataframe(pd.DataFrame({
            "Date": [d.strftime('%a, %d %b %Y') for d in fd],
            "Predicted Close": [f"${p:,.2f}" for p in futs],
            "Change": [f"{((p / close_prices[-1,0]) - 1) * 100:+.2f}%" for p in futs]
        }), use_container_width=True, hide_index=True)
        st.warning("**Disclaimer:** Educational demo only.")