"""
INTRADAY LONG (BUY) SCREENER - Enhanced UI with Full Control
============================================================
Scan for stocks showing upward momentum for intraday buying
Complete customization: upload files, custom tickers, range selection
"""

import streamlit as st
# ── yf_ratelimit shim ──────────────────────────────────────────
# Replaces direct yfinance calls with rate-limit-safe wrappers.
# DO NOT remove this block.
from yf_ratelimit import safe_ticker as _rl_ticker, safe_download as _rl_download, clear_cache as _rl_clear_cache

class _YFShim:
    """Makes existing yf.Ticker() / yf.download() calls use safe wrappers."""
    @staticmethod
    def Ticker(symbol, **_):
        return _rl_ticker(symbol)
    @staticmethod
    def download(tickers, **kwargs):
        return _rl_download(tickers, **kwargs)

yf = _YFShim()
# ── end shim ───────────────────────────────────────────────────

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import io

# ── Cached fetch helpers ────────────────────────────────────────
# Module-level (not class methods) so Streamlit can cache them across
# reruns/scans, and so the WHOLE scan list can be fetched in a small
# number of bulk yf.download() calls instead of one call per ticker.
# This is the single biggest speed lever: 300 tickers used to mean
# ~600 individual requests; now it's 2-6 bulk requests total.

_BULK_CHUNK_SIZE = 75   # tickers per yf.download() call
# Since threads=2 below caps yfinance's INTERNAL concurrency regardless
# of chunk size, a bigger chunk doesn't create a bigger burst — it just
# means fewer separate yf.download() calls, i.e. less _throttle()/retry
# overhead. This is the safe speed lever: raise chunk size, not
# concurrency. (Was cut to 40 originally alongside threads=True → 2;
# 75 keeps burst size the same while cutting total round-trips ~2x.)

def _bulk_download_impl(full_tickers: tuple, period: str, interval: str, chunk_workers: int = 3):
    """Bulk-fetch OHLCV for MANY tickers via yf.download(), chunked + parallel
    only at the chunk level (not per-ticker). Uncached — call via
    _bulk_download() (cached) or directly when you need a guaranteed fresh hit."""
    chunks = [full_tickers[i:i + _BULK_CHUNK_SIZE] for i in range(0, len(full_tickers), _BULK_CHUNK_SIZE)]

    def _dl(chunk):
        tickers_str = " ".join(chunk)
        try:
            # NOTE: don't pass progress= here — yf_ratelimit.safe_download()
            # hardcodes progress=False internally; passing it again raises
            # "got multiple values for keyword argument 'progress'".
            return yf.download(tickers_str, period=period, interval=interval,
                                group_by='ticker', threads=2,
                                auto_adjust=False)
        except TypeError:
            # Shim may not accept every kwarg — retry minimal.
            return yf.download(tickers_str, period=period, interval=interval)
        except Exception:
            return pd.DataFrame()

    if len(chunks) == 1:
        return _dl(chunks[0])

    frames = []
    with ThreadPoolExecutor(max_workers=max(1, min(chunk_workers, len(chunks)))) as executor:
        for df in executor.map(_dl, chunks):
            if df is not None and not df.empty:
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)


@st.cache_data(ttl=45, show_spinner=False)
def _bulk_download(full_tickers: tuple, period: str, interval: str, chunk_workers: int = 3):
    """Cached entry point — reuses a result from the last 45s for the same
    ticker set/timeframe. Skipped entirely by the scan handler when
    'Force fresh data' is on (it calls _bulk_download_impl directly instead)."""
    return _bulk_download_impl(full_tickers, period, interval, chunk_workers)


def _extract_ticker_df(batch_df: pd.DataFrame, full_ticker: str) -> pd.DataFrame:
    """Pull one ticker's OHLCV slice out of a bulk-downloaded multi-ticker frame."""
    if batch_df is None or batch_df.empty:
        return pd.DataFrame()
    if isinstance(batch_df.columns, pd.MultiIndex):
        if full_ticker not in batch_df.columns.get_level_values(0):
            return pd.DataFrame()
        return batch_df[full_ticker].dropna(how='all')
    # A single-ticker bulk call sometimes comes back with flat columns
    return batch_df.dropna(how='all')


@st.cache_data(ttl=45, show_spinner=False)
def _fetch_core_data(full_ticker: str):
    """Fallback single-ticker fetch — only hit if a ticker is missing from
    a bulk batch (rare, e.g. newly listed / delisted symbols)."""
    stock = yf.Ticker(full_ticker)
    intraday = stock.history(period='1d', interval='1m')
    daily = stock.history(period='5d', interval='1d')
    return intraday, daily


@st.cache_data(ttl=120, show_spinner=False)
def _fetch_chart_data(full_ticker: str, period: str, interval: str):
    """Fetch chart data for a given timeframe, cached so re-rendering
    doesn't re-hit yfinance."""
    stock = yf.Ticker(full_ticker)
    return stock.history(period=period, interval=interval)


class IntradayLongScreener:
    """Scan for intraday long/buy opportunities"""

    def __init__(self, params):
        """Initialize with all adjustable parameters"""
        self.min_volume = params.get('min_volume', 100000)
        self.min_price = params.get('min_price', 20)
        self.min_conditions = params.get('min_conditions', 4)
        self.min_score = params.get('min_score', 50)

        # Advanced thresholds
        self.price_change_threshold = params.get('price_change_threshold', 0.0)
        self.dist_from_low_threshold = params.get('dist_from_low_threshold', 2.0)
        self.trend_threshold = params.get('trend_threshold', 2.0)
        self.momentum_threshold = params.get('momentum_threshold', 0.5)
        self.volume_ratio_threshold = params.get('volume_ratio_threshold', 1.2)
        self.rsi_threshold = params.get('rsi_threshold', 35)
        self.atr_threshold = params.get('atr_threshold', 1.0)

        # Technical indicator settings
        self.rsi_period = params.get('rsi_period', 14)
        self.atr_period = params.get('atr_period', 14)
        self.momentum_window = params.get('momentum_window', 30)
        self.strong_score = params.get('strong_score', 70)

    def get_default_stock_list(self, exchange='NSE'):
        """Get default stock list from file"""
        try:
            file_path = 'nse.txt' if exchange == 'NSE' else 'bse.txt'
            with open(file_path, 'r') as f:
                tickers = [line.strip() for line in f if line.strip()]
            return tickers
        except Exception as e:
            st.error(f"Error reading {file_path}: {str(e)}")
            return ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']

    def analyze_stock(self, ticker, exchange='NSE', intraday=None, daily=None):
        """Analyze individual stock for long/buy setup.
        Pass pre-fetched `intraday`/`daily` (sliced from a bulk download) to
        avoid an extra network call — that's the fast path used by the scan.
        """
        try:
            suffix = '.NS' if exchange == 'NSE' else '.BO'
            full_ticker = f"{ticker}{suffix}"

            if intraday is None or daily is None or intraday.empty or daily.empty:
                intraday, daily = _fetch_core_data(full_ticker)

            if intraday.empty or daily.empty:
                return None

            current_price = intraday['Close'].iloc[-1]
            open_price = intraday['Open'].iloc[0]
            high_price = intraday['High'].max()
            low_price = intraday['Low'].min()
            volume = intraday['Volume'].sum()

            if current_price < self.min_price or volume < self.min_volume:
                return None

            # For buying: price change should be positive (rising from open)
            price_change_pct = ((current_price - open_price) / open_price) * 100
            # Distance from day low (want price to be near the day low for bounce, or breaking out from low)
            dist_from_low = ((current_price - low_price) / low_price) * 100

            if len(daily) >= 2:
                recent_change = ((daily['Close'].iloc[-1] - daily['Close'].iloc[0]) / daily['Close'].iloc[0]) * 100
            else:
                recent_change = 0

            if len(intraday) >= self.momentum_window * 2:
                last_n = intraday['Close'].iloc[-self.momentum_window:].mean()
                prev_n = intraday['Close'].iloc[-self.momentum_window * 2:-self.momentum_window].mean()
                momentum_change = ((last_n - prev_n) / prev_n) * 100
            else:
                momentum_change = 0

            avg_volume_5d = daily['Volume'].mean()
            volume_ratio = volume / avg_volume_5d if avg_volume_5d > 0 else 0

            rsi = self.calculate_rsi(intraday['Close'], period=self.rsi_period)
            atr = self.calculate_atr(intraday, period=self.atr_period)
            atr_pct = (atr / current_price) * 100

            # ---- BUY conditions (opposite of short screener) ----
            conditions_met = []

            # Price rising from open
            if price_change_pct > self.price_change_threshold:
                conditions_met.append("Up from open")
            elif price_change_pct >= -0.5:
                conditions_met.append("Flat/recovering")

            # Price not too far from day low (potential breakout zone OR early momentum)
            if dist_from_low < self.dist_from_low_threshold:
                conditions_met.append("Near day low / bounce zone")

            # Positive 5-day trend
            if recent_change > self.trend_threshold:
                conditions_met.append("5-day uptrend")

            # Positive intraday momentum
            if momentum_change > self.momentum_threshold:
                conditions_met.append("Positive momentum")

            # High volume confirms the move
            if volume_ratio > self.volume_ratio_threshold:
                conditions_met.append("High volume")

            # RSI oversold → bounce opportunity
            if rsi and rsi < self.rsi_threshold:
                conditions_met.append("RSI oversold")

            # Good volatility for intraday moves
            if atr_pct > self.atr_threshold:
                conditions_met.append("Good volatility")

            if len(conditions_met) < self.min_conditions:
                return None

            # ---- Score calculation ----
            score = 0

            if price_change_pct > 2:
                score += 30
            elif price_change_pct > 1:
                score += 20
            elif price_change_pct > 0:
                score += 10

            # Near day low is a good buying zone
            if dist_from_low < 1:
                score += 20
            elif dist_from_low < 2:
                score += 10

            if recent_change > 5:
                score += 20
            elif recent_change > 2:
                score += 10

            if momentum_change > 1:
                score += 15
            elif momentum_change > 0.5:
                score += 8

            if volume_ratio > 1.5:
                score += 10
            elif volume_ratio > 1.2:
                score += 5

            if rsi and rsi < 30:
                score += 5
            elif rsi and rsi < 35:
                score += 3

            return {
                'ticker': ticker,
                'full_ticker': full_ticker,
                'price': current_price,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'change_pct': price_change_pct,
                'volume': volume,
                'volume_ratio': volume_ratio,
                'dist_from_low': dist_from_low,
                'recent_trend': recent_change,
                'momentum': momentum_change,
                'rsi': rsi if rsi else 0,
                'atr_pct': atr_pct,
                'score': score,
                'conditions': ', '.join(conditions_met),
                'signal_strength': 'STRONG' if score >= self.strong_score else 'MODERATE' if score >= 50 else 'WEAK',
            }

        except Exception as e:
            return None

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1]
        except:
            return None

    def calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        try:
            high = df['High']
            low = df['Low']
            close = df['Close']
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean().iloc[-1]
            return atr
        except:
            return 0


def show_intraday_long_screener():
    """Display the intraday long/buy screener"""

    st.markdown("""
        <style>
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        h1 { font-size: 1.8rem !important; margin-bottom: 0.5rem !important; }
        h2 { font-size: 1.3rem !important; margin-top: 1rem !important; margin-bottom: 0.5rem !important; }
        h3 { font-size: 1.1rem !important; }
        .stMetric label { font-size: 0.85rem !important; }
        .stMetric .metric-value { font-size: 1.2rem !important; }
        </style>
    """, unsafe_allow_html=True)

    st.title("📈 Intraday Long (Buy) Screener")

    # ── Stock Selection ──────────────────────────────────────────────
    st.markdown("#### 📋 Stock Selection")

    col1, col2 = st.columns([1, 2])

    with col1:
        exchange = st.selectbox("Exchange", ["NSE", "BSE"], key="long_exchange")

    with col2:
        stock_source = st.radio(
            "Stock Source",
            ["Default List", "Upload File", "Custom Tickers"],
            horizontal=True,
            key="long_stock_source"
        )

    stock_list = []

    if stock_source == "Upload File":
        uploaded_file = st.file_uploader(
            f"Upload {exchange} ticker file (one ticker per line)",
            type=['txt'],
            help="Upload a text file with one ticker symbol per line",
            key="long_upload"
        )
        if uploaded_file:
            content = uploaded_file.read().decode('utf-8')
            stock_list = [line.strip() for line in content.split('\n') if line.strip()]
            st.success(f"✅ Loaded {len(stock_list)} tickers from file")
        else:
            st.info("👆 Please upload a ticker file")

    elif stock_source == "Custom Tickers":
        custom_input = st.text_area(
            "Enter tickers (comma or newline separated)",
            placeholder="RELIANCE, TCS, INFY\nor\nRELIANCE\nTCS\nINFY",
            height=100,
            key="long_custom"
        )
        if custom_input:
            if ',' in custom_input:
                stock_list = [t.strip().upper() for t in custom_input.split(',') if t.strip()]
            else:
                stock_list = [t.strip().upper() for t in custom_input.split('\n') if t.strip()]
            st.success(f"✅ {len(stock_list)} tickers entered")
        else:
            st.info("👆 Enter ticker symbols")

    else:  # Default List
        screener_temp = IntradayLongScreener({})
        stock_list = screener_temp.get_default_stock_list(exchange)
        st.info(f"📊 Using default list with {len(stock_list)} stocks")

    # ── Range Selection ──────────────────────────────────────────────
    if stock_list:
        st.markdown("#### 🎯 Scan Range")
        col1, col2, col3 = st.columns(3)

        with col1:
            start_index = st.number_input(
                "Start from index",
                min_value=0,
                max_value=len(stock_list) - 1,
                value=0,
                help=f"First ticker: {stock_list[0] if stock_list else 'N/A'}",
                key="long_start"
            )

        with col2:
            end_index = st.number_input(
                "End at index",
                min_value=start_index,
                max_value=len(stock_list) - 1,
                value=min(start_index + 29, len(stock_list) - 1),
                help="Last ticker in range",
                key="long_end"
            )

        with col3:
            scan_count = end_index - start_index + 1
            st.metric("Stocks to Scan", scan_count)

        if scan_count > 0:
            preview_list = stock_list[start_index:end_index + 1]
            st.caption(f"**Scan Range:** {preview_list[0]} to {preview_list[-1]} ({scan_count} stocks)")
            with st.expander("Preview ticker list"):
                st.write(", ".join(preview_list[:50]))
                if len(preview_list) > 50:
                    st.caption(f"... and {len(preview_list) - 50} more")

    st.markdown("---")

    # ── Screening Parameters ─────────────────────────────────────────
    st.markdown("#### ⚙️ Screening Parameters")

    with st.expander("Basic Filters", expanded=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            min_price = st.number_input("Min Price (₹)", min_value=1, max_value=500, value=20, step=5, key="long_min_price")

        with col2:
            min_volume = st.number_input("Min Volume", min_value=10000, max_value=10000000, value=100000, step=10000, key="long_min_vol")

        with col3:
            min_conditions = st.slider("Min Conditions (out of 7)", min_value=2, max_value=7, value=4, key="long_min_cond")

        with col4:
            min_score = st.slider("Min Score (0-100)", min_value=20, max_value=90, value=50, step=5, key="long_min_score")

    with st.expander("Advanced Thresholds"):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            price_change_threshold = st.slider("Price Change (%)", min_value=-1.0, max_value=5.0, value=0.0, step=0.5, key="long_pc")
            momentum_threshold = st.slider("Momentum (%)", min_value=0.0, max_value=5.0, value=0.5, step=0.1, key="long_mom")

        with col2:
            dist_from_low_threshold = st.slider("Dist from Low (%)", min_value=0.5, max_value=10.0, value=2.0, step=0.5, key="long_dfl")
            volume_ratio_threshold = st.slider("Volume Ratio", min_value=1.0, max_value=3.0, value=1.2, step=0.1, key="long_vr")

        with col3:
            trend_threshold = st.slider("5-Day Trend (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.5, key="long_trend")
            rsi_threshold = st.slider("RSI Oversold", min_value=20, max_value=50, value=35, step=5, key="long_rsi")

        with col4:
            atr_threshold = st.slider("ATR % Threshold", min_value=0.5, max_value=5.0, value=1.0, step=0.1, key="long_atr")

    with st.expander("Technical Indicators & Trading Settings"):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            rsi_period = st.number_input("RSI Period", min_value=5, max_value=50, value=14, step=1, key="long_rsi_p")
            atr_period = st.number_input("ATR Period", min_value=5, max_value=50, value=14, step=1, key="long_atr_p")

        with col2:
            momentum_window = st.number_input("Momentum Window (min)", min_value=10, max_value=120, value=30, step=5, key="long_mw")
            max_workers = st.number_input("Bulk Download Chunk Workers", min_value=1, max_value=2, value=2, step=1, key="long_workers",
                                           help="Tickers are fetched in bulk batches of 40. This used to allow up to 6 "
                                                "parallel batches, which is what was bursting past Yahoo's real rate "
                                                "limiter and crashing scans. Capped at 2 now — an app-wide request "
                                                "limiter (yf_ratelimit) enforces the real ceiling regardless of this "
                                                "setting, so raising it mainly adds retries, not speed.")

        with col3:
            stop_loss_pct = st.number_input("Stop Loss % below Entry Price", min_value=0.1, max_value=5.0, value=0.5, step=0.1, key="long_sl")
            target_pct = st.number_input("Target % above Entry Price", min_value=0.5, max_value=20.0, value=2.0, step=0.5, key="long_tgt")

        with col4:
            strong_score = st.number_input("Strong Signal Score", min_value=60, max_value=90, value=70, step=5, key="long_ss")
            chart_height = st.number_input("Chart Height (px)", min_value=200, max_value=500, value=250, step=50, key="long_ch")

        force_fresh = st.checkbox(
            "🔄 Force fresh data this scan", value=False, key="long_force_fresh",
            help="Off by default now. Turning this on clears the cache and skips the "
                 "45s st.cache_data safety net, forcing a brand-new live fetch for every "
                 "ticker on every scan — on a large scan list this is exactly the pattern "
                 "that triggers Yahoo's rate limiter. Leave off unless you specifically "
                 "need up-to-the-second prices for a small watchlist."
        )

    # Build params dict
    params = {
        'min_volume': min_volume,
        'min_price': min_price,
        'min_conditions': min_conditions,
        'min_score': min_score,
        'price_change_threshold': price_change_threshold,
        'dist_from_low_threshold': dist_from_low_threshold,
        'trend_threshold': trend_threshold,
        'momentum_threshold': momentum_threshold,
        'volume_ratio_threshold': volume_ratio_threshold,
        'rsi_threshold': rsi_threshold,
        'atr_threshold': atr_threshold,
        'rsi_period': rsi_period,
        'atr_period': atr_period,
        'momentum_window': momentum_window,
        'strong_score': strong_score
    }

    screener = IntradayLongScreener(params)

    st.markdown("---")

    if not stock_list:
        st.warning("⚠️ Please select or upload stocks to scan")
        return

    # ── Hard scan-size cap ────────────────────────────────────────────
    # Yahoo's real rate limiter (not just our shim) throttles the shared
    # IPs Streamlit Cloud runs on. Scanning the full NSE list (1500-2000+
    # tickers) in one click means dozens of retried/backed-off chunks,
    # which can block for 10-20+ minutes — long enough that Streamlit
    # Cloud's own health check kills the connection outright. Use the
    # "Scan Range" controls above to work through a big list in batches
    # instead of raising this cap.
    _MAX_SCAN_SIZE = 300
    if scan_count > _MAX_SCAN_SIZE:
        st.error(
            f"⚠️ {scan_count} stocks selected — that's over the {_MAX_SCAN_SIZE}-stock "
            f"safe limit per scan. Yahoo Finance's rate limiter (not just this app's own "
            f"throttling) will very likely block or crash a scan this large on Streamlit "
            f"Cloud's shared IPs. Please narrow the 'Start from index' / 'End at index' "
            f"range above to {_MAX_SCAN_SIZE} stocks or fewer, then scan in batches."
        )
        return

    # ── Session State Init ───────────────────────────────────────────
    if 'long_scan_results' not in st.session_state:
        st.session_state.long_scan_results = None
        st.session_state.long_scan_params = None

    # ── Scan Button ──────────────────────────────────────────────────
    if st.button(f"🔍 SCAN {scan_count} {exchange} STOCKS FOR BUY SIGNALS", type="primary", use_container_width=True):
        scan_list = stock_list[start_index:end_index + 1]
        suffix = '.NS' if exchange == 'NSE' else '.BO'
        full_tickers = tuple(f"{t}{suffix}" for t in scan_list)

        t0 = time.time()
        status_text = st.empty()

        if force_fresh:
            _rl_clear_cache()  # bust yf_ratelimit's 1-hour in-process cache

        status_text.text(f"📡 Bulk-downloading {len(full_tickers)} tickers (intraday 1m)...")
        if force_fresh:
            intraday_batch = _bulk_download_impl(full_tickers, '1d', '1m', chunk_workers=max_workers)
        else:
            intraday_batch = _bulk_download(full_tickers, '1d', '1m', chunk_workers=max_workers)

        status_text.text(f"📡 Bulk-downloading {len(full_tickers)} tickers (daily 5d)...")
        if force_fresh:
            daily_batch = _bulk_download_impl(full_tickers, '5d', '1d', chunk_workers=max_workers)
        else:
            daily_batch = _bulk_download(full_tickers, '5d', '1d', chunk_workers=max_workers)

        status_text.text("⚙️ Scoring stocks...")
        progress_bar = st.progress(0)

        results = []
        total = len(scan_list)

        # ── Diagnostics ────────────────────────────────────────────────
        # "No stocks found" can mean either "genuinely no setups today" or
        # "the data fetch silently failed for everything" — those look
        # identical to the user otherwise. Track which bucket each ticker
        # falls into so we can tell them apart afterward.
        n_no_data = 0            # intraday/daily came back empty for this ticker
        n_filtered_out = 0       # had data, but failed price/volume/condition filters
        n_below_min_score = 0    # passed conditions, but score < Min Score slider

        for idx, ticker in enumerate(scan_list):
            full_ticker = f"{ticker}{suffix}"
            intraday = _extract_ticker_df(intraday_batch, full_ticker)
            daily = _extract_ticker_df(daily_batch, full_ticker)

            if intraday.empty or daily.empty:
                n_no_data += 1
                continue

            result = screener.analyze_stock(ticker, exchange, intraday=intraday, daily=daily)
            if result is None:
                n_filtered_out += 1
            elif result['score'] >= screener.min_score:
                results.append(result)
            else:
                n_below_min_score += 1

            if idx % 5 == 0 or idx == total - 1:
                progress_bar.progress((idx + 1) / total)

        progress_bar.empty()
        elapsed = time.time() - t0
        status_text.text(f"✅ Scanned {total} stocks in {elapsed:.1f}s")
        time.sleep(0.6)
        status_text.empty()

        # Store in session state
        st.session_state.long_scan_results = results
        st.session_state.long_scan_diagnostics = {
            'total': total, 'no_data': n_no_data,
            'filtered_out': n_filtered_out, 'below_min_score': n_below_min_score,
        }
        st.session_state.long_scan_params = params
        st.session_state.long_stop_loss_pct = stop_loss_pct
        st.session_state.long_target_pct = target_pct
        st.session_state.long_chart_height = chart_height

    # ── Display Results ──────────────────────────────────────────────
    if st.session_state.long_scan_results is not None:
        results = st.session_state.long_scan_results

        if not results:
            st.warning("⚠️ No stocks found matching criteria")
            diag = st.session_state.get('long_scan_diagnostics')
            if diag:
                total = diag['total']
                if total and diag['no_data'] == total:
                    st.error(
                        f"🚨 All {total} tickers came back with no price data at all. "
                        f"This is a **data-fetch failure**, not \"no setups today\" — "
                        f"most likely Yahoo rate-limited the bulk download, or "
                        f"'Force fresh data' is off and the cache is stale/empty. "
                        f"Check the Streamlit Cloud logs for `yf_ratelimit` warnings, "
                        f"try a smaller scan (20-30 tickers), or try again in a minute."
                    )
                elif total and diag['no_data'] > total * 0.5:
                    st.warning(
                        f"⚠️ {diag['no_data']} of {total} tickers had no price data "
                        f"(likely rate-limited or delisted) — results may be incomplete. "
                        f"Only {total - diag['no_data']} were actually analyzed."
                    )
                st.caption(
                    f"Diagnostics: {diag['no_data']} no data · "
                    f"{diag['filtered_out']} had data but didn't meet the buy conditions · "
                    f"{diag['below_min_score']} met conditions but scored below your "
                    f"Min Score ({screener.min_score}) — try lowering it if this number is high."
                )
        else:
            results.sort(key=lambda x: x['score'], reverse=True)

            st.success(f"✅ Found {len(results)} potential BUY opportunities!")

            # Summary table
            st.markdown("---")
            st.markdown("#### Screener Results Summary")

            summary_data = []
            for r in results:
                summary_data.append({
                    'Ticker': r['ticker'],
                    'Price': f"₹{r['price']:.2f}",
                    'Change %': f"{r['change_pct']:.2f}%",
                    'Score': r['score'],
                    'Signal': r['signal_strength'],
                    'Volume Ratio': f"{r['volume_ratio']:.2f}x",
                    'Dist from Low': f"{r['dist_from_low']:.2f}%",
                    '5D Trend': f"{r['recent_trend']:.2f}%",
                    'RSI': f"{r['rsi']:.1f}",
                    'ATR %': f"{r['atr_pct']:.2f}%",
                    'Conditions': r['conditions']
                })

            df_summary = pd.DataFrame(summary_data)

            def color_signal(val):
                if 'STRONG' in str(val):
                    return 'background-color: #d4edda'
                elif 'MODERATE' in str(val):
                    return 'background-color: #fff3cd'
                return ''

            def color_change(val):
                try:
                    num = float(str(val).replace('₹', '').replace('%', '').replace('x', ''))
                    if num > 0:
                        return 'background-color: #d4edda'
                    elif num < 0:
                        return 'background-color: #f8d7da'
                except:
                    pass
                return ''

            styled_df = df_summary.style.applymap(color_signal, subset=['Signal']).applymap(color_change, subset=['Change %', '5D Trend'])

            st.dataframe(styled_df, use_container_width=True, height=400)

            # Individual stock detailed analysis
            st.markdown("---")

            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown("#### Detailed Analysis")
            with col2:
                fast_mode = st.checkbox("⚡ Fast mode (skip charts)", value=True, key="long_fast_mode",
                                         help="Skips per-stock chart fetching/rendering — the biggest resource cost. Turn off only when you need visuals.")
            with col3:
                top_n_charts = st.number_input("Charts for top N", min_value=1, max_value=15, value=5, step=1,
                                                disabled=fast_mode, key="long_top_n")

            chart_timeframe = st.selectbox(
                "Chart Timeframe",
                ["1 Day", "1 Week", "1 Month", "3 Months", "6 Months", "1 Year", "3 Years", "All Time"],
                index=0,
                key="long_chart_tf",
                disabled=fast_mode
            )

            timeframe_map = {
                "1 Day": ("1d", "1m"),
                "1 Week": ("5d", "15m"),
                "1 Month": ("1mo", "1h"),
                "3 Months": ("3mo", "1d"),
                "6 Months": ("6mo", "1d"),
                "1 Year": ("1y", "1d"),
                "3 Years": ("3y", "1wk"),
                "All Time": ("max", "1wk")
            }

            period, interval = timeframe_map[chart_timeframe]

            # Retrieve trading settings from session state
            _sl_pct = st.session_state.get('long_stop_loss_pct', stop_loss_pct)
            _tgt_pct = st.session_state.get('long_target_pct', target_pct)
            _ch = st.session_state.get('long_chart_height', chart_height)

            for i, result in enumerate(results):
                st.markdown(f"##### {i + 1}. {result['ticker']} - {result['signal_strength']} (Score: {result['score']})")

                col1, col2, col3, col4, col5, col6 = st.columns(6)

                with col1:
                    st.metric("Price", f"₹{result['price']:.2f}", f"{result['change_pct']:.2f}%")
                with col2:
                    st.metric("Low", f"₹{result['low']:.2f}")
                with col3:
                    st.metric("Dist from Low", f"{result['dist_from_low']:.2f}%")
                with col4:
                    st.metric("Vol Ratio", f"{result['volume_ratio']:.2f}x")
                with col5:
                    st.metric("RSI", f"{result['rsi']:.1f}")
                with col6:
                    st.metric("5D Trend", f"{result['recent_trend']:.2f}%")

                # Charts: only fetched/rendered for the top N results, and only
                # when Fast Mode is off. This is the single biggest lever for
                # cutting yfinance calls + browser render load on Streamlit Cloud.
                if not fast_mode and i < top_n_charts:
                    try:
                        chart_data = _fetch_chart_data(result['full_ticker'], period, interval)

                        if not chart_data.empty:
                            rsi_calc_period = min(rsi_period, max(len(chart_data) // 2, 1))
                            rsi_values, rsi_index = [], []
                            if len(chart_data) > rsi_calc_period:
                                closes = chart_data['Close']
                                delta = closes.diff()
                                gain = (delta.where(delta > 0, 0)).rolling(rsi_calc_period).mean()
                                loss = (-delta.where(delta < 0, 0)).rolling(rsi_calc_period).mean()
                                rs = gain / loss
                                rsi_series = (100 - (100 / (1 + rs))).iloc[rsi_calc_period:]
                                rsi_values = rsi_series.values
                                rsi_index = rsi_series.index

                            # One combined 3-row subplot instead of 3 separate
                            # plotly figures — far cheaper to render.
                            fig = make_subplots(rows=1, cols=3, subplot_titles=(
                                f"Price ({chart_timeframe})", f"Volume ({chart_timeframe})", f"RSI ({chart_timeframe})"))
                            fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['Close'], mode='lines',
                                                      name='Price', line=dict(color='#28a745', width=2)), row=1, col=1)
                            if 'open' in result:
                                fig.add_hline(y=result['open'], line_dash="dash", line_color="gray", line_width=1, row=1, col=1)
                            fig.add_trace(go.Bar(x=chart_data.index, y=chart_data['Volume'], name='Volume',
                                                  marker_color='#17a2b8'), row=1, col=2)
                            if len(rsi_values):
                                fig.add_trace(go.Scatter(x=rsi_index, y=rsi_values, mode='lines', name='RSI',
                                                          line=dict(color='#007bff', width=2)), row=1, col=3)
                                fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, row=1, col=3)
                                fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=1, col=3)
                            fig.update_layout(height=_ch, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"No chart data available for {result['ticker']}")
                    except Exception as e:
                        st.error(f"Error loading charts for {result['ticker']}: {str(e)}")
                elif not fast_mode and i == top_n_charts:
                    st.caption(f"Charts hidden beyond top {top_n_charts} — increase 'Charts for top N' above to see more.")

                # ── Trading Levels ───────────────────────────────────────
                # BUY: stop loss is BELOW entry, target is ABOVE entry
                stop_loss = result['price'] * (1 - _sl_pct / 100)
                target = result['price'] * (1 + _tgt_pct / 100)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.info(f"💡 Entry: ₹{result['price']:.2f}")
                with col2:
                    st.error(f"🛑 Stop: ₹{stop_loss:.2f}")
                with col3:
                    st.success(f"🎯 Target: ₹{target:.2f}")
                with col4:
                    risk = abs(result['price'] - stop_loss)
                    reward = abs(target - result['price'])
                    risk_reward = reward / risk if risk > 0 else 0
                    st.metric("R:R Ratio", f"1:{risk_reward:.2f}")

                st.caption(f"**Conditions:** {result['conditions']}")
                st.markdown("---")

    # ── Help Section ─────────────────────────────────────────────────
    with st.expander("📚 How to Use"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Stock Selection:**
            - Default: Pre-loaded major stocks
            - Upload File: Text file with one ticker per line
            - Custom: Enter tickers manually
            - Range: Scan from index X to Y

            **Best Scan Times:**
            - 10:00-11:30 AM (Post opening)
            - 1:30-2:30 PM (Post lunch dip recovery)
            """)
        with col2:
            st.markdown("""
            **Signal Strength:**
            - 🟢 STRONG (70+): High probability
            - 🟡 MODERATE (50-69): Good with confirmation

            **Risk Management:**
            - Stop loss: 0.5-1% below entry price
            - Position size: 1-2% of capital
            - Exit before 3:15 PM
            
            **Buy Conditions Checked:**
            - Price rising from open
            - Near day low (bounce zone)
            - 5-day uptrend
            - Positive momentum
            - High volume
            - RSI oversold (< threshold)
            - Good ATR volatility
            """)

    st.markdown("---")
    st.caption("⚠️ **Disclaimer:** Intraday trading is risky. For educational purposes only. Consult a financial advisor.")


if __name__ == "__main__":
    st.set_page_config(page_title="Long Buy Screener", layout="wide")
    show_intraday_long_screener()
