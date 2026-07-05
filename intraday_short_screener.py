"""
INTRADAY SHORT SELLING SCREENER - Enhanced UI with Full Control
================================================================
Scan for stocks showing downward momentum for intraday shorting
Complete customization: upload files, custom tickers, range selection
"""

import streamlit as st
# ── yf_ratelimit shim ──────────────────────────────────────────
# Replaces direct yfinance calls with rate-limit-safe wrappers.
# DO NOT remove this block.
from yf_ratelimit import safe_ticker as _rl_ticker, safe_download as _rl_download

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
# reruns/scans. Cuts repeated yfinance calls drastically, which is the
# main cause of Streamlit Cloud rate-limit / resource burn.

@st.cache_data(ttl=90, show_spinner=False)
def _fetch_core_data(full_ticker: str):
    """Fetch intraday (1d/1m) + daily (5d/1d) history for one ticker, cached."""
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


class IntradayShortScreener:
    """Scan for intraday short-selling opportunities"""
    
    def __init__(self, params):
        """Initialize with all adjustable parameters"""
        self.min_volume = params.get('min_volume', 100000)
        self.min_price = params.get('min_price', 20)
        self.min_conditions = params.get('min_conditions', 4)
        self.min_score = params.get('min_score', 50)
        
        # Advanced thresholds
        self.price_change_threshold = params.get('price_change_threshold', 0.0)
        self.dist_from_high_threshold = params.get('dist_from_high_threshold', 2.0)
        self.trend_threshold = params.get('trend_threshold', -2.0)
        self.momentum_threshold = params.get('momentum_threshold', -0.5)
        self.volume_ratio_threshold = params.get('volume_ratio_threshold', 1.2)
        self.rsi_threshold = params.get('rsi_threshold', 65)
        self.atr_threshold = params.get('atr_threshold', 1.0)
        
        # Technical indicator settings
        self.rsi_period = params.get('rsi_period', 14)
        self.atr_period = params.get('atr_period', 14)
        self.momentum_window = params.get('momentum_window', 30)
        self.strong_score = params.get('strong_score', 70)
    
    def get_stock_list_from_file(self, uploaded_file):
        """Get stock list from uploaded file"""
        try:
            content = uploaded_file.read().decode('utf-8')
            tickers = [line.strip() for line in content.split('\n') if line.strip()]
            return tickers
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return []
    
    def get_default_stock_list(self, exchange='NSE'):
        """Get default stock list from file"""
        try:
            if exchange == 'NSE':
                file_path = 'nse.txt'
            else:
                file_path = 'bse.txt'
            
            with open(file_path, 'r') as f:
                tickers = [line.strip() for line in f if line.strip()]
            
            return tickers
        except Exception as e:
            st.error(f"Error reading {file_path}: {str(e)}")
            # Fallback to minimal list if file not found
            return ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']
    
    def analyze_stock(self, ticker, exchange='NSE'):
        """Analyze individual stock for short-selling setup"""
        try:
            suffix = '.NS' if exchange == 'NSE' else '.BO'
            full_ticker = f"{ticker}{suffix}"
            
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
            
            price_change_pct = ((current_price - open_price) / open_price) * 100
            dist_from_high = ((high_price - current_price) / high_price) * 100
            
            if len(daily) >= 2:
                recent_change = ((daily['Close'].iloc[-1] - daily['Close'].iloc[0]) / daily['Close'].iloc[0]) * 100
            else:
                recent_change = 0
            
            if len(intraday) >= self.momentum_window * 2:
                last_n = intraday['Close'].iloc[-self.momentum_window:].mean()
                prev_n = intraday['Close'].iloc[-self.momentum_window*2:-self.momentum_window].mean()
                momentum_change = ((last_n - prev_n) / prev_n) * 100
            else:
                momentum_change = 0
            
            avg_volume_5d = daily['Volume'].mean()
            volume_ratio = volume / avg_volume_5d if avg_volume_5d > 0 else 0
            
            rsi = self.calculate_rsi(intraday['Close'], period=self.rsi_period)
            atr = self.calculate_atr(intraday, period=self.atr_period)
            atr_pct = (atr / current_price) * 100
            
            # Check conditions using adjustable thresholds
            conditions_met = []
            
            if price_change_pct < self.price_change_threshold:
                conditions_met.append("Down from open")
            elif price_change_pct < 0.5:
                conditions_met.append("Flat/weak")
            
            if dist_from_high < self.dist_from_high_threshold:
                conditions_met.append("Near day high")
            
            if recent_change < self.trend_threshold:
                conditions_met.append("5-day downtrend")
            
            if momentum_change < self.momentum_threshold:
                conditions_met.append("Negative momentum")
            
            if volume_ratio > self.volume_ratio_threshold:
                conditions_met.append("High volume")
            
            if rsi and rsi > self.rsi_threshold:
                conditions_met.append("RSI overbought")
            
            if atr_pct > self.atr_threshold:
                conditions_met.append("Good volatility")
            
            if len(conditions_met) < self.min_conditions:
                return None
            
            # Calculate score
            score = 0
            
            if price_change_pct < -2:
                score += 30
            elif price_change_pct < -1:
                score += 20
            elif price_change_pct < 0:
                score += 10
            
            if dist_from_high < 1:
                score += 20
            elif dist_from_high < 2:
                score += 10
            
            if recent_change < -5:
                score += 20
            elif recent_change < -2:
                score += 10
            
            if momentum_change < -1:
                score += 15
            elif momentum_change < -0.5:
                score += 8
            
            if volume_ratio > 1.5:
                score += 10
            elif volume_ratio > 1.2:
                score += 5
            
            if rsi and rsi > 70:
                score += 5
            elif rsi and rsi > 65:
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
                'dist_from_high': dist_from_high,
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


def show_intraday_short_screener():
    """Display the intraday short-selling screener"""
    
    # Custom CSS for better font sizing
    st.markdown("""
        <style>
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        h1 {
            font-size: 1.8rem !important;
            margin-bottom: 0.5rem !important;
        }
        h2 {
            font-size: 1.3rem !important;
            margin-top: 1rem !important;
            margin-bottom: 0.5rem !important;
        }
        h3 {
            font-size: 1.1rem !important;
        }
        .stMetric label {
            font-size: 0.85rem !important;
        }
        .stMetric .metric-value {
            font-size: 1.2rem !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("📉 Intraday Short Selling Screener")
    
    # Stock Selection Section
    st.markdown("#### 📋 Stock Selection")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        exchange = st.selectbox("Exchange", ["NSE", "BSE"])
    
    with col2:
        stock_source = st.radio(
            "Stock Source",
            ["Default List", "Upload File", "Custom Tickers"],
            horizontal=True
        )
    
    stock_list = []
    
    if stock_source == "Upload File":
        uploaded_file = st.file_uploader(
            f"Upload {exchange} ticker file (one ticker per line)",
            type=['txt'],
            help="Upload a text file with one ticker symbol per line"
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
            height=100
        )
        
        if custom_input:
            # Support both comma and newline separation
            if ',' in custom_input:
                stock_list = [t.strip().upper() for t in custom_input.split(',') if t.strip()]
            else:
                stock_list = [t.strip().upper() for t in custom_input.split('\n') if t.strip()]
            st.success(f"✅ {len(stock_list)} tickers entered")
        else:
            st.info("👆 Enter ticker symbols")
            
    else:  # Default List
        screener_temp = IntradayShortScreener({})
        stock_list = screener_temp.get_default_stock_list(exchange)
        st.info(f"📊 Using default list with {len(stock_list)} stocks")
    
    # Range Selection
    if stock_list:
        st.markdown("#### 🎯 Scan Range")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            start_index = st.number_input(
                "Start from index",
                min_value=0,
                max_value=len(stock_list)-1,
                value=0,
                help=f"First ticker: {stock_list[0] if stock_list else 'N/A'}"
            )
        
        with col2:
            end_index = st.number_input(
                "End at index",
                min_value=start_index,
                max_value=len(stock_list)-1,
                value=min(start_index + 29, len(stock_list)-1),
                help=f"Last ticker in range"
            )
        
        with col3:
            scan_count = end_index - start_index + 1
            st.metric("Stocks to Scan", scan_count)
        
        # Show range preview
        if scan_count > 0:
            preview_list = stock_list[start_index:end_index+1]
            st.caption(f"**Scan Range:** {preview_list[0]} to {preview_list[-1]} ({scan_count} stocks)")
            with st.expander("Preview ticker list"):
                st.write(", ".join(preview_list[:50]))
                if len(preview_list) > 50:
                    st.caption(f"... and {len(preview_list) - 50} more")
    
    st.markdown("---")
    
    # Parameters Section
    st.markdown("#### ⚙️ Screening Parameters")
    
    with st.expander("Basic Filters", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            min_price = st.number_input("Min Price (₹)", min_value=1, max_value=500, value=20, step=5)
        
        with col2:
            min_volume = st.number_input("Min Volume", min_value=10000, max_value=10000000, value=100000, step=10000)
        
        with col3:
            min_conditions = st.slider("Min Conditions (out of 7)", min_value=2, max_value=7, value=4)
        
        with col4:
            min_score = st.slider("Min Score (0-100)", min_value=20, max_value=90, value=50, step=5)
    
    with st.expander("Advanced Thresholds"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            price_change_threshold = st.slider("Price Change (%)", min_value=-5.0, max_value=1.0, value=0.0, step=0.5)
            momentum_threshold = st.slider("Momentum (%)", min_value=-5.0, max_value=0.0, value=-0.5, step=0.1)
        
        with col2:
            dist_from_high_threshold = st.slider("Dist from High (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
            volume_ratio_threshold = st.slider("Volume Ratio", min_value=1.0, max_value=3.0, value=1.2, step=0.1)
        
        with col3:
            trend_threshold = st.slider("5-Day Trend (%)", min_value=-10.0, max_value=0.0, value=-2.0, step=0.5)
            rsi_threshold = st.slider("RSI Overbought", min_value=50, max_value=80, value=65, step=5)
        
        with col4:
            atr_threshold = st.slider("ATR % Threshold", min_value=0.5, max_value=5.0, value=1.0, step=0.1)
    
    with st.expander("Technical Indicators & Trading Settings"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            rsi_period = st.number_input("RSI Period", min_value=5, max_value=50, value=14, step=1)
            atr_period = st.number_input("ATR Period", min_value=5, max_value=50, value=14, step=1)
        
        with col2:
            momentum_window = st.number_input("Momentum Window (min)", min_value=10, max_value=120, value=30, step=5)
            max_workers = st.number_input("Parallel Workers", min_value=1, max_value=8, value=4, step=1,
                                           help="Kept low to avoid yfinance rate-limits / Streamlit Cloud resource exhaustion")
        
        with col3:
            stop_loss_pct = st.number_input("Stop Loss % above Entry Price", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
            target_pct = st.number_input("Target % below Entry Price", min_value=0.5, max_value=20.0, value=2.0, step=0.5)
        
        with col4:
            strong_score = st.number_input("Strong Signal Score", min_value=60, max_value=90, value=70, step=5)
            chart_height = st.number_input("Chart Height (px)", min_value=200, max_value=500, value=250, step=50)
    
    # Initialize screener with all parameters
    params = {
        'min_volume': min_volume,
        'min_price': min_price,
        'min_conditions': min_conditions,
        'min_score': min_score,
        'price_change_threshold': price_change_threshold,
        'dist_from_high_threshold': dist_from_high_threshold,
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
    
    screener = IntradayShortScreener(params)
    
    st.markdown("---")
    
    # Scan button
    if not stock_list:
        st.warning("⚠️ Please select or upload stocks to scan")
        return
    
    # Store results in session state to persist across reruns
    if 'scan_results' not in st.session_state:
        st.session_state.scan_results = None
        st.session_state.scan_params = None
    
    if st.button(f"🔍 SCAN {scan_count} {exchange} STOCKS", type="primary", use_container_width=True):
        # Get the range to scan
        scan_list = stock_list[start_index:end_index+1]
        
        st.info(f"📊 Scanning {len(scan_list)} stocks from index {start_index} to {end_index}...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(screener.analyze_stock, ticker, exchange): ticker for ticker in scan_list}
            
            completed = 0
            total = len(scan_list)
            
            for future in as_completed(future_to_ticker):
                completed += 1
                progress_bar.progress(completed / total)
                status_text.text(f"Scanning... {completed}/{total}")
                
                result = future.result()
                if result and result['score'] >= screener.min_score:
                    results.append(result)
        
        progress_bar.empty()
        status_text.empty()
        
        # Store results in session state
        st.session_state.scan_results = results
        st.session_state.scan_params = params
        st.session_state.stop_loss_pct = stop_loss_pct
        st.session_state.target_pct = target_pct
        st.session_state.chart_height = chart_height
    
    # Display results from session state
    if st.session_state.scan_results is not None:
        results = st.session_state.scan_results
        
        if not results:
            st.warning("⚠️ No stocks found matching criteria")
        else:
            results.sort(key=lambda x: x['score'], reverse=True)
            
            st.success(f"✅ Found {len(results)} potential opportunities!")
            
            # Summary Table at Top
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
                    'Dist from High': f"{r['dist_from_high']:.2f}%",
                    '5D Trend': f"{r['recent_trend']:.2f}%",
                    'RSI': f"{r['rsi']:.1f}",
                    'ATR %': f"{r['atr_pct']:.2f}%",
                    'Conditions': r['conditions']
                })
            
            df_summary = pd.DataFrame(summary_data)
            
            def color_score(val):
                if 'STRONG' in str(val):
                    return 'background-color: #ffcccc'
                elif 'MODERATE' in str(val):
                    return 'background-color: #fff3cd'
                return ''
            
            def color_change(val):
                try:
                    num = float(val.replace('₹', '').replace('%', '').replace('x', ''))
                    if num < 0:
                        return 'background-color: #ffcccc'
                    elif num > 0:
                        return 'background-color: #d4edda'
                except:
                    pass
                return ''
            
            styled_df = df_summary.style.applymap(color_score, subset=['Signal']).applymap(color_change, subset=['Change %', '5D Trend'])
            
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # Individual stock analysis with charts
            st.markdown("---")
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown("#### Detailed Analysis")
            with col2:
                fast_mode = st.checkbox("⚡ Fast mode (skip charts)", value=True, key="short_fast_mode",
                                         help="Skips per-stock chart fetching/rendering — the biggest resource cost. Turn off only when you need visuals.")
            with col3:
                top_n_charts = st.number_input("Charts for top N", min_value=1, max_value=15, value=5, step=1,
                                                disabled=fast_mode, key="short_top_n")

            chart_timeframe = st.selectbox(
                "Chart Timeframe",
                ["1 Day", "1 Week", "1 Month", "3 Months", "6 Months", "1 Year", "3 Years", "All Time"],
                index=0,
                key="chart_timeframe",
                disabled=fast_mode
            )
            
            # Map timeframe to yfinance parameters
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
            
            for i, result in enumerate(results):
                st.markdown(f"##### {i+1}. {result['ticker']} - {result['signal_strength']} (Score: {result['score']})")
                
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                
                with col1:
                    st.metric("Price", f"₹{result['price']:.2f}", f"{result['change_pct']:.2f}%")
                with col2:
                    st.metric("High", f"₹{result['high']:.2f}")
                with col3:
                    st.metric("Dist from High", f"{result['dist_from_high']:.2f}%")
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
                            # One combined 3-row subplot instead of 3 separate
                            # plotly figures — far cheaper to render.
                            rsi_calc_period = min(rsi_period, max(len(chart_data) // 2, 1))
                            rsi_values, rsi_index = [], []
                            if len(chart_data) > rsi_calc_period:
                                closes = chart_data['Close']
                                delta = closes.diff()
                                gain = (delta.where(delta > 0, 0)).rolling(rsi_calc_period).mean()
                                loss = (-delta.where(delta < 0, 0)).rolling(rsi_calc_period).mean()
                                rs = gain / loss
                                rsi_series = 100 - (100 / (1 + rs))
                                rsi_series = rsi_series.iloc[rsi_calc_period:]
                                rsi_values = rsi_series.values
                                rsi_index = rsi_series.index

                            fig = make_subplots(rows=1, cols=3, subplot_titles=(
                                f"Price ({chart_timeframe})", f"Volume ({chart_timeframe})", f"RSI ({chart_timeframe})"))
                            fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['Close'], mode='lines',
                                                      name='Price', line=dict(color='#dc3545', width=2)), row=1, col=1)
                            if 'open' in result:
                                fig.add_hline(y=result['open'], line_dash="dash", line_color="gray", line_width=1, row=1, col=1)
                            fig.add_trace(go.Bar(x=chart_data.index, y=chart_data['Volume'], name='Volume',
                                                  marker_color='#17a2b8'), row=1, col=2)
                            if len(rsi_values):
                                fig.add_trace(go.Scatter(x=rsi_index, y=rsi_values, mode='lines', name='RSI',
                                                          line=dict(color='#28a745', width=2)), row=1, col=3)
                                fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, row=1, col=3)
                                fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=1, col=3)
                            fig.update_layout(height=chart_height, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"No chart data available for {result['ticker']}")
                    except Exception as e:
                        st.error(f"Error loading charts for {result['ticker']}: {str(e)}")
                elif not fast_mode and i == top_n_charts:
                    st.caption(f"Charts hidden beyond top {top_n_charts} — increase 'Charts for top N' above to see more.")
                
                # Trading levels (use session state to preserve values after scan)
                _sl_pct = st.session_state.get('stop_loss_pct', stop_loss_pct)
                _tgt_pct = st.session_state.get('target_pct', target_pct)
                _ch = st.session_state.get('chart_height', chart_height)
                stop_loss = result['price'] * (1 + _sl_pct/100)
                target = result['price'] * (1 - _tgt_pct/100)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.info(f"💡 Entry: ₹{result['price']:.2f}")
                with col2:
                    st.error(f"🛑 Stop: ₹{stop_loss:.2f}")
                with col3:
                    st.success(f"🎯 Target: ₹{target:.2f}")
                with col4:
                    risk_reward = abs((result['price'] - target) / (stop_loss - result['price']))
                    st.metric("R:R Ratio", f"1:{risk_reward:.2f}")
                
                st.caption(f"**Conditions:** {result['conditions']}")
                st.markdown("---")
    
    # Help section
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
            - 1:30-2:30 PM (Post lunch)
            """)
        with col2:
            st.markdown("""
            **Signal Strength:**
            - 🔴 STRONG (70+): High probability
            - 🟡 MODERATE (50-69): Good with confirmation
            
            **Risk Management:**
            - Stop loss: 0.5-1% above day high
            - Position size: 1-2% of capital
            - Exit before 3:15 PM
            """)
    
    st.markdown("---")
    st.caption("⚠️ **Disclaimer:** Short selling is risky. For educational purposes only. Consult a financial advisor.")


if __name__ == "__main__":
    st.set_page_config(page_title="Short Screener", layout="wide")
    show_intraday_short_screener()
