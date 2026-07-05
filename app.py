"""
INTRADAY SCREENER SUITE — Parent App / Landing Page
=====================================================
Single entry point that hosts both the Long (Buy) and Short (Sell)
intraday screeners behind one navigation, with ONE st.set_page_config
call and lazy imports so the app stays light on Streamlit Cloud.
"""

import streamlit as st

st.set_page_config(
    page_title="Intraday Screener Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global light-weight CSS (kept minimal on purpose) ───────────────
st.markdown("""
    <style>
    .main .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
    h1 { font-size: 1.9rem !important; }
    </style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Screener Suite")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📈 Long (Buy) Screener", "📉 Short (Sell) Screener"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption(
        "⚡ **Tip:** Keep 'Fast mode' on and Parallel Workers low (3–5) "
        "to avoid yfinance rate-limits and Streamlit Cloud resource caps."
    )
    st.caption("⚠️ Educational use only. Not investment advice.")

# ── Lazy-loaded pages ─────────────────────────────────────────────────
# Only the module the user actually asks for gets imported/executed —
# keeps memory and startup cost down instead of always loading both.

if page == "🏠 Home":
    st.title("📊 Intraday Screener Suite")
    st.markdown(
        "A single hub for two intraday NSE/BSE screeners. Pick a screener "
        "from the sidebar to get started."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📈 Long (Buy) Screener")
        st.markdown(
            "- Finds stocks with **upward** intraday momentum\n"
            "- Bounce / breakout setups near the day low\n"
            "- Best used post-open (10:00–11:30) and post-lunch (1:30–2:30)"
        )
    with col2:
        st.markdown("### 📉 Short (Sell) Screener")
        st.markdown(
            "- Finds stocks with **downward** intraday momentum\n"
            "- Weakness near the day high, overbought RSI fading\n"
            "- Exit all positions before 3:15 PM"
        )

    st.markdown("---")
    st.markdown("### ⚙️ Performance notes")
    st.markdown(
        "- Both screeners default to **Fast Mode** (no per-stock charts) so a scan "
        "returns a clean, results-first table with score, signal, and key metrics.\n"
        "- Charts are opt-in and capped to your **top N** results to avoid extra "
        "yfinance calls and heavy Plotly rendering.\n"
        "- Stock history is cached for 60–120 seconds, so re-running a scan or "
        "switching timeframes doesn't always re-hit yfinance.\n"
        "- Parallel workers are capped at 8 (default 4) to avoid bursts that trigger "
        "yfinance rate limiting on Streamlit Cloud's shared IPs."
    )

elif page == "📈 Long (Buy) Screener":
    from intraday_long_screener import show_intraday_long_screener
    show_intraday_long_screener()

else:
    from intraday_short_screener import show_intraday_short_screener
    show_intraday_short_screener()
