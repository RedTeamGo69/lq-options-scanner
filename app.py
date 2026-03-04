import math
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime
import pytz
import streamlit as st
import urllib.request
import json

# --- STREAMLIT PAGE CONFIG ---
st.set_page_config(page_title="LQ Quant Options Scanner", page_icon="📈", layout="centered")

# --- CORE MATH CLASS (Merton Dividend-Adjusted Black-Scholes) ---
class BlackScholesCalculator:
    def __init__(self, S, K, T, r, sigma, q=0.0):
        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = max(float(sigma), 0.0001)
        self.q = float(q)

    def _get_d1_d2(self):
        if self.T <= 0: return 0.0, 0.0
        d1 = (math.log(self.S / self.K) + (self.r - self.q + (self.sigma ** 2) / 2) * self.T) / (self.sigma * math.sqrt(self.T))
        d2 = d1 - self.sigma * math.sqrt(self.T)
        return d1, d2

    def get_call_data(self):
        if self.T <= 0: return (max(0.0, self.S - self.K), 1.0 if self.S > self.K else 0.0)
        d1, d2 = self._get_d1_d2()
        price = (self.S * math.exp(-self.q * self.T) * norm.cdf(d1)) - (self.K * math.exp(-self.r * self.T) * norm.cdf(d2))
        return round(price, 2), round(math.exp(-self.q * self.T) * norm.cdf(d1), 2)

    def get_put_data(self):
        if self.T <= 0: return (max(0.0, self.K - self.S), -1.0 if self.S < self.K else 0.0)
        d1, d2 = self._get_d1_d2()
        price = (self.K * math.exp(-self.r * self.T) * norm.cdf(-d2)) - (self.S * math.exp(-self.q * self.T) * norm.cdf(-d1))
        return round(price, 2), round(math.exp(-self.q * self.T) * (norm.cdf(d1) - 1.0), 2)

    def get_greeks(self, opt_type='C'):
        if self.T <= 0:
            return {'theta': 0.0, 'gamma': 0.0, 'vega': 0.0}
        d1, d2 = self._get_d1_d2()
        sqrt_T = math.sqrt(self.T)
        pdf_d1 = norm.pdf(d1)
        eq = math.exp(-self.q * self.T)

        # Gamma (same for calls and puts)
        gamma = (eq * pdf_d1) / (self.S * self.sigma * sqrt_T)

        # Vega per 1% move in IV
        vega = (self.S * eq * pdf_d1 * sqrt_T) / 100.0

        # Theta per calendar day
        common = -(self.S * eq * pdf_d1 * self.sigma) / (2 * sqrt_T)
        if opt_type == 'C':
            theta = (common
                     + self.q * self.S * eq * norm.cdf(d1)
                     - self.r * self.K * math.exp(-self.r * self.T) * norm.cdf(d2)) / 365.0
        else:
            theta = (common
                     - self.q * self.S * eq * norm.cdf(-d1)
                     + self.r * self.K * math.exp(-self.r * self.T) * norm.cdf(-d2)) / 365.0

        return {
            'theta': round(theta, 4),
            'gamma': round(gamma, 4),
            'vega': round(vega, 4)
        }

# --- DATA FUNCTIONS ---
@st.cache_data(ttl=86400)
def get_company_name(ticker_symbol):
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={ticker_symbol}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=3) as response:
            data = json.loads(response.read().decode())
            quotes = data.get('quotes', [])
            if quotes:
                name = quotes[0].get('longname') or quotes[0].get('shortname')
                if name:
                    return name
    except Exception: pass
    return ticker_symbol

@st.cache_data(ttl=86400)
def get_event_metrics(ticker_symbol):
    earnings_date = "N/A"
    ex_div_date = "N/A"
    try:
        stock = yf.Ticker(ticker_symbol)
        try:
            cal = stock.calendar
            if isinstance(cal, dict):
                if 'Earnings Date' in cal and cal['Earnings Date']:
                    earnings_date = cal['Earnings Date'][0].strftime('%Y-%m-%d')
                if 'Ex-Dividend Date' in cal and cal['Ex-Dividend Date']:
                    ex_div_date = cal['Ex-Dividend Date'].strftime('%Y-%m-%d')
            elif isinstance(cal, pd.DataFrame):
                if 'Earnings Date' in cal.index:
                    earnings_date = cal.loc['Earnings Date'].iloc[0].strftime('%Y-%m-%d')
                if 'Ex-Dividend Date' in cal.index:
                    ex_div_date = cal.loc['Ex-Dividend Date'].iloc[0].strftime('%Y-%m-%d')
        except Exception: pass
    except Exception: pass
    return earnings_date, ex_div_date

@st.cache_data(ttl=60)
def get_live_data(ticker_symbol, lookback_days=90):
    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period=f"{lookback_days}d")
        if hist.empty: return "EMPTY", None, None, None

        # Fast path first for price, slow path fallback
        try:
            current_price = stock.fast_info.last_price
            if current_price is None:
                current_price = hist['Close'].iloc[-1]
        except Exception:
            current_price = hist['Close'].iloc[-1]

        # Dividend yield for Merton model (requires full info call)
        # Yahoo sometimes returns yield as a whole number (e.g. 4.01 meaning 4.01%)
        # instead of decimal form (0.0401). Detect and convert.
        try:
            div_yield = stock.info.get('dividendYield', 0.0) or 0.0
            if div_yield > 1.0:
                div_yield = div_yield / 100.0
        except Exception:
            div_yield = 0.0

        hist['Log_Return'] = np.log(hist['Close'] / hist['Close'].shift(1))
        sigma = hist['Log_Return'].std() * np.sqrt(252)

        options_dates = stock.options

        return current_price, sigma, options_dates, div_yield

    except Exception as e:
        if "RateLimit" in str(type(e).__name__) or "429" in str(e):
            return "RATE_LIMIT", None, None, None
        return "ERROR", None, None, None

@st.cache_data(ttl=86400)
def get_iv_rank_data(ticker_symbol):
    """Get 1-year rolling HV range for IV rank calculation."""
    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period="1y")
        if len(hist) < 60: return None, None
        hist['Log_Return'] = np.log(hist['Close'] / hist['Close'].shift(1))
        hist['Rolling_HV'] = hist['Log_Return'].rolling(window=30).std() * np.sqrt(252)
        hv_series = hist['Rolling_HV'].dropna()
        if hv_series.empty: return None, None
        return float(hv_series.min()), float(hv_series.max())
    except Exception:
        return None, None

@st.cache_data(ttl=3600)
def get_risk_free_rate():
    try:
        irx = yf.Ticker("^IRX")
        hist = irx.history(period="5d")
        if not hist.empty: return hist['Close'].iloc[-1] / 100.0
    except Exception: pass
    return 0.045

def get_raw_moneyness(S, K, opt_type, closest_strike):
    if K == closest_strike: return "ATM"
    if opt_type == 'P': return "ITM" if K > S else "OTM"
    else: return "ITM" if K < S else "OTM"

def style_dataframe(df, hv):
    def highlight_moneyness(val):
        if val == 'OTM': return 'color: #00FF00; font-weight: bold'
        elif val == 'ITM': return 'color: #FF4B4B; font-weight: bold'
        else: return 'color: #FFD700; font-weight: bold'

    def highlight_edge(val):
        if val > 0: return 'color: #00FF00'
        elif val < 0: return 'color: #FF4B4B'
        return ''

    def highlight_iv(val):
        if hv == 0: return ''
        ratio = (val / 100.0) / hv
        if ratio >= 1.5: return 'color: #FF4B4B; font-weight: bold'
        elif ratio <= 0.8: return 'color: #00FFFF; font-weight: bold'
        return ''

    def highlight_spread(val):
        if val > 20: return 'color: #FF4B4B; font-weight: bold'
        elif val > 10: return 'color: #FFD700'
        return 'color: #00FF00'

    subset_map = {
        'Moneyness': highlight_moneyness,
        'Edge (%)': highlight_edge,
        'IV (%)': highlight_iv,
        'Spread (%)': highlight_spread,
    }

    styled_df = df.style
    for col, func in subset_map.items():
        if col in df.columns:
            styled_df = styled_df.map(func, subset=[col])

    styled_df = styled_df.format(precision=2)
    return styled_df

def format_date_dropdown(date_str):
    ny_tz = pytz.timezone('America/New_York')
    today_ny = datetime.now(ny_tz).date()
    days_to_exp = (datetime.strptime(date_str, "%Y-%m-%d").date() - today_ny).days
    return f"{date_str} ({days_to_exp} DTE)"

# --- PER-TICKER PROCESSING ---
def process_ticker(ticker, action, opt_type, key_suffix=""):
    """Full dashboard for a single ticker. key_suffix ensures unique widget keys for multi-ticker."""
    with st.spinner(f"Pulling live data for {ticker}..."):
        S, sigma, options_dates, div_yield = get_live_data(ticker)

    if S == "RATE_LIMIT":
        st.warning("Yahoo Rate Limit Hit: Please wait 60 seconds before fetching data again.")
        return
    elif S == "EMPTY" or S == "ERROR":
        st.error(f"Could not find valid data for {ticker}. Please check the symbol.")
        return
    elif not options_dates:
        st.error(f"No options available for {ticker}.")
        return

    r = get_risk_free_rate()
    earnings_date, ex_div_date = get_event_metrics(ticker)

    company_name = get_company_name(ticker)
    if company_name.upper() == ticker.upper():
        st.subheader(f"{ticker}")
    else:
        st.subheader(f"{company_name} ({ticker})")

    st.divider()

    # --- Metrics Row 1: Price & Volatility ---
    m1, m2, m3 = st.columns(3)
    m1.metric("Live Price", f"${S:.2f}")
    m2.metric("Hist. Volatility", f"{sigma*100:.1f}%")
    m3.metric("Div. Yield", f"{div_yield*100:.1f}%" if div_yield > 0 else "N/A")

    # --- Metrics Row 2: Events & Rate ---
    m4, m5, m6 = st.columns(3)
    m4.metric("Next Earnings", earnings_date)
    m5.metric("Ex-Div Date", ex_div_date)
    m6.metric("Risk-Free", f"{r*100:.1f}%")

    # --- IV Rank ---
    hv_low, hv_high = get_iv_rank_data(ticker)

    target_date = st.selectbox(
        "Select Expiration Date:",
        options_dates,
        format_func=format_date_dropdown,
        key=f"exp_{key_suffix}"
    )

    # Lock DTE to New York Time
    ny_tz = pytz.timezone('America/New_York')
    today_ny = datetime.now(ny_tz).date()
    dte = (datetime.strptime(target_date, "%Y-%m-%d").date() - today_ny).days

    if dte < 0:
        st.warning("This expiration date is in the past. Select a valid date.")
        return

    if earnings_date != "N/A":
        try:
            earn_dt = datetime.strptime(earnings_date, "%Y-%m-%d").date()
            target_dt = datetime.strptime(target_date, "%Y-%m-%d").date()
            if today_ny <= earn_dt <= target_dt:
                st.warning(f"**Earnings Risk:** An earnings report is scheduled for {earnings_date}. Expect heavy IV Crush.")
        except Exception: pass
    if ex_div_date != "N/A":
        try:
            ex_dt = datetime.strptime(ex_div_date, "%Y-%m-%d").date()
            target_dt = datetime.strptime(target_date, "%Y-%m-%d").date()
            if today_ny <= ex_dt <= target_dt:
                st.warning(f"**Dividend Risk:** The Ex-Dividend date is {ex_div_date}. The stock price will mechanically drop.")
        except Exception: pass

    # --- Expected Move ---
    if dte == 0:
        now_ny = datetime.now(ny_tz)
        market_close = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
        seconds_left = (market_close - now_ny).total_seconds()
        if seconds_left <= 0:
            T = 0.0
        else:
            T = (max(seconds_left, 300) / 86400.0) / 365.0
    else:
        T = dte / 365.0

    if T > 0:
        expected_move = S * sigma * math.sqrt(T)
        em1, em2 = st.columns(2)
        em1.metric("Expected Move (1σ)", f"± ${expected_move:.2f}")
        em2.markdown(f"**Expected Range**  \n### {S - expected_move:.2f} — {S + expected_move:.2f}")

    run_scan = st.button("Scan Options Chain", width="stretch", type="primary", key=f"scan_{key_suffix}")

    if run_scan:
        with st.spinner("Crunching Black-Scholes model..."):
            try:
                stock_obj = yf.Ticker(ticker)
                chain = stock_obj.option_chain(target_date)

                target_chain = chain.puts if opt_type == 'PUTS' else chain.calls

                if not target_chain.empty:
                    target_chain = target_chain.fillna(0)
                    price_col = 'ask' if action == 'BUY' else 'bid'

                    MIN_OPEN_INTEREST = 50
                    MIN_VOLUME = 10
                    MIN_ABS_DELTA = 0.15
                    MAX_ABS_DELTA = 0.80

                    results = []
                    closest_strike = min(target_chain['strike'].tolist(), key=lambda x: abs(x - S))

                    for _, row in target_chain.iterrows():
                        strike = row['strike']
                        market_price = row.get(price_col, 0)
                        bid = row.get('bid', 0)
                        ask = row.get('ask', 0)
                        oi = row.get('openInterest', 0)
                        vol = row.get('volume', 0)
                        iv = row.get('impliedVolatility', 0)

                        if market_price == 0 or iv == 0 or oi < MIN_OPEN_INTEREST or vol < MIN_VOLUME: continue

                        calc = BlackScholesCalculator(S, strike, T, r, sigma, q=div_yield)
                        if opt_type == 'PUTS':
                            fair_value, delta = calc.get_put_data()
                            greeks = calc.get_greeks(opt_type='P')
                        else:
                            fair_value, delta = calc.get_call_data()
                            greeks = calc.get_greeks(opt_type='C')

                        if abs(delta) < MIN_ABS_DELTA or abs(delta) > MAX_ABS_DELTA: continue

                        if action == 'BUY':
                            edge = fair_value - market_price
                        else:
                            edge = market_price - fair_value

                        edge_pct = (edge / market_price * 100) if market_price > 0 else 0

                        # Bid-Ask Spread as % of mid
                        mid = (bid + ask) / 2.0 if (bid + ask) > 0 else 0
                        spread_pct = ((ask - bid) / mid * 100) if mid > 0 else 0

                        row_data = {
                            'Moneyness': get_raw_moneyness(S, strike, opt_type[0], closest_strike),
                            'Strike': strike,
                            'Delta': delta,
                            'Theta': greeks['theta'],
                            'Vega': greeks['vega'],
                            'Price': market_price,
                            'Fair Val': round(fair_value, 2),
                            'Edge (%)': round(edge_pct, 1),
                            'Spread (%)': round(spread_pct, 1),
                            'IV (%)': round(iv * 100, 1),
                            'Vol': int(vol),
                            'OI': int(oi)
                        }

                        if action == 'SELL':
                            effective_dte = max(dte, 1)
                            capital_at_risk = strike if opt_type == 'PUTS' else S
                            ann_roc = (market_price / capital_at_risk) * 100 * (365 / effective_dte)
                            row_data['Ann.ROC (%)'] = round(ann_roc, 1)

                        results.append(row_data)

                    df = pd.DataFrame(results)

                    if not df.empty:
                        best_setups = df.sort_values(by='Edge (%)', ascending=False).head(20)

                        action_word = "Buying" if action == 'BUY' else "Selling"
                        type_word = "Puts" if opt_type == 'PUTS' else "Calls"
                        st.subheader(f"Top Contracts for {action_word} {type_word} | {target_date} ({dte} DTE)")

                        # --- IV Rank display ---
                        if hv_low is not None and hv_high is not None and (hv_high - hv_low) > 0:
                            # Use ATM IV from the chain as current IV proxy
                            atm_rows = best_setups[best_setups['Moneyness'] == 'ATM']
                            if not atm_rows.empty:
                                atm_iv = atm_rows.iloc[0]['IV (%)'] / 100.0
                            else:
                                atm_iv = best_setups.iloc[0]['IV (%)'] / 100.0
                            iv_rank = (atm_iv - hv_low) / (hv_high - hv_low) * 100
                            iv_rank = max(0, min(iv_rank, 100))

                            if iv_rank >= 70:
                                rank_label = "HIGH — Premium is rich. Favor selling."
                            elif iv_rank <= 30:
                                rank_label = "LOW — Premium is cheap. Favor buying."
                            else:
                                rank_label = "NEUTRAL"

                            st.metric("IV Rank (52-wk)", f"{iv_rank:.0f}%  —  {rank_label}")

                        styled_df = style_dataframe(best_setups, sigma)
                        st.dataframe(styled_df, width="stretch", hide_index=True)

                        csv = best_setups.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Clean CSV",
                            data=csv,
                            file_name=f"{ticker}_{action}_{opt_type}_{target_date}.csv",
                            mime="text/csv",
                            width="stretch",
                            key=f"dl_{key_suffix}"
                        )
                    else:
                        st.warning("No liquid, viable contracts found for that date.")
                else:
                    st.warning("No options chain returned for this date.")

            except Exception as e:
                if "RateLimit" in str(type(e).__name__) or "429" in str(e):
                    st.warning("Yahoo Rate Limit Hit: Please wait 60 seconds before scanning the chain.")
                else:
                    st.error(f"Error crunching data: {e}")

# --- MAIN APP UI ---
st.title("📈 LQ Quant Options Scanner")
st.markdown("Identify mathematical edge via Black-Scholes pricing.")

col1, col2 = st.columns(2)
with col1:
    action = st.radio("Action:", ["SELL", "BUY"], horizontal=True)
with col2:
    opt_type = st.radio("Type:", ["PUTS", "CALLS"], horizontal=True)

with st.form("search_form"):
    ticker_input = st.text_input("Enter Ticker(s) — comma-separated:", value="", placeholder="AAPL, TSLA, SPY").strip().upper()
    submit_search = st.form_submit_button("Fetch Options Data", type="primary", use_container_width=True)

if submit_search and ticker_input:
    tickers = [t.strip() for t in ticker_input.split(",") if t.strip()]
    st.session_state['active_tickers'] = tickers
elif submit_search and not ticker_input:
    st.warning("Please enter a ticker symbol.")

if 'active_tickers' in st.session_state:
    tickers = st.session_state['active_tickers']

    if len(tickers) == 1:
        process_ticker(tickers[0], action, opt_type)
    else:
        tabs = st.tabs(tickers)
        for tab, tkr in zip(tabs, tickers):
            with tab:
                process_ticker(tkr, action, opt_type, key_suffix=tkr)
