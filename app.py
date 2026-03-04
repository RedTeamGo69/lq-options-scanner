import math
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime
import streamlit as st
import urllib.request
import json

# --- STREAMLIT PAGE CONFIG ---
st.set_page_config(page_title="LQ Quant Options Scanner", page_icon="📈", layout="centered")

# --- CORE MATH CLASS ---
class BlackScholesCalculator:
    def __init__(self, S, K, T, r, sigma):
        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = max(float(sigma), 0.0001) 

    def _get_d1_d2(self):
        if self.T <= 0: return 0.0, 0.0
        d1 = (math.log(self.S / self.K) + (self.r + (self.sigma ** 2) / 2) * self.T) / (self.sigma * math.sqrt(self.T))
        d2 = d1 - self.sigma * math.sqrt(self.T)
        return d1, d2

    def get_call_data(self):
        if self.T <= 0: return max(0.0, self.S - self.K), 1.0 if self.S > self.K else 0.0
        d1, d2 = self._get_d1_d2()
        price = (self.S * norm.cdf(d1)) - (self.K * math.exp(-self.r * self.T) * norm.cdf(d2))
        return round(price, 2), round(norm.cdf(d1), 2)

    def get_put_data(self):
        if self.T <= 0: return max(0.0, self.K - self.S), -1.0 if self.S < self.K else 0.0
        d1, d2 = self._get_d1_d2()
        price = (self.K * math.exp(-self.r * self.T) * norm.cdf(-d2)) - (self.S * norm.cdf(-d1))
        return round(price, 2), round(norm.cdf(d1) - 1.0, 2)

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
    except Exception:
        pass
    return ticker_symbol

@st.cache_data(ttl=86400)
def get_event_metrics(ticker_symbol):
    earnings_date = "N/A"
    ex_div_date = "N/A"
    
    try:
        stock = yf.Ticker(ticker_symbol)
        
        try:
            # yfinance's built-in calendar handles the anti-bot cookies automatically
            cal = stock.calendar
            if isinstance(cal, dict):
                # 1. Grab Earnings
                if 'Earnings Date' in cal and cal['Earnings Date']:
                    earnings_date = cal['Earnings Date'][0].strftime('%Y-%m-%d')
                
                # 2. Grab Ex-Dividend Date
                if 'Ex-Dividend Date' in cal and cal['Ex-Dividend Date']:
                    ex_div_date = cal['Ex-Dividend Date'].strftime('%Y-%m-%d')
            # Fallback for DataFrame calendar format (sometimes yfinance shifts formats)
            elif isinstance(cal, pd.DataFrame):
                if 'Earnings Date' in cal.index:
                    earnings_date = cal.loc['Earnings Date'].iloc[0].strftime('%Y-%m-%d')
                if 'Ex-Dividend Date' in cal.index:
                    ex_div_date = cal.loc['Ex-Dividend Date'].iloc[0].strftime('%Y-%m-%d')
        except Exception:
            pass
            
    except Exception:
        pass
        
    return earnings_date, ex_div_date

@st.cache_data(ttl=900) 
def get_live_data(ticker_symbol, lookback_days=90):
    stock = yf.Ticker(ticker_symbol)
    hist = stock.history(period=f"{lookback_days}d")
    if hist.empty: return None, None, None
    
    current_price = hist['Close'].iloc[-1]
    hist['Log_Return'] = np.log(hist['Close'] / hist['Close'].shift(1))
    sigma = hist['Log_Return'].std() * np.sqrt(252)
    
    return current_price, sigma, stock.options

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

# --- PANDAS STYLING FOR MOBILE UI ---
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

    styled_df = df.style.map(highlight_moneyness, subset=['Moneyness']) \
                        .map(highlight_edge, subset=['Edge (%)']) \
                        .map(highlight_iv, subset=['IV (%)']) \
                        .format(precision=2)
    return styled_df

# --- MAIN APP UI ---
st.title("📈 LQ Quant Options Scanner")
st.markdown("Identify mathematical edge via Black-Scholes pricing.")

col1, col2 = st.columns(2)
with col1:
    action = st.radio("Action:", ["SELL", "BUY"], horizontal=True)
with col2:
    opt_type = st.radio("Type:", ["PUTS", "CALLS"], horizontal=True)

ticker = st.text_input("Enter Ticker Symbol:", value="").strip().upper()

if ticker:
    with st.spinner(f"Pulling data for {ticker}..."):
        S, sigma, options_dates = get_live_data(ticker)
        r = get_risk_free_rate()
        earnings_date, ex_div_date = get_event_metrics(ticker)

    if S is None:
        st.error(f"Could not find data for {ticker}. Please check the symbol.")
    elif not options_dates:
        st.error(f"No options available for {ticker}.")
    else:
        # Get the clean company name and handle redundant formats
        company_name = get_company_name(ticker)
        if company_name.upper() == ticker.upper():
            st.subheader(f"{ticker}")
        else:
            st.subheader(f"{company_name} ({ticker})")
        
        st.divider()
        
        m1, m2 = st.columns(2)
        m1.metric("Live Price", f"${S:.2f}")
        m2.metric("Hist. Volatility", f"{sigma*100:.1f}%")
        
        m3, m4, m5 = st.columns(3)
        m3.metric("Next Earnings", earnings_date)
        m4.metric("Ex-Div Date", ex_div_date)
        m5.metric("Risk-Free", f"{r*100:.1f}%")

        target_date = st.selectbox("Select Expiration Date:", options_dates)
        
        dte = (datetime.strptime(target_date, "%Y-%m-%d") - datetime.today()).days
        if dte <= 0:
            st.warning("This expiration date is in the past. Select another date.")
        else:
            # Highlight danger if holding through an Earnings Call!
            if earnings_date != "N/A":
                try:
                    earn_dt = datetime.strptime(earnings_date, "%Y-%m-%d").date()
                    target_dt = datetime.strptime(target_date, "%Y-%m-%d").date()
                    today_dt = datetime.today().date()
                    
                    if today_dt <= earn_dt <= target_dt:
                        st.warning(f"⚠️ **Earnings Risk:** An earnings report is scheduled for {earnings_date}. Because this option expires AFTER that date, you will be holding through the event. Expect heavy IV Crush.")
                except Exception:
                    pass
            
            # Highlight danger if holding through an Ex-Dividend Date!
            if ex_div_date != "N/A":
                try:
                    ex_dt = datetime.strptime(ex_div_date, "%Y-%m-%d").date()
                    target_dt = datetime.strptime(target_date, "%Y-%m-%d").date()
                    today_dt = datetime.today().date()
                    
                    if today_dt <= ex_dt <= target_dt:
                        st.warning(f"⚠️ **Dividend Risk:** The Ex-Dividend date is {ex_div_date}. The stock price will mechanically drop on this day, artificially pushing puts closer to being In-The-Money.")
                except Exception:
                    pass

            run_scan = st.button("🚀 Scan Options Chain", width="stretch", type="primary")
            
            if run_scan:
                with st.spinner("Crunching Black-Scholes model..."):
                    T = dte / 365.0 
                    stock_obj = yf.Ticker(ticker)
                    chain = stock_obj.option_chain(target_date)
                    
                    MIN_OPEN_INTEREST = 50
                    MIN_VOLUME = 10
                    MIN_ABS_DELTA = 0.15 
                    
                    results = []
                    target_chain = chain.puts if opt_type == 'PUTS' else chain.calls
                    target_chain = target_chain.fillna(0)
                    price_col = 'ask' if action == 'BUY' else 'bid'
                    
                    # NEW: Find the single exact strike price closest to the live stock price (S)
                    closest_strike = min(target_chain['strike'].tolist(), key=lambda x: abs(x - S))
                    
                    for _, row in target_chain.iterrows():
                        strike, market_price, oi, vol, iv = row['strike'], row[price_col], row['openInterest'], row['volume'], row['impliedVolatility']
                        
                        if market_price == 0 or oi < MIN_OPEN_INTEREST or vol < MIN_VOLUME: continue 
                            
                        calc = BlackScholesCalculator(S, strike, T, r, sigma)
                        fair_value, delta = calc.get_put_data() if opt_type == 'PUTS' else calc.get_call_data()
                            
                        if abs(delta) < MIN_ABS_DELTA: continue
                            
                        if action == 'BUY':
                            edge = fair_value - market_price
                        else:
                            edge = market_price - fair_value
                            
                        edge_pct = (edge / market_price * 100) if market_price > 0 else 0
                        
                        row_data = {
                            # NEW: Pass the closest_strike into the moneyness function
                            'Moneyness': get_raw_moneyness(S, strike, opt_type[0], closest_strike),
                            'Strike': strike,
                            'Delta': delta,
                            'Price': market_price,
                            'Fair Val': round(fair_value, 2),
                            'Edge (%)': round(edge_pct, 1),
                            'IV (%)': round(iv * 100, 1),
                            'Vol': int(vol),
                            'OI': int(oi)
                        }
                        
                        if action == 'SELL':
                            ann_roc = (market_price / strike) * 100 * (365 / dte) if dte > 0 else 0
                            row_data['Ann.ROC (%)'] = round(ann_roc, 1)
                            
                        results.append(row_data)

                    df = pd.DataFrame(results)
                    
                    if not df.empty:
                        best_setups = df.sort_values(by='Edge (%)', ascending=False).head(20)
                        
                        st.subheader(f"Top Setups ({dte} DTE)")
                        
                        styled_df = style_dataframe(best_setups, sigma)
                        st.dataframe(styled_df, width="stretch", hide_index=True)
                        
                        csv = best_setups.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Clean CSV",
                            data=csv,
                            file_name=f"{ticker}_{action}_{opt_type}_{target_date}.csv",
                            mime="text/csv",
                            width="stretch"
                        )
                    else:
                        st.warning("No liquid, viable contracts found for that date.")
