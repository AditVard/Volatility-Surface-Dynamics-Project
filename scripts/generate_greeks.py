import requests
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import norm

# ------------------ CONFIG ------------------
OPTION_DATA = "../../data/processed/filtered_options_combined.csv"
OUTPUT_FILE = "../../data/processed/greeks_iv_from_column.csv"
R = 0.065  # risk-free rate

# ------------------ Black-Scholes Functions ------------------
def bs_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_delta(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def bs_gamma(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def bs_vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T) / 100

def implied_volatility(S, K, T, r, market_price, tol=1e-5, max_iter=100):
    sigma = 0.2
    for _ in range(max_iter):
        price = bs_price(S, K, T, r, sigma)
        vega = bs_vega(S, K, T, r, sigma) * 100
        if vega == 0 or np.isnan(price):
            return np.nan
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        sigma -= diff / vega
    return np.nan

# ------------------ Main Logic ------------------
def main():
    print("📂 Reading data...")
    df = pd.read_csv(OPTION_DATA)

    # Parse datetime and expiry columns
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format="%d-%m-%Y %H:%M")
    df['expiry'] = pd.to_datetime(df['expiry'], format="%d-%b-%Y").dt.date

    rows = []

    for _, row in df.iterrows():
        dt = row['datetime'].date()
        expiry = row['expiry']
        strike = row['strike']
        spot = row['spot_price']
        premium = row['high']

        # Time to expiry
        T = (expiry - dt).days / 365
        if T <= 0:
            continue  # skip if expired

        try:
            iv = implied_volatility(spot, strike, T, R, premium)
            if np.isnan(iv):
                continue

            delta = bs_delta(spot, strike, T, R, iv)
            gamma = bs_gamma(spot, strike, T, R, iv)
            vega = bs_vega(spot, strike, T, R, iv)

            print(f"📅 {row['date']} {row['time']} → Expiry: {expiry.strftime('%d-%b-%Y')}")

            rows.append({
                "date": row['date'],
                "time": row['time'],
                "expiry": expiry.strftime("%d-%b-%Y"),
                "strike": strike,
                "spot": spot,
                "premium": premium,
                "iv": iv,
                "delta": delta,
                "gamma": gamma,
                "vega": vega
            })
        except Exception as e:
            print(f"⚠️ Error at {row['date']} {row['time']} strike {strike}: {e}")

    pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Done! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

