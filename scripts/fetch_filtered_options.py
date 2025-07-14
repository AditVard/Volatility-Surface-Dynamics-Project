import requests
import csv
import pandas as pd
from datetime import datetime
import time

# ---------------- CONFIG ----------------
spot_file = r"C:\Users\adity\Downloads\vol surface dynamics\data\raw\spot.csv"
output_file = r"C:\Users\adity\Downloads\vol surface dynamics\data\processed\filtered_options_combined.csv"
strike_step = 50
allowed_time = "09:16"

# ---------------- Fetch Valid Expiries ----------------
def fetch_all_expiries():
    url = "https://apih.stocksrin.com/history/simulator/allexpiry"
    params = {"symbol": "NIFTY", "type": "OPT"}

    headers = {
        "Accept": "application/json, text/plain, */*",
        "Authorization": "V2NkelpXTmpaWlFJ6Wd4MA==",
        "Connection": "keep-alive",
        "Host": "apih.stocksrin.com",
        "Origin": "https://www.stocksrin.com",
        "Referer": "https://www.stocksrin.com/",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:138.0) Gecko/20100101 Firefox/138.0"
    }

    response = requests.get(url, headers=headers, params=params, timeout=10)
    response.raise_for_status()
    expiry_strings = response.json()
    return sorted([datetime.strptime(d, "%d-%b-%Y").date() for d in expiry_strings])

# ---------------- Get Next N Expiries > Today ----------------
def get_next_n_expiries(date, all_expiries, n=3):
    future_expiries = [e for e in all_expiries if e > date]
    return future_expiries[:n]

# ---------------- Fetch Option Data ----------------
def fetch_option_data(df_915, expiry_str, expiry_date):
    atm_strikes_data = []
    for _, row in df_915.iterrows():
        spot = row['spot']
        base_strike = round(spot / strike_step) * strike_step
        strikes = [base_strike + i * strike_step for i in range(-5, 6)]
        atm_strikes_data.append({
            'date': row['date'],
            'time': row['time'],
            'spot': spot,
            'strikes': strikes
        })

    all_strikes = sorted(set(strike for atm in atm_strikes_data for strike in atm['strikes']))

    url = "https://apih.stocksrin.com/history/simulator/option/data"
    headers = {
        'Accept': '*/*',
        'Authorization': 'Y2NKalpXTkpaWFJ6WVd4MA==',
        'User': 'RS-22196',
        'User-Agent': 'Mozilla/5.0'
    }

    output_rows = []

    for strike in all_strikes:
        params = {
            'from': int(datetime(2025, 1, 1).timestamp() * 1000),
            'to': int(datetime(2025, 12, 31).timestamp() * 1000),
            'index': 'NIFTY',
            'strike': str(strike),
            'optiontype': 'CE',
            'expiry': expiry_str
        }

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            if data.get('status') != 'success':
                continue

            for entry in data.get('datav2', []):
                parts = entry.split(',')
                if len(parts) < 6:
                    continue

                ts = int(parts[1])
                entry_time = datetime.fromtimestamp(ts)
                time_str = entry_time.strftime('%H:%M')
                date_str = entry_time.strftime('%d-%m-%Y')

                if time_str != allowed_time:
                    continue

                spot = ''
                for row in atm_strikes_data:
                    if row['date'] == date_str and row['time'] == time_str and strike in row['strikes']:
                        spot = row['spot']
                        break

                if spot == '':
                    continue

                output_rows.append({
                    'date': date_str,
                    'time': time_str,
                    'expiry': expiry_str,
                    'strike': strike,
                    'spot_price': spot,
                    'high': float(parts[3])
                })

            time.sleep(0.4)

        except Exception as e:
            print(f"❌ Error fetching strike {strike} for {expiry_str}: {e}")

    return output_rows

# ---------------- MAIN ----------------
def main():
    all_expiries = fetch_all_expiries()
    print(f"✅ Got {len(all_expiries)} expiries")

    df = pd.read_csv(spot_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.strftime('%d-%m-%Y')
    df['time'] = df['datetime'].dt.strftime('%H:%M')

    df_915 = df[df['time'] == allowed_time].copy()

    all_data = []

    for _, row in df_915.iterrows():
        date = row['datetime'].date()
        next_expiries = get_next_n_expiries(date, all_expiries, n=3)
        temp_df = pd.DataFrame([row])
        for expiry in next_expiries:
            expiry_str = expiry.strftime('%d-%b-%Y')
            print(f"📅 Fetching for {row['date']} and expiry {expiry_str}")
            rows = fetch_option_data(temp_df, expiry_str, expiry)
            all_data.extend(rows)

    pd.DataFrame(all_data).to_csv(output_file, index=False)
    print(f"✅ Saved all option data to: {output_file}")

if __name__ == "__main__":
    main()
