import requests
import csv
import pandas as pd
from datetime import datetime
import time

# ---------------- CONFIG ----------------
spot_file = r"C:\Users\adity\Downloads\vol surface dynamics\data\processed\spot_14to21jun2025.csv"
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

# ---------------- Get Next Expiry > Today ----------------
def get_next_expiry(date, all_expiries):
    future_expiries = [e for e in all_expiries if e > date]
    if len(future_expiries) >= 2:
        return future_expiries[1]  # Skip the nearest, use the next
    elif len(future_expiries) == 1:
        return future_expiries[0]  # Fallback if only one available
    else:
        return None


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

    df['date_only'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    start_date = pd.to_datetime("2025-06-14")
    end_date = pd.to_datetime("2025-06-20")

    df_915 = df[(df['time'] == allowed_time) & (df['date_only'] >= start_date) & (df['date_only'] <= end_date)].copy()

    df_915['expiry_date'] = df_915['datetime'].dt.date.apply(lambda d: get_next_expiry(d, all_expiries))
    df_915['expiry_str'] = df_915['expiry_date'].apply(lambda d: d.strftime('%d-%b-%Y') if d else '')

    all_data = []

    for expiry_str, group in df_915.groupby('expiry_str'):
        if not expiry_str:
            continue
        expiry_date = datetime.strptime(expiry_str, "%d-%b-%Y").date()
        print(f"📅 Fetching for expiry {expiry_str} ({len(group)} days)")
        rows = fetch_option_data(group, expiry_str, expiry_date)
        all_data.extend(rows)

    pd.DataFrame(all_data).to_csv(output_file, index=False)
    print(f"✅ Saved all option data to: {output_file}")

if __name__ == "__main__":
    main()
