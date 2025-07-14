import pandas as pd
import numpy as np
import os
from scipy.interpolate import SmoothBivariateSpline

# ------------------------ #
# Config
# ------------------------ #
input_csv = '../../data/processed/greeks_iv_from_column.csv'
save_dir = './fengler_iv_surfaces'
os.makedirs(save_dir, exist_ok=True)

# ------------------------ #
# Load and clean data
# ------------------------ #
df = pd.read_csv(input_csv)
df.dropna(subset=['strike', 'spot', 'iv', 'expiry', 'date'], inplace=True)
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
df['expiry'] = pd.to_datetime(df['expiry'], format='%d-%b-%Y', errors='coerce')
df.dropna(subset=['date', 'expiry'], inplace=True)
df['ttm'] = (df['expiry'] - df['date']).dt.days / 365
df = df[df['ttm'] > 1/252]
df['log_moneyness'] = np.log(df['strike'] / df['spot'])

# ------------------------ #
# Track results
# ------------------------ #
skipped_dates = []
failed_dates = []
saved_dates = []

# ------------------------ #
# Loop through all dates
# ------------------------ #
all_dates = sorted(df['date'].unique())

for d in all_dates:
    df_day = df[df['date'] == d].copy()
    date_str = d.strftime('%Y-%m-%d')

    x = df_day['strike'].values
    y = df_day['ttm'].values
    z = df_day['iv'].values

    if len(df_day) < 5:
        skipped_dates.append((date_str, "very few rows (<5)"))
        print(f"⚠️ Skipping {date_str} → very few rows")
        continue

    try:
        spline = SmoothBivariateSpline(x, y, z, kx=2, ky=2, s=1e-5)
        x_dense = np.linspace(min(x), max(x), 50)
        y_dense = np.linspace(min(y), max(y), 50)
        STRIKE, TTM = np.meshgrid(x_dense, y_dense)
        Z = spline.ev(STRIKE.ravel(), TTM.ravel()).reshape(STRIKE.shape)

        np.savez_compressed(
            os.path.join(save_dir, f'iv_surface_{date_str}.npz'),
            strike=STRIKE,
            ttm=TTM,
            iv=Z
        )
        print(f"✅ Saved {date_str}")
        saved_dates.append(date_str)

    except Exception as e:
        print(f"❌ Failed {date_str} → {str(e)}")
        failed_dates.append((date_str, str(e)))

# ------------------------ #
# Summary
# ------------------------ #
print("\n🧾 Summary:")
print(f"✅ Saved: {len(saved_dates)} surfaces")
print(f"⚠️ Skipped: {len(skipped_dates)}")
print(f"❌ Failed: {len(failed_dates)}")

if skipped_dates:
    print("\n⚠️ Skipped Dates (with reasons):")
    for d, reason in skipped_dates:
        print(f"  - {d} → {reason}")

if failed_dates:
    print("\n❌ Failed Dates (with errors):")
    for d, err in failed_dates:
        print(f"  - {d} → {err}")
