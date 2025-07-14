# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import SmoothBivariateSpline
# from scipy.stats import norm
# import os

# # -------------------------
# # Config
# # -------------------------
# r = 0.06
# S0 = 24000
# save_dir = '../../data/processed/dupire_surfaces'
# os.makedirs(save_dir, exist_ok=True)

# # -------------------------
# # Load Data
# # -------------------------
# df = pd.read_csv('../../data/processed/greeks_iv_from_column.csv')
# df.dropna(subset=['strike', 'spot', 'iv', 'expiry', 'date'], inplace=True)
# df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
# df['expiry'] = pd.to_datetime(df['expiry'], format='%d-%b-%Y')
# df['ttm'] = (df['expiry'] - df['date']).dt.days / 365
# df = df[df['ttm'] > 0]
# df['log_moneyness'] = np.log(df['strike'] / df['spot'])

# # -------------------------
# # Black-Scholes Call Price
# # -------------------------
# def bs_call_price(S, K, T, sigma, r):
#     if T <= 0 or sigma <= 0:
#         return 0
#     d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
#     d2 = d1 - sigma * np.sqrt(T)
#     return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# # -------------------------
# # Loop Over All Dates
# # -------------------------
# dates = sorted(df['date'].unique())

# for dt in dates:
#     print(f"📅 Processing date: {dt.date()}")

#     df_day = df[df['date'] == dt]
#     if df_day.empty:
#         print(f"❌ Skipping {dt.date()} — no data")
#         continue

#     try:
#         x = df_day['log_moneyness'].values
#         y = df_day['ttm'].values
#         z = df_day['iv'].values

#         spline = SmoothBivariateSpline(x, y, z, kx=3, ky=3, s=0.001)

#         k_vals = np.linspace(min(x), max(x), 50)
#         T_vals = np.linspace(min(y), max(y), 50)
#         K_grid = S0 * np.exp(k_vals)
#         K_mesh, T_mesh = np.meshgrid(K_grid, T_vals)

#         dK = K_grid[1] - K_grid[0]
#         dT = T_vals[1] - T_vals[0]

#         local_vol_surface = np.zeros_like(K_mesh)

#         for i in range(1, len(T_vals) - 1):
#             for j in range(1, len(k_vals) - 1):
#                 K = K_mesh[i, j]
#                 T = T_mesh[i, j]

#                 sigma = spline.ev(np.log(K / S0), T)

#                 C = bs_call_price(S0, K, T, sigma, r)
#                 C_T_plus = bs_call_price(S0, K, T + dT, spline.ev(np.log(K / S0), T + dT), r)
#                 dC_dT = (C_T_plus - C) / dT

#                 C_Kplus = bs_call_price(S0, K + dK, T, spline.ev(np.log((K + dK) / S0), T), r)
#                 C_Kminus = bs_call_price(S0, K - dK, T, spline.ev(np.log((K - dK) / S0), T), r)
#                 d2C_dK2 = (C_Kplus - 2 * C + C_Kminus) / (dK ** 2)

#                 if d2C_dK2 > 0:
#                     local_var = dC_dT / (0.5 * K ** 2 * d2C_dK2)
#                     local_vol_surface[i, j] = np.sqrt(max(local_var, 0))
#                 else:
#                     local_vol_surface[i, j] = np.nan

#         # Save .npz file
#         save_path = os.path.join(save_dir, f'local_vol_surface_{dt.date()}.npz')
#         np.savez_compressed(save_path,
#                             K=K_mesh,
#                             T=T_mesh,
#                             local_vol=local_vol_surface)
#         print(f"✅ Saved: {save_path}")

#     except Exception as e:
#         print(f"❌ Error on {dt.date()}: {e}")
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import SmoothBivariateSpline
# from scipy.stats import norm
# import os

# # -------------------------
# # Config
# # -------------------------
# r = 0.06
# S0 = 24000
# save_dir = '../../data/processed/dupire_surfaces'
# os.makedirs(save_dir, exist_ok=True)

# # -------------------------
# # Load Data
# # -------------------------
# df = pd.read_csv('../../data/processed/greeks_iv_from_column.csv')
# df.dropna(subset=['strike', 'spot', 'iv', 'expiry', 'date'], inplace=True)
# df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
# df['expiry'] = pd.to_datetime(df['expiry'], format='%d-%b-%Y')
# df['ttm'] = (df['expiry'] - df['date']).dt.days / 365
# df = df[df['ttm'] > 0]
# df['log_moneyness'] = np.log(df['strike'] / df['spot'])

# # -------------------------
# # Black-Scholes Call Price
# # -------------------------
# def bs_call_price(S, K, T, sigma, r):
#     if T <= 0 or sigma <= 0:
#         return 0
#     d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
#     d2 = d1 - sigma * np.sqrt(T)
#     return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# # -------------------------
# # Loop Over All Dates
# # -------------------------
# dates = sorted(df['date'].unique())

# for dt in dates:
#     print(f"📅 Processing date: {dt.date()}")

#     df_day = df[df['date'] == dt]
#     if df_day.empty:
#         print(f"❌ Skipping {dt.date()} — no data")
#         continue

#     try:
#         x = df_day['log_moneyness'].values
#         y = df_day['ttm'].values
#         z = df_day['iv'].values

#         spline = SmoothBivariateSpline(x, y, z, kx=3, ky=3, s=0.001)

#         k_vals = np.linspace(min(x), max(x), 50)
#         T_vals = np.linspace(min(y), max(y), 50)
#         K_grid = S0 * np.exp(k_vals)
#         K_mesh, T_mesh = np.meshgrid(K_grid, T_vals)

#         dK = K_grid[1] - K_grid[0]
#         dT = T_vals[1] - T_vals[0]

#         local_vol_surface = np.full_like(K_mesh, np.nan)
#         nan_count = 0

#         for i in range(1, len(T_vals) - 1):
#             for j in range(1, len(k_vals) - 1):
#                 K = K_mesh[i, j]
#                 T = T_mesh[i, j]

#                 sigma = spline.ev(np.log(K / S0), T)

#                 C = bs_call_price(S0, K, T, sigma, r)
#                 C_T_plus = bs_call_price(S0, K, T + dT, spline.ev(np.log(K / S0), T + dT), r)
#                 dC_dT = (C_T_plus - C) / dT

#                 C_Kplus = bs_call_price(S0, K + dK, T, spline.ev(np.log((K + dK) / S0), T), r)
#                 C_Kminus = bs_call_price(S0, K - dK, T, spline.ev(np.log((K - dK) / S0), T), r)
#                 d2C_dK2 = (C_Kplus - 2 * C + C_Kminus) / (dK ** 2)

#                 if d2C_dK2 > 0:
#                     local_var = dC_dT / (0.5 * K ** 2 * d2C_dK2)
#                     if local_var > 0:
#                         local_vol_surface[i, j] = np.sqrt(local_var)
#                     else:
#                         nan_count += 1
#                 else:
#                     nan_count += 1

#         # Check NaN threshold
#         nan_ratio = nan_count / local_vol_surface.size
#         if nan_ratio > 0.3:
#             print(f"⚠️ Too many NaNs ({nan_ratio:.1%}) — skipping {dt.date()}")
#             continue
        
#         # Save .npz
#         save_path = os.path.join(save_dir, f'local_vol_surface_{dt.date()}.npz')
#         np.savez_compressed(save_path,
#                             K=K_mesh,
#                             T=T_mesh,
#                             local_vol=local_vol_surface)
#         print(f"✅ Saved: {save_path}")

#     except Exception as e:
#         print(f"❌ Error on {dt.date()}: {e}")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import SmoothBivariateSpline
from scipy.stats import norm
import os

# -------------------------
# Config
# -------------------------
r = 0.06
S0 = 24000
save_dir = '../../data/processed/dupire_surfaces'
os.makedirs(save_dir, exist_ok=True)

# -------------------------
# Load Data
# -------------------------
df = pd.read_csv('../../data/processed/greeks_iv_from_column.csv')
df.dropna(subset=['strike', 'spot', 'iv', 'expiry', 'date'], inplace=True)
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
df['expiry'] = pd.to_datetime(df['expiry'], format='%d-%b-%Y')
df['ttm'] = (df['expiry'] - df['date']).dt.days / 365
df = df[df['ttm'] > 0]
df['log_moneyness'] = np.log(df['strike'] / df['spot'])

# -------------------------
# Black-Scholes Call Price
# -------------------------
def bs_call_price(S, K, T, sigma, r):
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# -------------------------
# Loop Over All Dates
# -------------------------
dates = sorted(df['date'].unique())

for dt in dates:
    print(f"📅 Processing date: {dt.date()}")
    df_day = df[df['date'] == dt]
    if df_day.empty:
        print(f"❌ Skipping {dt.date()} — no data")
        continue

    try:
        x = df_day['log_moneyness'].values
        y = df_day['ttm'].values
        z = df_day['iv'].values

        spline = SmoothBivariateSpline(x, y, z, kx=3, ky=3, s=0.001)

        k_vals = np.linspace(min(x), max(x), 50)
        T_vals = np.linspace(min(y), max(y), 50)
        K_grid = S0 * np.exp(k_vals)
        K_mesh, T_mesh = np.meshgrid(K_grid, T_vals)

        dK = K_grid[1] - K_grid[0]
        dT = T_vals[1] - T_vals[0]

        local_vol_surface = np.full_like(K_mesh, np.nan)
        nan_count = 0

        for i in range(1, len(T_vals) - 1):
            for j in range(1, len(k_vals) - 1):
                K = K_mesh[i, j]
                T = T_mesh[i, j]

                sigma = spline.ev(np.log(K / S0), T)

                C = bs_call_price(S0, K, T, sigma, r)
                C_T_plus = bs_call_price(S0, K, T + dT, spline.ev(np.log(K / S0), T + dT), r)
                dC_dT = (C_T_plus - C) / dT

                C_Kplus = bs_call_price(S0, K + dK, T, spline.ev(np.log((K + dK) / S0), T), r)
                C_Kminus = bs_call_price(S0, K - dK, T, spline.ev(np.log((K - dK) / S0), T), r)
                d2C_dK2 = (C_Kplus - 2 * C + C_Kminus) / (dK ** 2)

                if d2C_dK2 > 0:
                    local_var = dC_dT / (0.5 * K ** 2 * d2C_dK2)
                    if local_var > 0:
                        local_vol_surface[i, j] = np.sqrt(local_var)
                    else:
                        nan_count += 1
                else:
                    nan_count += 1

        # -------------------------
        # NaN Handling & Save
        # -------------------------
        nan_ratio = nan_count / local_vol_surface.size
        if nan_ratio > 0.3:
            print(f"⚠️ Too many NaNs ({nan_ratio:.1%}) — skipping {dt.date()}")
            continue

        # Fill remaining NaNs (mean or 0.0 — choose one)
        # Option 1: Fill with mean
        mean_val = np.nanmean(local_vol_surface)
        local_vol_surface = np.nan_to_num(local_vol_surface, nan=mean_val)

        # Option 2: Fill with 0 instead (uncomment this instead of above if you prefer)
        # local_vol_surface = np.nan_to_num(local_vol_surface, nan=0.0)

        save_path = os.path.join(save_dir, f'local_vol_surface_{dt.date()}.npz')
        np.savez_compressed(save_path,
                            K=K_mesh,
                            T=T_mesh,
                            local_vol=local_vol_surface)
        print(f"✅ Saved: {save_path}")

    except Exception as e:
        print(f"❌ Error on {dt.date()}: {e}")
