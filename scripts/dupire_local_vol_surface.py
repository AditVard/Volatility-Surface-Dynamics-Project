# from datetime import date
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import SmoothBivariateSpline
# from scipy.stats import norm

# # -------------------------
# # Config
# # -------------------------
# r = 0.06  # Risk-free rate
# S0 = 24000  # Approx spot price

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
# # Choose Date
# # -------------------------
# available_dates = df['date'].drop_duplicates().sort_values()
# print("Available Dates:")
# for i, dt in enumerate(available_dates.dt.strftime('%d-%b-%Y')):
#     print(f"{i+1}. {dt}")
    
# choice = int(input("Enter the index of the date you want to use: ")) - 1
# chosen_date = available_dates.iloc[choice]
# print(f"Selected Date: {chosen_date.strftime('%d-%b-%Y')}")

# df_day = df[df['date'] == chosen_date].copy()

# # -------------------------
# # Interpolate IV Surface
# # -------------------------
# x = df_day['log_moneyness'].values
# y = df_day['ttm'].values
# z = df_day['iv'].values

# spline = SmoothBivariateSpline(x, y, z, kx=3, ky=3, s=0.001)

# # Define grid
# k_vals = np.linspace(min(x), max(x), 100)
# T_vals = np.linspace(min(y), max(y), 100)
# K_grid = S0 * np.exp(k_vals)
# K_mesh, T_mesh = np.meshgrid(K_grid, T_vals)

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
# # Compute Local Vol using Dupire
# # -------------------------
# dK = K_grid[1] - K_grid[0]
# dT = T_vals[1] - T_vals[0]

# local_vol_surface = np.zeros_like(K_mesh)

# for i in range(1, len(T_vals) - 1):
#     for j in range(1, len(k_vals) - 1):
#         K = K_mesh[i, j]
#         T = T_mesh[i, j]

#         sigma = spline.ev(np.log(K / S0), T)

#         # Partial Derivatives
#         C = bs_call_price(S0, K, T, sigma, r)
#         C_T_plus = bs_call_price(S0, K, T + dT, spline.ev(np.log(K / S0), T + dT), r)
#         dC_dT = (C_T_plus - C) / dT

#         C_Kplus = bs_call_price(S0, K + dK, T, spline.ev(np.log((K + dK) / S0), T), r)
#         C_Kminus = bs_call_price(S0, K - dK, T, spline.ev(np.log((K - dK) / S0), T), r)
#         d2C_dK2 = (C_Kplus - 2 * C + C_Kminus) / (dK ** 2)
#         threshold = 0.05 * local_vol_surface.size 
#         if d2C_dK2 > 0:
#             local_var = dC_dT / (0.5 * K ** 2 * d2C_dK2)
#             local_vol_surface[i, j] = np.sqrt(max(local_var, 0))
#         else:
#             local_vol_surface[i, j] = np.nan
#             print(f"⚠️ Skipping {date} — too many NaNs in surface")
#             continue
#             print(f"NaN at (T={T:.4f}, K={K:.2f}) — d2C_dK2={d2C_dK2:.6f}, dC_dT={dC_dT:.6f}, sigma={sigma:.4f}")

# # -------------------------
# # Plot Local Vol Surface
# # -------------------------
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(np.log(K_mesh / S0), T_mesh, local_vol_surface,
#                        cmap='plasma', edgecolor='k', alpha=0.9)
# ax.set_xlabel('Log-Moneyness')
# ax.set_ylabel('Time to Expiry (Years)')
# ax.set_zlabel('Local Volatility')
# ax.set_title(f"Dupire Local Vol Surface on {chosen_date.strftime('%d-%b-%Y')}")
# fig.colorbar(surf, shrink=0.5, aspect=10)
# plt.tight_layout()
# plt.show()
from datetime import date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import SmoothBivariateSpline
from scipy.stats import norm

# -------------------------
# Config
# -------------------------
r = 0.06  # Risk-free rate
S0 = 24000  # Approx spot price

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
# Choose Date
# -------------------------
available_dates = df['date'].drop_duplicates().sort_values()
print("Available Dates:")
for i, dt in enumerate(available_dates.dt.strftime('%d-%b-%Y')):
    print(f"{i+1}. {dt}")
    
choice = int(input("Enter the index of the date you want to use: ")) - 1
chosen_date = available_dates.iloc[choice]
print(f"Selected Date: {chosen_date.strftime('%d-%b-%Y')}")
df_day = df[df['date'] == chosen_date].copy()

# -------------------------
# Interpolate IV Surface
# -------------------------
x = df_day['log_moneyness'].values
y = df_day['ttm'].values
z = df_day['iv'].values

spline = SmoothBivariateSpline(x, y, z, kx=3, ky=3, s=0.001)

k_vals = np.linspace(min(x), max(x), 100)
T_vals = np.linspace(min(y), max(y), 100)
K_grid = S0 * np.exp(k_vals)
K_mesh, T_mesh = np.meshgrid(K_grid, T_vals)

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
# Compute Dupire Local Volatility
# -------------------------
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
# Skip plotting if too many NaNs
# -------------------------
nan_ratio = nan_count / local_vol_surface.size
if nan_ratio > 0.3:
    print(f"⚠️ Skipping plot: Too many NaNs on {chosen_date.strftime('%d-%b-%Y')} ({nan_ratio:.2%})")
    exit()

# -------------------------
# Plot Surface
# -------------------------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(np.log(K_mesh / S0), T_mesh, local_vol_surface,
                       cmap='plasma', edgecolor='k', alpha=0.9)
ax.set_xlabel('Log-Moneyness')
ax.set_ylabel('Time to Expiry (Years)')
ax.set_zlabel('Local Volatility')
ax.set_title(f"Dupire Local Vol Surface on {chosen_date.strftime('%d-%b-%Y')}")
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.tight_layout()
plt.show()
