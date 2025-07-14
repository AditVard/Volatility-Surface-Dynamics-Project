# # import numpy as np
# # import matplotlib.pyplot as plt
# # from matplotlib import cm
# # import os

# # # --------------------------
# # # Config paths
# # # --------------------------
# # pca_path = r'C:\Users\adity\Downloads\vol surface dynamics\data\processed\iv_pca_output.npz'
# # ref_surface_path = r'C:\Users\adity\Downloads\vol surface dynamics\data\processed\fengler_iv_surfaces\iv_surface_2025-06-13.npz'

# # # --------------------------
# # # Forecasted PCA values (you can replace this with actual forecasted values)
# # # Shape: (n_days, 5 components)
# # X_pca_forecast = np.array([
# #     [6.1, -0.2, 0.12, -0.05, 0.01],
# #     [6.5, -0.1, 0.13, -0.04, 0.00],
# #     [6.8,  0.0, 0.14, -0.03, 0.01],
# #     [7.0,  0.1, 0.15, -0.02, 0.01],
# #     [7.3,  0.2, 0.16, -0.01, 0.02]
# # ])

# # forecast_dates = ['2025-06-14', '2025-06-17', '2025-06-18', '2025-06-19', '2025-06-20']

# # # --------------------------
# # # Load PCA components
# # # --------------------------
# # data = np.load(pca_path)
# # components = data['pca_model']  # shape: (5, 2500)

# # # --------------------------
# # # Load STRIKE, TTM grid from any surface
# # # --------------------------
# # ref = np.load(ref_surface_path)
# # STRIKE = ref['strike']
# # TTM = ref['ttm']

# # # --------------------------
# # # Let user choose a date to view
# # # --------------------------
# # print("\n📅 Forecasted Dates:")
# # for i, date in enumerate(forecast_dates):
# #     print(f"{i + 1}: {date}")

# # choice = int(input("\nEnter the number of the forecast date to view: ")) - 1
# # chosen_date = forecast_dates[choice]
# # chosen_pca = X_pca_forecast[choice]

# # # --------------------------
# # # Reconstruct IV Surface
# # # --------------------------
# # iv_flat = np.dot(chosen_pca, components)  # shape (2500,)
# # iv_surface = iv_flat.reshape(50, 50)

# # # --------------------------
# # # Plot Surface
# # # --------------------------
# # fig = plt.figure(figsize=(12, 8))
# # ax = fig.add_subplot(111, projection='3d')
# # surf = ax.plot_surface(STRIKE, TTM, iv_surface, cmap=cm.viridis, edgecolor='k', alpha=0.85)

# # ax.set_title(f"📅 Predicted IV Surface - {chosen_date}", fontsize=15)
# # ax.set_xlabel("Strike")
# # ax.set_ylabel("TTM (yrs)")
# # ax.set_zlabel("Implied Volatility")
# # fig.colorbar(surf, shrink=0.5, aspect=10)
# # plt.tight_layout()
# # plt.show()
# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # from matplotlib import cm
# # from scipy.interpolate import RegularGridInterpolator
# # from sklearn.metrics import mean_squared_error
# # from datetime import datetime

# # # -----------------------------------
# # # Config Paths
# # # -----------------------------------
# # pca_path = r'C:\Users\adity\Downloads\vol surface dynamics\data\processed\iv_pca_output.npz'
# # ref_surface_path = r'C:\Users\adity\Downloads\vol surface dynamics\data\processed\fengler_iv_surfaces\iv_surface_2025-06-13.npz'
# # actual_iv_path = r'C:\Users\adity\Downloads\vol surface dynamics\data\processed\options_with_iv_junelater.csv'
# # chosen_date = '2025-06-17'  # 🔧 Change to desired date
# # forecasted_pcs = [6.5, -0.1, 0.13, -0.04, 0.00]  # 🔧 Replace with your actual ARIMA forecast

# # # -----------------------------------
# # # Load PCA & Grid Info
# # # -----------------------------------
# # data = np.load(pca_path)
# # components = data['pca_model']  # shape: (5, 2500)

# # ref = np.load(ref_surface_path)
# # STRIKE = ref['strike']  # shape (50, 50)
# # TTM = ref['ttm']        # shape (50, 50)

# # strike_flat = STRIKE[0]
# # ttm_flat = TTM[:, 0]

# # # -----------------------------------
# # # Reconstruct Predicted IV Surface
# # # -----------------------------------
# # iv_flat = np.dot(forecasted_pcs, components)
# # iv_surface_pred = iv_flat.reshape(50, 50)

# # # -----------------------------------
# # # Load Actual IV Data for the Date
# # # -----------------------------------
# # df = pd.read_csv(actual_iv_path)
# # df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
# # df = df[df['date'] == pd.to_datetime(chosen_date)]

# # # Clean any missing data
# # df = df.dropna(subset=['strike', 'IV', 'spot_price', 'T'])

# # # -----------------------------------
# # # Compute log-moneyness
# # # -----------------------------------
# # df['log_moneyness'] = np.log(df['strike'] / df['spot_price'])

# # # Interpolator from predicted surface (for comparison)
# # interp_func = RegularGridInterpolator((ttm_flat, strike_flat), iv_surface_pred)

# # # Prepare comparison points
# # actual_pts = []
# # predicted_pts = []

# # for _, row in df.iterrows():
# #     strike = row['strike']
# #     ttm = row['T']
# #     iv_actual = row['IV']

# #     # Only compare if strike and ttm are within bounds
# #     if ttm_flat.min() <= ttm <= ttm_flat.max() and strike_flat.min() <= strike <= strike_flat.max():
# #         try:
# #             iv_pred = interp_func([[ttm, strike]])[0]
# #             actual_pts.append(iv_actual)
# #             predicted_pts.append(iv_pred)
# #         except:
# #             continue

# # # -----------------------------------
# # # Compute RMSE
# # # -----------------------------------
# # rmse = np.sqrt(mean_squared_error(actual_pts, predicted_pts))
# # print(f"✅ RMSE between predicted surface and actual IVs: {rmse:.4f}")

# # # -----------------------------------
# # # Plotting: Surface + Actual Points
# # # -----------------------------------
# # fig = plt.figure(figsize=(12, 8))
# # ax = fig.add_subplot(111, projection='3d')

# # # Surface
# # surf = ax.plot_surface(STRIKE, TTM, iv_surface_pred, cmap=cm.viridis, alpha=0.8, edgecolor='k')

# # # Actual IV points
# # ax.scatter(df['strike'], df['T'], df['IV'], color='red', s=30, label='Actual IV', depthshade=True)

# # ax.set_title(f"📊 Predicted IV Surface vs Actual IVs on {chosen_date}\nRMSE: {rmse:.4f}", fontsize=14)
# # ax.set_xlabel("Strike")
# # ax.set_ylabel("TTM (yrs)")
# # ax.set_zlabel("Implied Volatility")
# # ax.legend()
# # plt.tight_layout()
# # plt.show()
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from sklearn.metrics import mean_squared_error
# from scipy.interpolate import RegularGridInterpolator

# # --------------------------
# # CONFIG PATHS
# # --------------------------
# pca_path = r'C:\Users\adity\Downloads\vol surface dynamics\data\processed\iv_pca_output.npz'
# ref_surface_path = r'C:\Users\adity\Downloads\vol surface dynamics\data\processed\fengler_iv_surfaces\iv_surface_2025-06-13.npz'
# iv_csv = r'C:\Users\adity\Downloads\vol surface dynamics\data\processed\options_with_iv_junelater.csv'

# # --------------------------
# # Forecasted PCA values
# # --------------------------
# X_pca_forecast = np.array([
#     [-0.6507, -0.2848, -0.1328, 0.0653, 0.0382],
#     [-0.3877, -0.1317, -0.1113, 0.0464, 0.0280],
#     [-0.2084, -0.0837, -0.0938, 0.0360, 0.0208],
#     [-0.0862, -0.0687, -0.0794, 0.0303, 0.0156],
#     [-0.0029, -0.0640, -0.0677, 0.0271, 0.0119],
#     [0.0539, -0.0625, -0.0582, 0.0254, 0.0092],
#     [0.0927, -0.0620, -0.0504, 0.0244, 0.0073],
#     [0.1191, -0.0619, -0.0440, 0.0239, 0.0060],
#     [0.1371, -0.0618, -0.0388, 0.0236, 0.0050],
#     [0.1493, -0.0618, -0.0346, 0.0234, 0.0043],
# ])

# forecast_dates = [
#     '2025-06-20',
#     '2025-06-21',
#     '2025-06-22',
#     '2025-06-23',
#     '2025-06-24',
#     '2025-06-25',
#     '2025-06-26',
#     '2025-06-27',
#     '2025-06-28',
#     '2025-06-29',
# ]
# # --------------------------
# # Load PCA components
# # --------------------------
# data = np.load(pca_path)
# components = data['pca_model']  # shape: (5, 2500)

# # --------------------------
# # Load STRIKE, TTM grid from reference surface
# # --------------------------
# ref = np.load(ref_surface_path)
# STRIKE = ref['strike']  # shape: (50, 50)
# TTM = ref['ttm']        # shape: (50, 50)

# # --------------------------
# # Prompt user to pick date
# # --------------------------
# print("\n📅 Forecasted Dates:")
# for i, date in enumerate(forecast_dates):
#     print(f"{i + 1}: {date}")

# choice = int(input("\nEnter number to choose date: ")) - 1
# chosen_date = forecast_dates[choice]
# chosen_pca = X_pca_forecast[choice]

# # --------------------------
# # Reconstruct PCA Surface
# # --------------------------
# iv_flat = np.dot(chosen_pca, components)  # shape (2500,)
# iv_surface = iv_flat.reshape(50, 50)

# # --------------------------
# # Load Actual IV Data
# # --------------------------
# df_iv = pd.read_csv(iv_csv)
# df_iv['date'] = pd.to_datetime(df_iv['date'], dayfirst=True)
# df_iv['expiry'] = pd.to_datetime(df_iv['expiry'], dayfirst=True)
# df_iv['T'] = (df_iv['expiry'] - df_iv['date']).dt.days / 365

# chosen_date_dt = pd.to_datetime(chosen_date)
# df_plot = df_iv[df_iv['date'] == chosen_date_dt][['strike', 'T', 'IV']].dropna()

# if df_plot.empty:
#     print(f"⚠️ No actual IV data available for {chosen_date}")
#     actual_iv_valid = []
#     predicted_iv_valid = []
#     rmse = np.nan
# else:
#     # Interpolator from PCA surface
#     x_strikes = np.unique(STRIKE[0])
#     y_ttm = np.unique(TTM[:, 0])
#     iv_interp = RegularGridInterpolator((y_ttm, x_strikes), iv_surface, bounds_error=False, fill_value=np.nan)

#     query_points = df_plot[['T', 'strike']].values
#     out_of_bounds = 0
#     for pt in query_points:
#         if not (y_ttm.min() <= pt[0] <= y_ttm.max() and x_strikes.min() <= pt[1] <= x_strikes.max()):
#             out_of_bounds += 1
#     print(f"❗ Out-of-bounds actual IV points: {out_of_bounds} / {len(query_points)}")

#     predicted_iv = iv_interp(query_points)
#     actual_iv = df_plot['IV'].values

#     valid_mask = ~np.isnan(predicted_iv)
#     actual_iv_valid = actual_iv[valid_mask]
#     predicted_iv_valid = predicted_iv[valid_mask]
#     valid_points = query_points[valid_mask]

#     if len(actual_iv_valid) == 0:
#         print(f"⚠️ No matching strikes/TTMs inside PCA surface grid on {chosen_date}")
#         rmse = np.nan
#     else:
#         rmse = np.sqrt(mean_squared_error(actual_iv_valid, predicted_iv_valid))

# # --------------------------
# # Plot Combined Surface + Actual IVs
# # --------------------------
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Plot PCA-predicted surface
# surf = ax.plot_surface(STRIKE, TTM, iv_surface, cmap=cm.viridis, alpha=0.85, edgecolor='k')

# # Plot actual IVs as red dots
# if len(actual_iv_valid) > 0:
#     ax.scatter(valid_points[:, 1], valid_points[:, 0], actual_iv_valid, color='red', s=40, label='Actual IV')

# # Labels & Titles
# title = f"📅 Predicted IV Surface vs Actual IVs on {chosen_date}\nRMSE: {rmse:.4f}" if not np.isnan(rmse) else f"❌ No matching IV points inside grid"
# ax.set_title(title, fontsize=14)
# ax.set_xlabel("Strike")
# ax.set_ylabel("TTM (yrs)")
# ax.set_zlabel("Implied Volatility")
# ax.legend()
# plt.tight_layout()
# plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import mean_squared_error
from scipy.interpolate import RegularGridInterpolator

# --------------------------
# CONFIG PATHS
# --------------------------
pca_path = r'C:\Users\adity\Downloads\vol surface dynamics\data\processed\iv_pca_output.npz'
ref_surface_path = r'C:\Users\adity\Downloads\vol surface dynamics\data\processed\fengler_iv_surfaces\iv_surface_2025-06-13.npz'
iv_csv = r'C:\Users\adity\Downloads\vol surface dynamics\data\processed\options_with_iv_junelater.csv'

# --------------------------
# Forecasted PCA values (for June 16–20 only)
# --------------------------
X_pca_forecast = np.array([
    [-0.8650, -0.3132, -0.1444,  0.0770, 0.0449],  # 2025-06-16
    [-0.6507, -0.2848, -0.1328,  0.0653, 0.0382],  # 2025-06-17
    [-0.3877, -0.1317, -0.1113,  0.0464, 0.0280],  # 2025-06-18
    [-0.2084, -0.0837, -0.0938,  0.0360, 0.0208],  # 2025-06-19
    [-0.0862, -0.0687, -0.0794,  0.0303, 0.0156],  # 2025-06-20
])

forecast_dates = [
    '2025-06-16',
    '2025-06-17',
    '2025-06-18',
    '2025-06-19',
    '2025-06-20',
]

# --------------------------
# Load PCA components
# --------------------------
data = np.load(pca_path)
components = data['pca_model']  # shape: (5, 2500)

# --------------------------
# Load STRIKE, TTM grid from reference surface
# --------------------------
ref = np.load(ref_surface_path)
STRIKE = ref['strike']  # shape: (50, 50)
TTM = ref['ttm']        # shape: (50, 50)

# --------------------------
# Prompt user to pick date
# --------------------------
print("\n📅 Forecasted Dates:")
for i, date in enumerate(forecast_dates):
    print(f"{i + 1}: {date}")

choice = int(input("\nEnter number to choose date: ")) - 1
chosen_date = forecast_dates[choice]
chosen_pca = X_pca_forecast[choice]

# --------------------------
# Reconstruct PCA Surface
# --------------------------
iv_flat = np.dot(chosen_pca, components)  # shape (2500,)
iv_surface = iv_flat.reshape(50, 50)

# --------------------------
# Load Actual IV Data
# --------------------------
df_iv = pd.read_csv(iv_csv)
df_iv['date'] = pd.to_datetime(df_iv['date'], dayfirst=True)
df_iv['expiry'] = pd.to_datetime(df_iv['expiry'], dayfirst=True)
df_iv['T'] = (df_iv['expiry'] - df_iv['date']).dt.days / 365

chosen_date_dt = pd.to_datetime(chosen_date)
df_plot = df_iv[df_iv['date'] == chosen_date_dt][['strike', 'T', 'IV']].dropna()

if df_plot.empty:
    print(f"⚠️ No actual IV data available for {chosen_date}")
    actual_iv_valid = []
    predicted_iv_valid = []
    rmse = np.nan
else:
    # Interpolator from PCA surface
    x_strikes = np.unique(STRIKE[0])
    y_ttm = np.unique(TTM[:, 0])
    iv_interp = RegularGridInterpolator((y_ttm, x_strikes), iv_surface, bounds_error=False, fill_value=np.nan)

    query_points = df_plot[['T', 'strike']].values
    out_of_bounds = 0
    for pt in query_points:
        if not (y_ttm.min() <= pt[0] <= y_ttm.max() and x_strikes.min() <= pt[1] <= x_strikes.max()):
            out_of_bounds += 1
    print(f"❗ Out-of-bounds actual IV points: {out_of_bounds} / {len(query_points)}")

    predicted_iv = iv_interp(query_points)
    actual_iv = df_plot['IV'].values

    valid_mask = ~np.isnan(predicted_iv)
    actual_iv_valid = actual_iv[valid_mask]
    predicted_iv_valid = predicted_iv[valid_mask]
    valid_points = query_points[valid_mask]

    if len(actual_iv_valid) == 0:
        print(f"⚠️ No matching strikes/TTMs inside PCA surface grid on {chosen_date}")
        rmse = np.nan
    else:
        rmse = np.sqrt(mean_squared_error(actual_iv_valid, predicted_iv_valid))

# --------------------------
# Plot Combined Surface + Actual IVs
# --------------------------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot PCA-predicted surface
surf = ax.plot_surface(STRIKE, TTM, iv_surface, cmap=cm.viridis, alpha=0.85, edgecolor='k')

# Plot actual IVs as red dots
if len(actual_iv_valid) > 0:
    ax.scatter(valid_points[:, 1], valid_points[:, 0], actual_iv_valid, color='red', s=40, label='Actual IV')

# Labels & Titles
title = f"📅 Predicted IV Surface vs Actual IVs on {chosen_date}\nRMSE: {rmse:.4f}" if not np.isnan(rmse) else f"❌ No matching IV points inside grid"
ax.set_title(title, fontsize=14)
ax.set_xlabel("Strike")
ax.set_ylabel("TTM (yrs)")
ax.set_zlabel("Implied Volatility")
ax.legend()
plt.tight_layout()
plt.show()
