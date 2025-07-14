import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime

# ---- Load Data ----
df = pd.read_csv(r'C:\Users\adity\Downloads\vol surface dynamics\data\processed\greeks_iv_from_column.csv')

# ---- Preprocess ----
df.dropna(subset=['strike', 'spot', 'iv', 'expiry'], inplace=True)
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
df['expiry'] = pd.to_datetime(df['expiry'], format='%d-%b-%Y')
df['ttm'] = (df['expiry'] - df['date']).dt.days
df = df[df['ttm'] > 0]
df['log_moneyness'] = np.log(df['strike'] / df['spot'])

# ---- Loop over all dates ----
for chosen_date in sorted(df['date'].unique()):
    df_day = df[df['date'] == chosen_date]
    
    pivot = df_day.pivot_table(index='ttm', columns='log_moneyness', values='iv')
    pivot = pivot.dropna(axis=1, thresh=2)
    pivot = pivot.dropna(axis=0, thresh=2)

    if pivot.empty or pivot.shape[1] < 2 or pivot.shape[0] < 2:
        continue  # Skip if not enough data

    X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)
    Z = pivot.values

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, edgecolor='k', alpha=0.85)

    ax.set_xlabel('Log-Moneyness (ln(K/S))')
    ax.set_ylabel('Time to Expiry (days)')
    ax.set_zlabel('Implied Volatility')
    ax.set_title(f"IV Surface on {chosen_date.strftime('%d-%b-%Y')}")
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.show()
