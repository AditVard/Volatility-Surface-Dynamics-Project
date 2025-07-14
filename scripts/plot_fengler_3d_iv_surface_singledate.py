#TO PLOT IV LIKE PROPERLY IN 3D SURFACE USING FENGLER SMOOTHENING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import SmoothBivariateSpline
from matplotlib import cm

# ------------------------
# Load Data
# ------------------------
df = pd.read_csv('../../data/processed/greeks_iv_from_column.csv')
df.dropna(subset=['strike', 'spot', 'iv', 'expiry', 'date'], inplace=True)

# Parse dates
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
df['expiry'] = pd.to_datetime(df['expiry'], format='%d-%b-%Y', errors='coerce')
df.dropna(subset=['date', 'expiry'], inplace=True)

# Time to maturity in years
df['ttm'] = (df['expiry'] - df['date']).dt.days / 365
df = df[df['ttm'] > 0]

# Log-moneyness
df['log_moneyness'] = np.log(df['strike'] / df['spot'])

# ------------------------
# Select a Date
# ------------------------
available_dates = sorted(df['date'].unique())
print("📅 Available dates:")
for i, d in enumerate(available_dates):
    print(f"{i+1}: {d.strftime('%d-%b-%Y')}")

choice = int(input("Enter the number of the date to plot IV surface for: "))
chosen_date = available_dates[choice - 1]
print(f"\n🔍 Plotting IV surface for {chosen_date.strftime('%d-%b-%Y')}")

# Filter for chosen date
df_day = df[df['date'] == chosen_date].copy()
x = df_day['log_moneyness'].values
y = df_day['ttm'].values
z = df_day['iv'].values

# ------------------------
# Fit Fengler Bivariate Spline
# ------------------------
spline = SmoothBivariateSpline(x, y, z, kx=3, ky=3, s=0.001)

# Dense grid for surface
x_dense = np.linspace(min(x), max(x), 100)
y_dense = np.linspace(min(y), max(y), 100)
LOGM, TTM = np.meshgrid(x_dense, y_dense)
Z = spline.ev(LOGM.ravel(), TTM.ravel()).reshape(LOGM.shape)

# ------------------------
# Plot 3D Surface
# ------------------------
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(LOGM, TTM, Z, cmap=cm.viridis, edgecolor='k', alpha=0.85)

ax.set_title(f"Fengler IV Surface\n{chosen_date.strftime('%d-%b-%Y')}", fontsize=14)
ax.set_xlabel("Log-Moneyness")
ax.set_ylabel("TTM (years)")
ax.set_zlabel("Implied Volatility")

fig.colorbar(surf, shrink=0.5, aspect=8)
plt.tight_layout()
plt.show()
