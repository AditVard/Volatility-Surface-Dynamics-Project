import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings("ignore")

# ----------------------------
# Config
# ----------------------------
surf_folder = r"C:\Users\adity\Downloads\vol surface dynamics\data\processed\fengler_iv_surfaces"
files = sorted([f for f in os.listdir(surf_folder) if f.endswith('.npz')])

# ----------------------------
# Load IV Surfaces & Flatten
# ----------------------------
surfaces = []
dates = []

for fname in files:
    path = os.path.join(surf_folder, fname)
    data = np.load(path)
    surfaces.append(data['iv'].reshape(-1))  # flatten 50x50 = 2500
    dates.append(fname.replace('iv_surface_', '').replace('.npz', ''))

X = np.vstack(surfaces)  # shape = (n_days, 2500)

# ----------------------------
# PCA on IV Surfaces
# ----------------------------
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)

# ----------------------------
# Convert PC1 to Time Series
# ----------------------------
pc1_series = pd.Series(X_pca[:, 0], index=pd.to_datetime(dates))
pc1_series = pc1_series.asfreq('D')           # daily frequency
pc1_series = pc1_series.interpolate()         # fill missing dates

# ----------------------------
# ADF Test for Stationarity
# ----------------------------
adf_stat, pval, *_ = adfuller(pc1_series)
d = 0 if pval < 0.05 else 1
series_to_model = pc1_series if d == 0 else pc1_series.diff().dropna()

# ----------------------------
# ARIMA Fit & Forecast
# ----------------------------
model = ARIMA(series_to_model, order=(1, d, 1))
fit = model.fit()
forecast = fit.forecast(steps=10)

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(12, 6))
pc1_series.plot(label='Historical PC1')
forecast.index = pd.date_range(start=pc1_series.index[-1] + pd.Timedelta(days=1), periods=10)
forecast.plot(label='Forecast', linestyle='--', marker='o')
plt.title('ARIMA Forecast for PC1')
plt.xlabel('Date')
plt.ylabel('PC1 Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
