import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import os

# ========== CONFIG ========== #
forecast_pca_path = r'C:\Users\adity\Downloads\vol surface dynamics\data\processed\forecasted_pca_vectors.npz'
pca_components_path = r'C:\Users\adity\Downloads\vol surface dynamics\data\processed\iv_pca_output.npz'
ref_surface_path = r'C:\Users\adity\Downloads\vol surface dynamics\data\processed\fengler_iv_surfaces\iv_surface_2025-06-13.npz'
output_dir = r'C:\Users\adity\Downloads\vol surface dynamics\data\processed\forecasted_surfaces'
os.makedirs(output_dir, exist_ok=True)

# ========== LOAD FORECASTED PCA ========== #
print(f"📁 Loading forecasted PCA vectors from: {forecast_pca_path}")
forecast_data = np.load(forecast_pca_path, allow_pickle=True)
X_pca_forecast = forecast_data['X_pca_forecast']             # shape (10, 5)
forecast_dates = forecast_data['forecast_dates'].tolist()    # len = 10
print(f"✅ X_pca_forecast shape: {X_pca_forecast.shape}")
print(f"✅ forecast_dates: {forecast_dates}")

# ========== LOAD PCA COMPONENTS ========== #
print(f"\n📁 Loading PCA components from: {pca_components_path}")
pca_data = np.load(pca_components_path, allow_pickle=True)
components = pca_data['pca_model']                            # shape (5, 2500)
print(f"✅ PCA basis shape: {components.shape}")

# ========== LOAD REFERENCE SURFACE GRID ========== #
print(f"\n📁 Loading reference surface from: {ref_surface_path}")
ref = np.load(ref_surface_path, allow_pickle=True)
STRIKE = ref['strike']       # shape (50, 50)
TTM = ref['ttm']             # shape (50, 50)
print(f"✅ STRIKE shape: {STRIKE.shape}, TTM shape: {TTM.shape}")
grid_shape = STRIKE.shape    # → (50, 50)

# ========== RECONSTRUCT & SAVE FORECASTED SURFACES ========== #
for i, date in enumerate(forecast_dates):
    pca_vec = X_pca_forecast[i]                     # shape: (5,)
    iv_flat = np.dot(pca_vec, components)           # → shape: (2500,)
    
    try:
        iv_surface = iv_flat.reshape(grid_shape)    # → shape: (50, 50)
    except Exception as e:
        print(f"❌ Frame {i}: Error reshaping to {grid_shape} → {e}")
        continue

    # Plot and save
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(STRIKE, TTM, iv_surface, cmap=cm.viridis, edgecolor='none')
    ax.set_title(f'📅 Forecasted IV Surface: {date}')
    ax.set_xlabel("Strike")
    ax.set_ylabel("TTM (yrs)")
    ax.set_zlabel("Implied Volatility")

    save_path = os.path.join(output_dir, f'iv_surface_forecasted_{date}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved: {save_path}")
