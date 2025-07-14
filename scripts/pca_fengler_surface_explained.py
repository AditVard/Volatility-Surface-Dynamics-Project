import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# --------------------------
# Load all .npz surfaces
# --------------------------
surf_folder = r'C:\Users\adity\Downloads\vol surface dynamics\data\processed\fengler_iv_surfaces'
files = sorted([f for f in os.listdir(surf_folder) if f.endswith('.npz')])
dates = []
surfaces = []

for fname in files:
    path = os.path.join(surf_folder, fname)
    data = np.load(path)
    surfaces.append(data['iv'].reshape(-1))  # flatten 50x50 → 2500
    dates.append(fname.replace('iv_surface_', '').replace('.npz', ''))

X = np.vstack(surfaces)  # shape (n_days, 2500)

# --------------------------
# Apply PCA
# --------------------------
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_pca.shape}")


# Convert dates to np.datetime64
dates_np = np.array(dates, dtype='datetime64[D]')

# Save PCA output
np.savez_compressed(
    "iv_pca_output.npz",
    pca_components=X_pca,       # (n_days, 5)
    dates=dates_np,             # (n_days,)
    pca_model=pca.components_,  # (5, 2500)
    explained_variance=pca.explained_variance_ratio_
)
print("✅ Saved PCA output to 'iv_pca_output.npz'")

# --------------------------
# Plot explained variance
# --------------------------
plt.figure(figsize=(6,4))
plt.plot(np.cumsum(pca.explained_variance_ratio_) * 100, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance (%)')
plt.title('Explained Variance by PCA Components')
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------
# Plot first 3 PCA time series
# --------------------------
plt.figure(figsize=(10,6))
for i in range(3):
    plt.plot(dates, X_pca[:, i], label=f'PC {i+1}')
plt.xticks(rotation=45)
plt.xlabel("Date")
plt.ylabel("PCA Value")
plt.title("Top 3 PCA Factors over Time")
plt.legend()
plt.tight_layout()
plt.show()
