import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib

# ------------------------ #
# Load IV surfaces
# ------------------------ #
surf_folder = r'C:\Users\adity\Downloads\vol surface dynamics\data\processed\fengler_iv_surfaces'
files = sorted([f for f in os.listdir(surf_folder) if f.endswith('.npz')])

dates = []
surfaces = []

for f in files:
    path = os.path.join(surf_folder, f)
    data = np.load(path)
    iv_surface = data['iv']  # shape (50, 50)
    
    if iv_surface.shape == (50, 50):  # ensure expected shape
        surfaces.append(iv_surface.flatten())  # shape (2500,)
        date_str = f.replace('iv_surface_', '').replace('.npz', '')
        dates.append(date_str)

X = np.array(surfaces)  # shape: (num_days, 2500)

# ------------------------ #
# Run PCA
# ------------------------ #
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)

# ------------------------ #
# Plot PC1 vs PC2 vs PC3
# ------------------------ #
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=range(len(dates)), cmap='viridis', s=50)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('PCA of IV Surfaces (First 3 PCs)')

plt.tight_layout()
plt.show()
