import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
import os

# -------------------
# Config
# -------------------
surf_folder = r"C:\Users\adity\Downloads\vol surface dynamics\data\processed\fengler_iv_surfaces"
files = sorted([f for f in os.listdir(surf_folder) if f.endswith('.npz')])
files = files[:34]  # Adjust this if needed

# -------------------
# Load surfaces
# -------------------
surfaces = []
dates = []

for fname in files:
    path = os.path.join(surf_folder, fname)
    data = np.load(path)
    surfaces.append(data['iv'])  # (50, 50)
    dates.append(fname.replace('iv_surface_', '').replace('.npz', ''))

surfaces = np.stack(surfaces, axis=0)
T = surfaces.shape[0]

# ✅ Use correct grid directly
sample = np.load(os.path.join(surf_folder, files[0]))
STRIKE = sample['strike']
TTM = sample['ttm']

# -------------------
# Animate
# -------------------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    ax.clear()
    ax.plot_surface(STRIKE, TTM, surfaces[frame], cmap=cm.viridis, edgecolor='k', alpha=0.85)
    ax.set_title(f"📅 IV Surface on {dates[frame]}", fontsize=15)
    ax.set_xlabel("Strike")
    ax.set_ylabel("TTM (yrs)")
    ax.set_zlabel("Implied Volatility")
    ax.set_zlim(0, np.nanmax(surfaces) * 1.1)
    plt.tight_layout()
    plt.pause(0.01)  # Helps smooth redraw in some environments

# 🎞️ Animate with slower speed (1500ms = 1.5s per frame)
ani = animation.FuncAnimation(fig, update, frames=T, interval=1500)

plt.show()
