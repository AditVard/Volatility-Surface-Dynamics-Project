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

# -------------------
# Display all available dates
# -------------------
print("📅 Available IV Surfaces:\n")
date_map = {}
for i, f in enumerate(files):
    date_str = f.replace('iv_surface_', '').replace('.npz', '')
    print(f"{i+1:>2}: {date_str}")
    date_map[i+1] = f

# -------------------
# User input for which dates to animate
# -------------------
choice = input("\n🔎 Enter the index (or comma-separated indices) of the date(s) you want to animate: ")
selected_indices = [int(i.strip()) for i in choice.split(",") if i.strip().isdigit()]

selected_files = [date_map[i] for i in selected_indices if i in date_map]
if not selected_files:
    print("❌ No valid files selected.")
    exit()

# -------------------
# Load selected surfaces
# -------------------
surfaces = []
dates = []

for fname in selected_files:
    path = os.path.join(surf_folder, fname)
    data = np.load(path)
    surfaces.append(data['iv'])  # (50, 50)
    dates.append(fname.replace('iv_surface_', '').replace('.npz', ''))

surfaces = np.stack(surfaces, axis=0)
T = surfaces.shape[0]

# ✅ Use correct grid directly
sample = np.load(os.path.join(surf_folder, selected_files[0]))
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

# 🎞️ Animate slower (e.g., 1500ms per frame)
ani = animation.FuncAnimation(fig, update, frames=T, interval=1500)

plt.show()
