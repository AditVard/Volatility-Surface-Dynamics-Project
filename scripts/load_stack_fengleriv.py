import numpy as np
import os

# Folder path
surf_folder = 'C:\\Users\\adity\\Downloads\\vol surface dynamics\\data\\processed\\fengler_iv_surfaces'

# List and sort files
files = sorted(os.listdir(surf_folder))
print("📂 Files found in folder:", files)

surfaces = []
dates = []

for fname in files:
    if fname.endswith('.npz'):
        path = os.path.join(surf_folder, fname)
        try:
            data = np.load(path)
            Z = data['iv']  # ✅ this is your implied volatility surface
            surfaces.append(Z)
            dates.append(fname.replace('.npz', ''))
            print(f"✅ Loaded {fname} → shape {Z.shape}")
        except Exception as e:
            print(f"❌ Failed to load {fname} →", e)

# Final check
if not surfaces:
    raise ValueError("❌ No valid surfaces loaded. Please check your .npz contents.")

# Stack all surfaces into 3D array (time, strike, ttm)
surfaces = np.stack(surfaces, axis=0)
print("✅ Final stacked array shape:", surfaces.shape)
