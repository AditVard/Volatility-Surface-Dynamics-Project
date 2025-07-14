# # # # # import numpy as np
# # # # # import os
# # # # # from sklearn.decomposition import PCA
# # # # # import matplotlib.pyplot as plt
# # # # # import pandas as pd
# # # # # from scipy.interpolate import griddata

# # # # # # ----------------------------
# # # # # # CONFIG
# # # # # # ----------------------------
# # # # # surf_folder = r'C:\Users\adity\Downloads\vol surface dynamics\data\processed\dupire_surfaces'
# # # # # output_file = r'C:\Users\adity\Downloads\vol surface dynamics\data\processed\dupire_pca_output.npz'

# # # # # # ----------------------------
# # # # # # LOAD AND FLATTEN SURFACES
# # # # # # ----------------------------
# # # # # files = sorted([f for f in os.listdir(surf_folder) if f.endswith('.npz')])
# # # # # surfaces = []
# # # # # dates = []

# # # # # for fname in files:
# # # # #     path = os.path.join(surf_folder, fname)
# # # # #     data = np.load(path)
    
# # # # #     if 'local_vol' not in data:
# # # # #         print(f"⚠️ Skipping {fname} — no 'local_vol' key")
# # # # #         continue

# # # # #     lv = data['local_vol']

# # # # #     if lv.shape != (50, 50):
# # # # #         print(f"⚠️ Skipping {fname} — unexpected shape {lv.shape}")
# # # # #         continue

# # # # #     if np.isnan(lv).any():
# # # # #         print(f"⚠️ {fname} contains {np.isnan(lv).sum()} NaNs — filling")

# # # # #         # Interpolate
# # # # #         x, y = np.meshgrid(np.arange(lv.shape[1]), np.arange(lv.shape[0]))
# # # # #         mask = ~np.isnan(lv)
# # # # #         filled_lv = griddata(
# # # # #             (x[mask], y[mask]),
# # # # #             lv[mask],
# # # # #             (x, y),
# # # # #             method='linear'
# # # # #         )

# # # # #         # Fallback: fill remaining NaNs with 0 (in case interpolation doesn't fully work)
# # # # #         if np.isnan(filled_lv).any():
# # # # #             filled_lv = np.nan_to_num(filled_lv, nan=0.0)

# # # # #         lv = filled_lv  # replace with filled version

# # # # #     surfaces.append(lv.reshape(-1))  # Flatten to 2500
# # # # #     date_str = fname.replace('local_vol_surface_', '').replace('.npz', '')
# # # # #     dates.append(date_str)

# # # # # # Final matrix
# # # # # if len(surfaces) == 0:
# # # # #     raise RuntimeError("❌ No surfaces loaded!")

# # # # # X = np.vstack(surfaces)
# # # # # dates_np = np.array(dates, dtype='datetime64[D]')
# # # # # print(f"✅ Loaded {X.shape[0]} surfaces, each of size {X.shape[1]}")

# # # # # # ----------------------------
# # # # # # PCA
# # # # # # ----------------------------
# # # # # n_components = min(5, X.shape[0])
# # # # # pca = PCA(n_components=n_components)
# # # # # X_pca = pca.fit_transform(X)

# # # # # print("✅ PCA complete")
# # # # # print("Explained variance ratio:", pca.explained_variance_ratio_)

# # # # # # ----------------------------
# # # # # # SAVE
# # # # # # ----------------------------
# # # # # np.savez_compressed(
# # # # #     output_file,
# # # # #     pca_components=X_pca,
# # # # #     dates=dates_np,
# # # # #     pca_model=pca.components_,
# # # # #     explained_variance=pca.explained_variance_ratio_
# # # # # )
# # # # # print(f"✅ Saved PCA output to: {output_file}")

# # # # # # ----------------------------
# # # # # # Plot variance
# # # # # # ----------------------------
# # # # # plt.figure(figsize=(6, 4))
# # # # # plt.plot(np.cumsum(pca.explained_variance_ratio_) * 100, marker='o')
# # # # # plt.title("Explained Variance (Dupire PCA)")
# # # # # plt.xlabel("Number of Components")
# # # # # plt.ylabel("Cumulative Variance (%)")
# # # # # plt.grid(True)
# # # # # plt.tight_layout()
# # # # # plt.show()
# # # # import numpy as np
# # # # import os
# # # # from sklearn.decomposition import PCA
# # # # import matplotlib.pyplot as plt
# # # # import pandas as pd
# # # # from scipy.interpolate import griddata

# # # # # ----------------------------
# # # # # CONFIG
# # # # # ----------------------------
# # # # surf_folder = r'C:\Users\adity\Downloads\vol surface dynamics\data\processed\dupire_surfaces'
# # # # output_file = r'C:\Users\adity\Downloads\vol surface dynamics\data\processed\dupire_pca_output.npz'

# # # # # ----------------------------
# # # # # LOAD AND FLATTEN SURFACES
# # # # # ----------------------------
# # # # files = sorted([f for f in os.listdir(surf_folder) if f.endswith('.npz')])
# # # # surfaces = []
# # # # dates = []

# # # # for fname in files:
# # # #     path = os.path.join(surf_folder, fname)
# # # #     data = np.load(path)
    
# # # #     if 'local_vol' not in data:
# # # #         print(f"⚠️ Skipping {fname} — no 'local_vol' key")
# # # #         continue

# # # #     lv = data['local_vol']

# # # #     if lv.shape != (50, 50):
# # # #         print(f"⚠️ Skipping {fname} — unexpected shape {lv.shape}")
# # # #         continue

# # # #     if np.isnan(lv).any():
# # # #         print(f"⚠️ {fname} contains {np.isnan(lv).sum()} NaNs — interpolating")

# # # #         x, y = np.meshgrid(np.arange(lv.shape[1]), np.arange(lv.shape[0]))
# # # #         mask = ~np.isnan(lv)

# # # #         filled_lv = griddata(
# # # #             (x[mask], y[mask]),
# # # #             lv[mask],
# # # #             (x, y),
# # # #             method='linear'
# # # #         )

# # # #         if np.isnan(filled_lv).any():
# # # #             mean_val = np.nanmean(filled_lv)
# # # #             filled_lv = np.nan_to_num(filled_lv, nan=mean_val)

# # # #         lv = filled_lv

# # # #     surfaces.append(lv.reshape(-1))
# # # #     date_str = fname.replace('local_vol_surface_', '').replace('.npz', '')
# # # #     dates.append(date_str)

# # # # # Final matrix
# # # # if len(surfaces) == 0:
# # # #     raise RuntimeError("❌ No surfaces loaded!")

# # # # X = np.vstack(surfaces)  # shape: (n_days, 2500)
# # # # dates_np = np.array(dates, dtype='datetime64[D]')
# # # # print(f"✅ Loaded {X.shape[0]} surfaces, each of size {X.shape[1]}")

# # # # # ----------------------------
# # # # # PCA
# # # # # ----------------------------
# # # # n_components = min(5, X.shape[0])  # prevent crash if < 5 samples
# # # # pca = PCA(n_components=n_components)
# # # # X_pca = pca.fit_transform(X)

# # # # print("✅ PCA complete")
# # # # print("Explained variance ratio:", pca.explained_variance_ratio_)

# # # # # ----------------------------
# # # # # SAVE
# # # # # ----------------------------
# # # # np.savez_compressed(
# # # #     output_file,
# # # #     pca_components=X_pca,
# # # #     dates=dates_np,
# # # #     pca_model=pca.components_,
# # # #     explained_variance=pca.explained_variance_ratio_
# # # # )
# # # # print(f"✅ Saved PCA output to: {output_file}")

# # # # # ----------------------------
# # # # # Plot variance
# # # # # ----------------------------
# # # # plt.figure(figsize=(6, 4))
# # # # plt.plot(np.cumsum(pca.explained_variance_ratio_) * 100, marker='o')
# # # # plt.title("Explained Variance (Dupire PCA)")
# # # # plt.xlabel("Number of Components")
# # # # plt.ylabel("Cumulative Variance (%)")
# # # # plt.grid(True)
# # # # plt.tight_layout()
# # # # plt.show()
# # # import numpy as np
# # # import os
# # # from sklearn.decomposition import PCA
# # # import matplotlib.pyplot as plt
# # # import pandas as pd
# # # import re

# # # # ----------------------------
# # # # CONFIG
# # # # ----------------------------
# # # surf_folder = r'C:\Users\adity\Downloads\vol surface dynamics\data\processed\dupire_surfaces'
# # # output_file = r'C:\Users\adity\Downloads\vol surface dynamics\data\processed\dupire_pca_output.npz'

# # # # ----------------------------
# # # # LOAD AND FLATTEN SURFACES
# # # # ----------------------------
# # # pattern = re.compile(r'local_vol_surface_(\d{4}-\d{2}-\d{2})\.npz')
# # # files = sorted([f for f in os.listdir(surf_folder) if pattern.match(f)])

# # # surfaces = []
# # # dates = []

# # # for fname in files:
# # #     match = pattern.match(fname)
# # #     if not match:
# # #         print(f"⚠️ Skipping invalid filename: {fname}")
# # #         continue

# # #     date_str = match.group(1)
# # #     date = np.datetime64(date_str)

# # #     path = os.path.join(surf_folder, fname)
# # #     data = np.load(path)

# # #     if 'local_vol' not in data:
# # #         print(f"⚠️ Skipping {fname} — missing 'local_vol'")
# # #         continue

# # #     lv = data['local_vol']
# # #     if lv.shape != (50, 50):
# # #         print(f"⚠️ Skipping {fname} — unexpected shape {lv.shape}")
# # #         continue

# # #     if np.isnan(lv).any():
# # #         print(f"⚠️ Warning: {fname} still has {np.isnan(lv).sum()} NaNs — they were supposed to be filled!")

# # #     surfaces.append(lv.reshape(-1))
# # #     dates.append(date)

# # # # Final matrix
# # # if len(surfaces) == 0:
# # #     raise RuntimeError("❌ No valid surfaces found!")

# # # X = np.vstack(surfaces)
# # # dates_np = np.array(dates, dtype='datetime64[D]')
# # # print(f"✅ Loaded {X.shape[0]} surfaces, each of size {X.shape[1]}")

# # # # ----------------------------
# # # # PCA
# # # # ----------------------------
# # # n_components = min(5, X.shape[0])
# # # pca = PCA(n_components=n_components)
# # # X_pca = pca.fit_transform(X)

# # # print("✅ PCA complete")
# # # print("Explained variance ratio:", pca.explained_variance_ratio_)

# # # # ----------------------------
# # # # SAVE
# # # # ----------------------------
# # # np.savez_compressed(
# # #     output_file,
# # #     pca_components=X_pca,
# # #     dates=dates_np,
# # #     pca_model=pca.components_,
# # #     explained_variance=pca.explained_variance_ratio_
# # # )
# # # print(f"✅ Saved PCA output to: {output_file}")

# # # # ----------------------------
# # # # Plot variance
# # # # ----------------------------
# # # plt.figure(figsize=(6, 4))
# # # plt.plot(np.cumsum(pca.explained_variance_ratio_) * 100, marker='o')
# # # plt.title("Explained Variance (Dupire PCA)")
# # # plt.xlabel("Number of Components")
# # # plt.ylabel("Cumulative Variance (%)")
# # # plt.grid(True)
# # # plt.tight_layout()
# # # plt.show()
# # # # ----------------------------
# # # # Plot PC1, PC2 over Time
# # # # ----------------------------
# # # plt.figure(figsize=(10, 6))

# # # for i in range(n_components):
# # #     plt.plot(dates_np, X_pca[:, i], label=f'PC{i+1}')

# # # plt.title("PCA Component Scores over Time")
# # # plt.xlabel("Date")
# # # plt.ylabel("Component Value")
# # # plt.legend()
# # # plt.grid(True)
# # # plt.tight_layout()
# # # plt.xticks(rotation=45)
# # # plt.show()
# # # spike_threshold = 40  # or whatever value you consider 'large'

# # # for i, val in enumerate(X_pca[:, 0]):
# # #     if abs(val) > spike_threshold:
# # #         print(f"⚡ PC1 spike on {dates_np[i]}: value = {val:.2f}")
# # # spike_index = np.argmax(np.abs(X_pca[:, 0]))  # or PC2
# # # spike_date = dates_np[spike_index]
# # # print(f"Largest spike in PC1 is on: {spike_date}")

# # # # Load the corresponding surface and plot
# # # spike_fname = f"local_vol_surface_{spike_date}.npz"
# # # data = np.load(os.path.join(surf_folder, spike_fname))
# # # plt.imshow(data['local_vol'], cmap='viridis', aspect='auto')
# # # plt.title(f"Dupire Local Vol Surface on {spike_date}")
# # # plt.colorbar()
# # # plt.tight_layout()
# # # plt.show()
# # import numpy as np
# # import os
# # from sklearn.decomposition import PCA
# # import matplotlib.pyplot as plt
# # import pandas as pd
# # import re

# # # ----------------------------
# # # CONFIG
# # # ----------------------------
# # surf_folder = r'C:\Users\adity\Downloads\vol surface dynamics\data\processed\dupire_surfaces'
# # output_file = r'C:\Users\adity\Downloads\vol surface dynamics\data\processed\dupire_pca_output_cleaned.npz'
# # spike_threshold = 40  # Customize threshold for spike detection

# # # ----------------------------
# # # LOAD AND FLATTEN SURFACES
# # # ----------------------------
# # pattern = re.compile(r'local_vol_surface_(\d{4}-\d{2}-\d{2})\.npz')
# # files = sorted([f for f in os.listdir(surf_folder) if pattern.match(f)])

# # surfaces = []
# # dates = []

# # for fname in files:
# #     match = pattern.match(fname)
# #     if not match:
# #         print(f"⚠️ Skipping invalid filename: {fname}")
# #         continue

# #     date_str = match.group(1)
# #     date = np.datetime64(date_str)

# #     path = os.path.join(surf_folder, fname)
# #     data = np.load(path)

# #     if 'local_vol' not in data:
# #         print(f"⚠️ Skipping {fname} — missing 'local_vol'")
# #         continue

# #     lv = data['local_vol']
# #     if lv.shape != (50, 50):
# #         print(f"⚠️ Skipping {fname} — unexpected shape {lv.shape}")
# #         continue

# #     if np.isnan(lv).any():
# #         print(f"⚠️ Warning: {fname} still has {np.isnan(lv).sum()} NaNs — should be filled already")

# #     surfaces.append(lv.reshape(-1))
# #     dates.append(date)

# # # Final matrix
# # if len(surfaces) == 0:
# #     raise RuntimeError("❌ No valid surfaces found!")

# # X_all = np.vstack(surfaces)
# # dates_all = np.array(dates, dtype='datetime64[D]')
# # print(f"✅ Loaded {X_all.shape[0]} surfaces, each of size {X_all.shape[1]}")

# # # ----------------------------
# # # TEMP PCA to detect spikes
# # # ----------------------------
# # pca_temp = PCA(n_components=5)
# # X_temp_pca = pca_temp.fit_transform(X_all)

# # # Detect spike dates in PC1
# # spike_indices = np.where(np.abs(X_temp_pca[:, 0]) > spike_threshold)[0]
# # spike_dates = dates_all[spike_indices]

# # print("\n⚠️ Spike Dates Detected in PC1:")
# # for idx in spike_indices:
# #     print(f"   {dates_all[idx]}: PC1 = {X_temp_pca[idx, 0]:.2f}")

# # # ----------------------------
# # # Remove spikes
# # # ----------------------------
# # mask = np.ones(len(X_all), dtype=bool)
# # mask[spike_indices] = False

# # X_clean = X_all[mask]
# # dates_clean = dates_all[mask]

# # # ----------------------------
# # # PCA on cleaned data
# # # ----------------------------
# # n_components = min(5, X_clean.shape[0])
# # pca = PCA(n_components=n_components)
# # X_pca = pca.fit_transform(X_clean)

# # print("\n✅ PCA complete on cleaned data")
# # print("Explained variance ratio:", pca.explained_variance_ratio_)

# # # ----------------------------
# # # SAVE
# # # ----------------------------
# # np.savez_compressed(
# #     output_file,
# #     pca_components=X_pca,
# #     dates=dates_clean,
# #     pca_model=pca.components_,
# #     explained_variance=pca.explained_variance_ratio_
# # )
# # print(f"✅ Saved cleaned PCA output to: {output_file}")

# # # ----------------------------
# # # Plot variance
# # # ----------------------------
# # plt.figure(figsize=(6, 4))
# # plt.plot(np.cumsum(pca.explained_variance_ratio_) * 100, marker='o')
# # plt.title("Explained Variance (Dupire PCA - Cleaned)")
# # plt.xlabel("Number of Components")
# # plt.ylabel("Cumulative Variance (%)")
# # plt.grid(True)
# # plt.tight_layout()
# # plt.show()

# # # ----------------------------
# # # Plot PC1–PC5 over Time (Cleaned)
# # # ----------------------------
# # plt.figure(figsize=(10, 6))
# # for i in range(n_components):
# #     plt.plot(dates_clean, X_pca[:, i], label=f'PC{i+1}')
# # plt.title("PCA Component Scores (Cleaned) over Time")
# # plt.xlabel("Date")
# # plt.ylabel("Component Value")
# # plt.legend()
# # plt.grid(True)
# # plt.tight_layout()
# # plt.xticks(rotation=45)
# # plt.show()

# # # ----------------------------
# # # Plot spike surface(s)
# # # ----------------------------
# # for spike_date in spike_dates:
# #     spike_fname = f"local_vol_surface_{str(spike_date)}.npz"
# #     spike_path = os.path.join(surf_folder, spike_fname)

# #     if not os.path.exists(spike_path):
# #         print(f"⚠️ File not found: {spike_fname}")
# #         continue

# #     data = np.load(spike_path)
# #     surface = data['local_vol']

# #     fig = plt.figure(figsize=(10, 7))
# #     ax = fig.add_subplot(111, projection='3d')
# #     x = np.linspace(-0.01, 0.01, 50)  # log-moneyness approx
# #     y = np.linspace(0.01, 0.06, 50)   # TTM approx
# #     Xg, Yg = np.meshgrid(x, y)

# #     ax.plot_surface(Xg, Yg, surface, cmap='plasma')
# #     ax.set_title(f"Dupire Local Vol Surface on {spike_date}")
# #     ax.set_xlabel("Log-Moneyness")
# #     ax.set_ylabel("Time to Expiry (years)")
# #     ax.set_zlabel("Local Volatility")
# #     plt.tight_layout()
# #     plt.show()

# # # ----------------------------
# # # Identify Max Spikes in Each PC
# # # ----------------------------

# # top_k = 3  # top N spikes per component

# # for i in range(n_components):
# #     abs_vals = np.abs(X_pca[:, i])
# #     top_indices = np.argsort(abs_vals)[-top_k:][::-1]  # largest first
# #     print(f"\n📈 Top {top_k} spikes in PC{i+1}:")
# #     for idx in top_indices:
# #         print(f"   {dates_clean[idx]} — PC{i+1} = {X_pca[idx, i]:.2f}")
# import numpy as np
# import os
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import re

# # ----------------------------
# # CONFIG
# # ----------------------------
# surf_folder = r'C:\Users\adity\Downloads\vol surface dynamics\data\processed\dupire_surfaces'
# output_file = r'C:\Users\adity\Downloads\vol surface dynamics\data\processed\dupire_pca_output_cleaned.npz'
# spike_threshold = 2  # NEW threshold for PC1 filtering

# # ----------------------------
# # LOAD AND FLATTEN SURFACES
# # ----------------------------
# pattern = re.compile(r'local_vol_surface_(\d{4}-\d{2}-\d{2})\.npz')
# files = sorted([f for f in os.listdir(surf_folder) if pattern.match(f)])

# surfaces = []
# dates = []

# for fname in files:
#     match = pattern.match(fname)
#     if not match:
#         print(f"⚠️ Skipping invalid filename: {fname}")
#         continue

#     date_str = match.group(1)
#     date = np.datetime64(date_str)

#     path = os.path.join(surf_folder, fname)
#     data = np.load(path)

#     if 'local_vol' not in data:
#         print(f"⚠️ Skipping {fname} — missing 'local_vol'")
#         continue

#     lv = data['local_vol']
#     if lv.shape != (50, 50):
#         print(f"⚠️ Skipping {fname} — unexpected shape {lv.shape}")
#         continue

#     if np.isnan(lv).any():
#         print(f"⚠️ Warning: {fname} has NaNs — should be filled already")

#     surfaces.append(lv.reshape(-1))
#     dates.append(date)

# # Final matrix
# if len(surfaces) == 0:
#     raise RuntimeError("❌ No valid surfaces found!")

# X_all = np.vstack(surfaces)
# dates_all = np.array(dates, dtype='datetime64[D]')
# print(f"✅ Loaded {X_all.shape[0]} surfaces, each of size {X_all.shape[1]}")

# # ----------------------------
# # TEMP PCA to detect PC1 spikes
# # ----------------------------
# pca_temp = PCA(n_components=5)
# X_temp_pca = pca_temp.fit_transform(X_all)

# # Detect spike dates in PC1 > 2
# spike_indices = np.where(np.abs(X_temp_pca[:, 0]) > spike_threshold)[0]
# spike_dates = dates_all[spike_indices]

# print("\n⚠️ Spike Dates Detected in PC1 > 2:")
# for idx in spike_indices:
#     print(f"   {dates_all[idx]}: PC1 = {X_temp_pca[idx, 0]:.2f}")

# # ----------------------------
# # Remove spikes
# # ----------------------------
# mask = np.ones(len(X_all), dtype=bool)
# mask[spike_indices] = False

# X_clean = X_all[mask]
# dates_clean = dates_all[mask]

# # ----------------------------
# # PCA on cleaned data
# # ----------------------------
# n_components = min(5, X_clean.shape[0])
# pca = PCA(n_components=n_components)
# X_pca = pca.fit_transform(X_clean)

# print("\n✅ PCA complete on cleaned data")
# print("Explained variance ratio:", pca.explained_variance_ratio_)

# # ----------------------------
# # SAVE
# # ----------------------------
# np.savez_compressed(
#     output_file,
#     pca_components=X_pca,
#     dates=dates_clean,
#     pca_model=pca.components_,
#     explained_variance=pca.explained_variance_ratio_
# )
# print(f"✅ Saved cleaned PCA output to: {output_file}")

# # ----------------------------
# # Plot variance
# # ----------------------------
# plt.figure(figsize=(6, 4))
# plt.plot(np.cumsum(pca.explained_variance_ratio_) * 100, marker='o')
# plt.title("Explained Variance (Dupire PCA - Cleaned)")
# plt.xlabel("Number of Components")
# plt.ylabel("Cumulative Variance (%)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # ----------------------------
# # Plot PC1–PC5 over Time (Cleaned)
# # ----------------------------
# plt.figure(figsize=(10, 6))
# for i in range(n_components):
#     plt.plot(dates_clean, X_pca[:, i], label=f'PC{i+1}')
# plt.title("PCA Component Scores (Cleaned) over Time")
# plt.xlabel("Date")
# plt.ylabel("Component Value")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.xticks(rotation=45)
# plt.show()

# # ----------------------------
# # Plot spike surface(s)
# # ----------------------------
# for spike_date in spike_dates:
#     spike_fname = f"local_vol_surface_{str(spike_date)}.npz"
#     spike_path = os.path.join(surf_folder, spike_fname)

#     if not os.path.exists(spike_path):
#         print(f"⚠️ File not found: {spike_fname}")
#         continue

#     data = np.load(spike_path)
#     surface = data['local_vol']

#     fig = plt.figure(figsize=(10, 7))
#     ax = fig.add_subplot(111, projection='3d')
#     x = np.linspace(-0.01, 0.01, 50)  # log-moneyness approx
#     y = np.linspace(0.01, 0.06, 50)   # TTM approx
#     Xg, Yg = np.meshgrid(x, y)

#     ax.plot_surface(Xg, Yg, surface, cmap='plasma')
#     ax.set_title(f"Dupire Local Vol Surface on {spike_date}")
#     ax.set_xlabel("Log-Moneyness")
#     ax.set_ylabel("Time to Expiry (years)")
#     ax.set_zlabel("Local Volatility")
#     plt.tight_layout()
#     plt.show()

# # ----------------------------
# # Identify Max Spikes in Each PC
# # ----------------------------
# top_k = 3  # top N spikes per component

# for i in range(n_components):
#     abs_vals = np.abs(X_pca[:, i])
#     top_indices = np.argsort(abs_vals)[-top_k:][::-1]  # largest first
#     print(f"\n📈 Top {top_k} spikes in PC{i+1}:")
#     for idx in top_indices:
#         print(f"   {dates_clean[idx]} — PC{i+1} = {X_pca[idx, i]:.2f}")
import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import re

# ----------------------------
# CONFIG
# ----------------------------
surf_folder = r'C:\Users\adity\Downloads\vol surface dynamics\data\processed\dupire_surfaces'
output_file = r'C:\Users\adity\Downloads\vol surface dynamics\data\processed\dupire_pca_output_cleaned.npz'
spike_threshold = 2  # PC1 spike threshold

# ----------------------------
# LOAD AND FLATTEN SURFACES
# ----------------------------
pattern = re.compile(r'local_vol_surface_(\d{4}-\d{2}-\d{2})\.npz')
files = sorted([f for f in os.listdir(surf_folder) if pattern.match(f)])

surfaces = []
dates = []

for fname in files:
    match = pattern.match(fname)
    if not match:
        continue

    date_str = match.group(1)
    date = np.datetime64(date_str)
    path = os.path.join(surf_folder, fname)

    data = np.load(path)
    if 'local_vol' not in data:
        continue

    lv = data['local_vol']
    if lv.shape != (50, 50):
        continue

    surfaces.append(lv.reshape(-1))
    dates.append(date)

X_all = np.vstack(surfaces)
dates_all = np.array(dates, dtype='datetime64[D]')
print(f"✅ Loaded {X_all.shape[0]} surfaces")

# ----------------------------
# TEMP PCA to DETECT SPIKES
# ----------------------------
pca_temp = PCA(n_components=5)
X_temp_pca = pca_temp.fit_transform(X_all)

# Find indices where |PC1| > 2
spike_indices = np.where(np.abs(X_temp_pca[:, 0]) > spike_threshold)[0]
spike_dates = dates_all[spike_indices]

print(f"\n⚠️ Found {len(spike_indices)} spike dates where |PC1| > {spike_threshold}")
for i in spike_indices:
    print(f"   {dates_all[i]} — PC1 = {X_temp_pca[i, 0]:.2f}")

# ----------------------------
# FILTER: Remove spike rows
# ----------------------------
mask = np.abs(X_temp_pca[:, 0]) <= spike_threshold
X_clean = X_all[mask]
dates_clean = dates_all[mask]

print(f"\n✅ Cleaned dataset shape: {X_clean.shape}")

# ----------------------------
# FINAL PCA on CLEANED DATA
# ----------------------------
n_components = min(5, X_clean.shape[0])
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_clean)

print("\n✅ Final PCA on cleaned data done.")
print("Explained variance ratio:", pca.explained_variance_ratio_)

# ----------------------------
# SAVE CLEANED OUTPUT
# ----------------------------
np.savez_compressed(
    output_file,
    pca_components=X_pca,
    dates=dates_clean,
    pca_model=pca.components_,
    explained_variance=pca.explained_variance_ratio_
)
print(f"✅ Saved cleaned PCA output to: {output_file}")

# ----------------------------
# PLOT: Cumulative variance
# ----------------------------
plt.figure(figsize=(6, 4))
plt.plot(np.cumsum(pca.explained_variance_ratio_) * 100, marker='o')
plt.title("Explained Variance (Cleaned)")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance (%)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------
# PLOT: PC1–PC5 over time
# ----------------------------
plt.figure(figsize=(10, 6))
for i in range(n_components):
    plt.plot(dates_clean, X_pca[:, i], label=f'PC{i+1}')
plt.title("Cleaned PCA Component Scores over Time")
plt.xlabel("Date")
plt.ylabel("Component Value")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
