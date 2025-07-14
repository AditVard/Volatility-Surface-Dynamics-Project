import pandas as pd
import numpy as np
import math
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

df = pd.read_csv('../../data/processed/greeks_iv_from_column-onlyonestrike.csv')

current_date = None
k_iv_list = []

for index, row in df.iterrows():
    date = row['date']
    expiry = row['expiry']
    K = row['strike']
    S = row['spot']
    iv = row['iv']

    if pd.isna(K) or pd.isna(S) or pd.isna(iv) or S == 0:
        continue

    # Start new smile when date changes
    if date != current_date:
        # Process the last batch
        if len(k_iv_list) >= 5:
            k_iv_list.sort(key=lambda x: x[0])
            k_vals = [x[0] for x in k_iv_list]
            iv_vals = [x[1] for x in k_iv_list]

            # Fit cubic spline
            cs = CubicSpline(k_vals, iv_vals)

            # Plot
            k_dense = np.linspace(min(k_vals), max(k_vals), 100)
            iv_smooth = cs(k_dense)

            plt.figure()
            plt.plot(k_vals, iv_vals, 'bo', label='Raw IV')
            plt.plot(k_dense, iv_smooth, 'r-', label='Spline IV')
            plt.title(f"IV Smile - {current_date} | Exp: {expiry}")
            plt.xlabel("log-moneyness")
            plt.ylabel("IV")
            plt.legend()
            plt.grid(True)
            plt.show()

        # Reset for next day
        current_date = date
        k_iv_list = []

    # Compute log-moneyness and store
    try:
        k = math.log(K / S)
        k_iv_list.append((k, iv))
    except:
        continue
