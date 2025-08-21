# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 19:39:28 2025

@author: mila.s
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})

# Load dataset
df = pd.read_csv('/Users/mila.s/Downloads/Research/DT in healthcare/liver/BHI/dataset/liver_cirrhosis.csv')
df_cirr = df[['Stage', 'Bilirubin']].dropna()

# Mapping  code to disease labels
stage_labels = {
    0: "Healthy",
    1: "NAFLD",
    2: "Fibrosis",
    3: "Cirrhosis 1",
    4: "Cirrhosis 2",
    5: "Cirrhosis 3"
}

def simulate_bilirubin_range_wide(disease_state, width_factor=8.0):
    """
    this function simulates a bilirubin prediction range for a given disease state (0-5).
    Uses a physiology-inspired production-clearance model with stage-dependent spread.
    Returns (low, high) bilirubin range in mg/dL.
    """
    # normalized progression 0->1
    sf = disease_state / 5.0

    # Subfunction factors 
    f_hep = 1 - 0.6 * sf
    oxi = 0.1 + 0.99 * sf
    heme_cat = 1 - 0.3 * sf
    hemolysis = 0.8 + 0.2 * sf
    ugt = 1 - 0.5 * sf

    # Raw (unscaled) conjugation capacity
    raw_conj = f_hep * ugt * heme_cat / oxi

    # Normalize the raw_conj to 0–1 across all possible stages
    # Estimate rough max/min based on endpoints
    raw_conj_min = (1 - 0.6) * (1 - 0.5) * (1 - 0.3) / (0.1 + 0.99 * 1)  # stage 5
    raw_conj_max = 1 * 1 * 1 / (0.1)                                     # stage 0
    normalized_conj = (raw_conj - raw_conj_min) / (raw_conj_max - raw_conj_min)

    # Now scale to target range
    target_min, target_max = {
        0: (0.75, 0.9),
        1: (0.7, 0.9),
        2: (0.6, 0.8),
        3: (0.5, 0.69),
        4: (0.45, 0.65),
        5: (0.25, 0.55),
        }[disease_state]

    # Scaled conjugation capacity
    conj_cap = target_min + normalized_conj * (target_max - target_min)


    # production and clearance
    production = hemolysis * 1.5
    clearance = production * conj_cap
    #total = max(production - clearance, 0.2)
    total = (production - clearance)

    # stage-dependent spread
    spread = 1.2 + sf * width_factor
    low = (round(total/spread , 2))
    high = round(total * spread, 2)
    #print(spread)
    #print(low)
    #print(high)

    return low, high

# Generate DT predictions for all stages 0-5
dt_results = []
for s in range(6):
    low, high = simulate_bilirubin_range_wide(s)
    dt_results.append({
        'Stage': s,
        'Label': stage_labels[s],
        'Predicted_Low': low,
        'Predicted_High': high
    })
dt_df = pd.DataFrame(dt_results)

# Display DT predictions
print("Digital Twin Predicted Bilirubin Ranges:")
print(dt_df)

# Prepare actual data for cirrhosis stages 1-3
plot_data = {
    'Cirrhosis 1': df_cirr[df_cirr['Stage'] == 1]['Bilirubin'].values,
    'Cirrhosis 2': df_cirr[df_cirr['Stage'] == 2]['Bilirubin'].values,
    'Cirrhosis 3': df_cirr[df_cirr['Stage'] == 3]['Bilirubin'].values
}
labels = list(plot_data.keys())
x = np.arange(len(labels))

# Choose a distinct color for each stage’s DT range
dt_colors = ['lightcoral', 'lightskyblue', 'lightgreen']

plt.figure(figsize=(8, 5))

# 1) Plot real data boxplots
for i, lbl in enumerate(labels):
    plt.boxplot(plot_data[lbl], positions=[i], widths=0.6)

# 2) Overlay DT ranges, one color per stage
for i, real_stage in enumerate([1, 2, 3]):
    # map real_stage → internal code (1→3, 2→4, 3→5)
    internal = real_stage + 2
    low, high = dt_df.loc[dt_df['Stage'] == internal, ['Predicted_Low','Predicted_High']].values[0]
    plt.fill_between(
        [i-0.3, i+0.3],
        [low, low],
        [high, high],
        color=dt_colors[i],
        alpha=0.5,
        label=f'Predicted Range for {labels[i]}'
    )

plt.xticks(x, labels)
plt.ylabel('Bilirubin (mg/dL)')
plt.title('Partial DT(generic) Predicted Ranges vs. Real Cirrhosis Data')
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

from sklearn.metrics import mean_absolute_error

# Initialize results
coverage_results = []
mae_results = []

for real_stage in [1, 2, 3]:
    internal_stage = real_stage + 2
    low, high = dt_df.loc[dt_df['Stage'] == internal_stage, ['Predicted_Low', 'Predicted_High']].values[0]
    real_vals = df_cirr[df_cirr['Stage'] == real_stage]['Bilirubin'].dropna().values

    # Coverage accuracy
    within_range = ((real_vals >= low) & (real_vals <= high)).sum()
    coverage = 100 * within_range / len(real_vals)
    coverage_results.append((stage_labels[internal_stage], round(coverage, 2)))

    # MAE between DT midpoint and real mean
    dt_midpoint = (low + high) / 2
    real_mean = real_vals.mean()
    mae = abs(dt_midpoint - real_mean)
    mae_results.append((stage_labels[internal_stage], round(mae, 2)))

# Display results
print("\n Coverage Accuracy (% of data within DT-predicted range):")
for stage, cov in coverage_results:
    print(f"  {stage}: {cov:.2f}%")

print("\n Mean Absolute Error (midrange DT vs. dataset mean):")
for stage, err in mae_results:
    print(f"  {stage}: {err:.2f} mg/dL")
