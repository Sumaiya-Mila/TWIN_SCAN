#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 20:16:49 2025

@author: mila.s
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv('/Users/mila.s/Downloads/Research/DT in healthcare/liver/BHI/dataset/liver_cirrhosis.csv')
df_cirr = df[['Stage', 'Bilirubin', 'Age', 'Sex']].dropna()

# Mapping code to disease labels
stage_labels = {
    0: "Healthy",
    1: "NAFLD",
    2: "Fibrosis",
    3: "Cirrhosis 1",
    4: "Cirrhosis 2",
    5: "Cirrhosis 3"
}

def personalized_dt_bilirubin_range(disease_state, age, gender, width_factor=8.0):
    sf = disease_state / 5.0

    # Increased sensitivity to personalization factors
    gender_factor = 0.9 if gender.lower() == 'male' else 1.05
    age_factor = max(0.85 - (age - 50) * 0.004, 0.7)

    # Optimized subfunctions
    f_hep = (1 - 0.55 * sf) * age_factor
    oxi = 0.1 + 1.1 * sf
    heme_cat = (1 - 0.25 * sf) * age_factor
    hemolysis = (0.85 + 0.25 * sf)
    ugt = (1 - 0.45 * sf) * gender_factor * age_factor

    # Normalize conjugation
    raw_conj = f_hep * ugt * heme_cat / oxi
    raw_conj_min = (1 - 0.55) * (1 - 0.45) * (1 - 0.25) / (0.1 + 1.1 * 1)
    raw_conj_max = 1 * 1 * 1 / 0.1
    normalized_conj = (raw_conj - raw_conj_min) / (raw_conj_max - raw_conj_min)

    target_min, target_max = {
        0: (0.75, 0.9),
        1: (0.7, 0.9),
        2: (0.6, 0.8),
        3: (0.5, 0.69),
        4: (0.45, 0.65),
        5: (0.25, 0.55),
    }[disease_state]

    conj_cap = target_min + normalized_conj * (target_max - target_min)
    production = hemolysis * 1.6
    clearance = production * conj_cap
    total = production - clearance

    spread = 1.2 + sf * width_factor
    low = round(total / spread, 2)
    high = round(total * spread, 2)
    return low, high

# Generate personalized DT predictions
dt_preds = []
for _, row in df_cirr.iterrows():
    stage = int(row['Stage'])
    age = float(row['Age'])
    gender = str(row['Sex'])
    internal_stage = stage + 2
    if internal_stage > 5:
        continue
    low, high = personalized_dt_bilirubin_range(internal_stage, age, gender)
    dt_preds.append({'Stage': internal_stage, 'Low': low, 'High': high, 'Actual': row['Bilirubin']})

pred_df = pd.DataFrame(dt_preds)

from sklearn.metrics import mean_absolute_error



# Coverage Accuracy Calculation
coverage_results = []
for real_stage in [1, 2, 3]:
    internal_stage = real_stage + 2
    stage_data = pred_df[pred_df['Stage'] == internal_stage]
    if stage_data.empty:
        continue
    low = stage_data['Low'].mean()
    high = stage_data['High'].mean()
    real_vals = df_cirr[df_cirr['Stage'] == real_stage]['Bilirubin'].dropna().values
    within_range = ((real_vals >= low) & (real_vals <= high)).sum()
    coverage = 100 * within_range / len(real_vals)
    coverage_results.append((stage_labels[internal_stage], round(coverage, 2)))

# Per-patient MAE using DT-predicted midpoints
print("\nCoverage Accuracy (% of data within DT-predicted range):")
for stage, cov in coverage_results:
    print(f"  {stage}: {cov:.2f}%")

print("\nMean Absolute Error (midrange DT vs. dataset mean):")
for real_stage in [1, 2, 3]:
    internal_stage = real_stage + 2
    stage_data = pred_df[pred_df['Stage'] == internal_stage]
    if stage_data.empty:
        continue
    mae = np.mean(np.abs(stage_data['Actual'] - (stage_data['Low'] + stage_data['High']) / 2))
    print(f"  Cirrhosis {real_stage}: {mae:.2f} mg/dL")

# Visualization
plot_data = {
    'Cirrhosis 1': df_cirr[df_cirr['Stage'] == 1]['Bilirubin'].values,
    'Cirrhosis 2': df_cirr[df_cirr['Stage'] == 2]['Bilirubin'].values,
    'Cirrhosis 3': df_cirr[df_cirr['Stage'] == 3]['Bilirubin'].values
}
labels = list(plot_data.keys())
x = np.arange(len(labels))

# Mean predicted DT range per stage
stage_ranges = []
for real_stage in [1, 2, 3]:
    internal_stage = real_stage + 2
    stage_data = pred_df[pred_df['Stage'] == internal_stage]
    low_mean = stage_data['Low'].mean()
    high_mean = stage_data['High'].mean()
    stage_ranges.append((low_mean, high_mean))

# Plot
plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.size': 16})
for i, lbl in enumerate(labels):
    plt.boxplot(plot_data[lbl], positions=[i], widths=0.6)

colors = ['lightcoral', 'lightblue', 'lightgreen']
for i, (low, high) in enumerate(stage_ranges):
    plt.fill_between([i - 0.3, i + 0.3], low, high, color=colors[i], alpha=0.5, label=f'Personalized DT Range {labels[i]}')

plt.xticks(ticks=x, labels=labels)
plt.ylabel('Bilirubin (mg/dL)')
plt.title('Personalized DT Predicted Ranges vs. Real Cirrhosis Data')
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()