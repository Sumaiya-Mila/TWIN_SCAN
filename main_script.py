# -*- coding: utf-8 -*-
"""
Created on Thu May  1 19:36:19 2025

@author: mila.s
"""

import importlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Dynamically import the generic and personalized DT modules
generic_dt = importlib.import_module("generic_dt")
personalized_dt = importlib.import_module("personalized_dt")

# Load dataset with age and gender for personalized DT
df = pd.read_csv("C:/Users/mila.s/Downloads/Research/DT in healthcare/liver/BHI/dataset/liver_cirrhosis.csv")
df_cirr = df[['Stage', 'Bilirubin', 'Age', 'Sex']].dropna()

# Stage mapping
stage_labels = {1: "Cirrhosis 1", 2: "Cirrhosis 2", 3: "Cirrhosis 3"}
x = np.arange(len(stage_labels))

# Get generic DT predictions
generic_ranges = []
for real_stage in [1, 2, 3]:
    internal_stage = real_stage + 2
    low, high = generic_dt.simulate_bilirubin_range_wide(internal_stage)
    generic_ranges.append((low, high))

# Get personalized DT predictions (average across patients)
personalized_ranges = []
for real_stage in [1, 2, 3]:
    internal_stage = real_stage + 2
    stage_data = df_cirr[df_cirr['Stage'] == real_stage]
    lows, highs = [], []

    for _, row in stage_data.iterrows():
        low, high = personalized_dt.personalized_dt_bilirubin_range(
            internal_stage, row['Age'], row['Sex']
        )
        lows.append(low)
        highs.append(high)

    personalized_ranges.append((np.mean(lows), np.mean(highs)))

# Get actual dataset values for plotting boxplots
plot_data = {
    'Cirrhosis 1': df_cirr[df_cirr['Stage'] == 1]['Bilirubin'].values,
    'Cirrhosis 2': df_cirr[df_cirr['Stage'] == 2]['Bilirubin'].values,
    'Cirrhosis 3': df_cirr[df_cirr['Stage'] == 3]['Bilirubin'].values
}

# Plotting
plt.figure(figsize=(10, 6))
labels = list(stage_labels.values())

# Boxplots of dataset
for i, label in enumerate(labels):
    plt.boxplot(plot_data[label], positions=[i], widths=0.4)

# Overlay predicted ranges
colors = ['lightcoral', 'lightskyblue', 'lightgreen']
for i, (g_range, p_range) in enumerate(zip(generic_ranges, personalized_ranges)):
    # Generic DT range (shifted slightly left)
    plt.fill_between([i - 0.25, i - 0.05], g_range[0], g_range[1], color=colors[i], alpha=0.99,hatch='///', edgecolor='black', label=f'Generic DT {labels[i]}')

    # Personalized DT range (shifted slightly right)
    plt.fill_between([i + 0.05, i + 0.25], p_range[0], p_range[1], color=colors[i], alpha=0.99, label=f'Personalized DT {labels[i]}')

# Final styling
plt.xticks(x, labels)
plt.ylabel("Bilirubin (mg/dL)")
plt.title("Generic vs. Personalized DT Predictions vs. Real Cirrhosis Data")
plt.grid(True)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
