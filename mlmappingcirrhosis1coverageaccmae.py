#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 21:20:14 2025

@author: mila.s
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Load the dataset
df = pd.read_csv('/Users/mila.s/Downloads/Research/DT in healthcare/liver/BHI/dataset/liver_cirrhosis.csv')

# Filter dataset for cirrhosis stages 1, 2, 3 and healthy synthetic data
df_cirr = df[['Stage', 'Bilirubin']].dropna()
df_cirr = df_cirr[df_cirr['Stage'].isin([1, 2, 3])]

# Add class labels: Healthy=0, Cirrhosis 1=3, 2=4, 3=5
df_cirr['Class'] = df_cirr['Stage'] + 2

# Create synthetic healthy data
healthy_samples = pd.DataFrame({
    'Stage': [0] * 800,
    'Bilirubin': np.random.uniform(0.1, 1.3, 800),
    'Class': [0] * 800
})

# Combine training data (Healthy, Cirrhosis 2 and 3)
df_train = pd.concat([
    healthy_samples,
    df_cirr[df_cirr['Stage'].isin([2, 3])]  # Cirrhosis 2 and 3
])

X_train = df_train[['Class']]
y_train = df_train['Bilirubin']

# Train the model
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# Predict for Cirrhosis Stage 1 (class 3)
pred_stage = np.array([[3]])
pred_bili = model.predict(pred_stage)[0]

# Estimate standard deviation for prediction range
residuals = y_train - model.predict(X_train)
std_error = np.std(residuals)

# Predicted range
low = round(max(pred_bili - std_error, 0.2), 2)
high = round(pred_bili + std_error, 2)

# Evaluation on Cirrhosis Stage 1
real_vals = df[df['Stage'] == 1]['Bilirubin'].dropna().values
within_range = ((real_vals >= low) & (real_vals <= high)).sum()
coverage = 100 * within_range / len(real_vals)

# MAE between midpoint and real mean
pred_mid = (low + high) / 2
real_mean = real_vals.mean()
mae = abs(pred_mid - real_mean)

# Plotting
stage_labels = ['Healthy', 'Cirrhosis 1', 'Cirrhosis 2', 'Cirrhosis 3']
x_positions = np.arange(len(stage_labels))
plot_data = {
    'Healthy': healthy_samples['Bilirubin'].values,
    'Cirrhosis 1': df[df['Stage'] == 1]['Bilirubin'].dropna().values,
    'Cirrhosis 2': df[df['Stage'] == 2]['Bilirubin'].dropna().values,
    'Cirrhosis 3': df[df['Stage'] == 3]['Bilirubin'].dropna().values
}

plt.figure(figsize=(10, 6))
for i, label in enumerate(stage_labels):
    if label in plot_data:
        plt.boxplot(plot_data[label], positions=[i], widths=0.5)

# Add shaded predicted range for Cirrhosis 1
plt.fill_between([1 - 0.2, 1 + 0.2], low, high, color='lightblue', alpha=0.6,
                 label='Predicted Range for Cirrhosis 1')

plt.axhline(y=1.3, color='red', linestyle='--', linewidth=1.5, label='Upper Limit (1.3 mg/dL)')
plt.xticks(ticks=x_positions, labels=stage_labels)
plt.ylabel("Bilirubin (mg/dL)")
plt.title("ML-Predicted vs Actual Bilirubin Ranges")
plt.grid(True)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

print(coverage, mae)
