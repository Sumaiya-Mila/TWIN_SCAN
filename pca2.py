# -*- coding: utf-8 -*-
"""
Created on Sun May 25 23:02:52 2025

@author: mila.s
"""

# Re-import necessary libraries after code execution state reset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Reload the dataset
df = pd.read_csv("C:/Users/mila.s/Downloads/Research/DT in healthcare/liver/BHI/dataset/liver_cirrhosis.csv")

# Drop non-numeric and 'Stage' column
features = df.drop(columns=['Stage'])
numeric_features = features.select_dtypes(include='number')

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_features)

# Run PCA
pca = PCA()
pca.fit(scaled_data)

# Calculate explained variance ratios
explained_var = pca.explained_variance_ratio_
cumulative_var = explained_var.cumsum()
components = [f'PC{i+1}' for i in range(len(explained_var))]

# Plot
plt.figure(figsize=(10, 6))
plt.bar(components, explained_var, alpha=0.7, label='Individual Explained Variance')
plt.plot(components, cumulative_var, marker='o', color='r', label='Cumulative Explained Variance')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Principal Components')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
