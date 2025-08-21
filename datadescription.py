#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 17:02:59 2025

@author: mila.s
"""

import pandas as pd

# Load the dataset
df = pd.read_csv('/Users/mila.s/Downloads/Research/DT in healthcare/liver/BHI/dataset/liver_cirrhosis.csv')


# Drop rows with missing bilirubin values
df_clean = df.dropna(subset=["Bilirubin"])

# Compute stats using agg (safer than apply for flat columns)
summary = df_clean.groupby("Stage")["Bilirubin"].agg(
    Total_Samples="count",
    Mean="mean",
    Std="std",
    Min="min",
    Max="max",
    Lower_12_5=lambda x: x.quantile(0.125),
    Upper_87_5=lambda x: x.quantile(0.875)
).reset_index()

# Add the "High Data Density Range" as a string column
summary["High_Data_Density_Range"] = summary.apply(
    lambda row: f"[{round(row['Lower_12_5'], 2)} – {round(row['Upper_87_5'], 2)}]",
    axis=1
)

# Select final columns to display
summary_display = summary[[
    "Stage", "Total_Samples", "Mean", "Std", "Min", "Max", "High_Data_Density_Range"
]].round(2)

# Rename for display
summary_display.rename(columns={"Stage": "Cirrhosis_Stage"}, inplace=True)

# Print the final table
print("\nBilirubin Summary by Cirrhosis Stage:")
print(summary_display.to_string(index=False))