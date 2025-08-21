import importlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import DT and ML mapping modules
generic_dt = importlib.import_module("generic_dt")
personalized_dt = importlib.import_module("personalized_dt")
mlmapping = importlib.import_module("ml_mapping_nafld_fibrosis")

# Load dataset
df = pd.read_csv("C:/Users/mila.s/Downloads/Research/DT in healthcare/liver/BHI/dataset/liver_cirrhosis.csv")
df = df[['Stage', 'Bilirubin', 'Age', 'Sex']].dropna()

# Define target early stages
target_stages = {1: "NAFLD", 2: "Fibrosis"}
x = np.arange(len(target_stages))
colors = ['coral','mediumseagreen']

# --- Generic DT predictions ---
generic_ranges = []
for stage in target_stages:
    low, high = generic_dt.simulate_bilirubin_range_wide(stage)
    generic_ranges.append((low, high))

# --- Personalized DT predictions ---
personalized_ranges = []
for stage in target_stages:
    lows, highs = [], []
    for _, row in df.iterrows():
        low, high = personalized_dt.personalized_dt_bilirubin_range(stage, row['Age'], row['Sex'])
        lows.append(low)
        highs.append(high)
    personalized_ranges.append((np.mean(lows), np.mean(highs)))

# --- ML Mapping predictions from mlmappingnafld script ---
ml_pred_stages = np.array([1, 2]).reshape(-1, 1)
ml_preds = mlmapping.model.predict(ml_pred_stages)
ml_std = np.std(mlmapping.y - mlmapping.model.predict(mlmapping.X))
ml_ranges = [(max(round(p - ml_std, 2), 0.2), round(p + ml_std, 2)) for p in ml_preds]

# --- Plotting ---
plt.figure(figsize=(10, 6))
labels = list(target_stages.values())


for i, (g_range, p_range, ml_range) in enumerate(zip(generic_ranges, personalized_ranges, ml_ranges)):
    plt.fill_between([i - 0.25, i - 0.15], g_range[0], g_range[1], color=colors[i], alpha=0.9,hatch='///', edgecolor='black',label=f'Generic DT {labels[i]}')
    plt.fill_between([i - 0.10, i + 0.00], p_range[0], p_range[1], color=colors[i], alpha=0.9, hatch='\\\\\\', edgecolor='black',label=f'Personalized DT {labels[i]}')
    plt.fill_between([i + 0.05, i + 0.15], ml_range[0], ml_range[1], color=colors[i], alpha=1,hatch='...', edgecolor='black', label=f'ML Mapping {labels[i]}')

plt.xticks(x, labels)
plt.ylabel("Bilirubin (mg/dL)")
plt.title("DT vs ML model Prediction Comparison for NAFLD and Fibrosis Stage")
plt.legend(loc='upper center')
plt.grid(True)
plt.tight_layout()
plt.show()




##################ml in different color
plt.figure(figsize=(10, 6))
ml_color = 'skyblue'  # Distinct color just for ML bar

for i, (g_range, p_range, ml_range) in enumerate(zip(generic_ranges, personalized_ranges, ml_ranges)):
    # Generic DT bar
    plt.fill_between([i - 0.25, i - 0.15], g_range[0], g_range[1],
                     color=colors[i], alpha=0.9, hatch='///', edgecolor='black',
                     label=f'Generic DT {labels[i]}')

    # Personalized DT bar
    plt.fill_between([i - 0.10, i + 0.00], p_range[0], p_range[1],
                     color=colors[i], alpha=0.9, hatch='\\\\\\', edgecolor='black',
                     label=f'Personalized DT {labels[i]}')

    # ML Mapping bar (only this bar has a different color)
    plt.fill_between([i + 0.05, i + 0.15], ml_range[0], ml_range[1],
                     color=ml_color, alpha=1.0, hatch='...', edgecolor='black',
                     label=f'ML Mapping {labels[i]}')

plt.xticks(x, labels)
plt.ylabel("Bilirubin (mg/dL)")
plt.title("DT vs ML model Prediction Comparison for NAFLD and Fibrosis Stage")
plt.legend(loc='upper center')
plt.grid(True)
plt.tight_layout()
plt.show()
