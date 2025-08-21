import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
plt.rcParams.update({'font.size': 14})

# Load the dataset
df = pd.read_csv('C:/Users/mila.s/Downloads/Research/DT in healthcare/liver/BHI/dataset/liver_cirrhosis.csv')

#  Filter and map classification stages
df_filtered = df[['Stage', 'Bilirubin']].dropna()
df_filtered = df_filtered[df_filtered['Stage'].isin([1, 2, 3])]  # Cirrhosis stages only
df_filtered['Class'] = df_filtered['Stage'] + 2  # Cirrhosis 1→3,2→4,3→5

#  synthetic healthy data (Stage 0 → Class 0)
healthy_samples = pd.DataFrame({
    'Stage': [0] * 800,
    'Bilirubin': np.random.uniform(0.1, 1.2, 800),
    'Class': [0] * 800
})

# Combine datasets
df_all = pd.concat([df_filtered[['Bilirubin', 'Class']], healthy_samples[['Bilirubin', 'Class']]])

#  Train  classifier
X = df_all[['Class']]
y = df_all['Bilirubin']
model = GradientBoostingRegressor(random_state=42)
model.fit( X, y)

pred_stages = np.array([1, 2]).reshape(-1, 1)
pred_bili = model.predict(pred_stages)

# Estimate prediction range using residual standard deviation
residuals = y - model.predict(X)
std_error = np.std(residuals)
pred_ranges = [(round(p - std_error, 2), round(p + std_error, 2)) for p in pred_bili]


# Clamp predictions to known lower bound
pred_ranges = np.clip(pred_ranges, 0.2, None)


# =============================================================================
# #  Predict probabilities for bilirubin values if classification problem
# bilirubin_values = np.linspace(0.1, 15.0, 500).reshape(-1, 1)
# probs = model.predict_proba(bilirubin_values)
# 
# #  Get predicted ranges for NAFLD (class 1) and Fibrosis (class 2)
# threshold = 0.60
# nafld_range = bilirubin_values[(probs[:, 1] > threshold)].flatten()
# fibrosis_range = bilirubin_values[(probs[:, 2] > threshold)].flatten()
# 
# nafld_bounds = (round(nafld_range.min(), 2), round(nafld_range.max(), 2)) if len(nafld_range) > 0 else (None, None)
# fibrosis_bounds = (round(fibrosis_range.min(), 2), round(fibrosis_range.max(), 2)) if len(fibrosis_range) > 0 else (None, None)
# 
# =============================================================================
#  Visualization
stage_labels = ['Healthy', 'NAFLD', 'Fibrosis', 'Cirrhosis 1', 'Cirrhosis 2', 'Cirrhosis 3']
x_positions = np.arange(len(stage_labels))

# Actual bilirubin values from dataset
plot_data = {
    'Healthy': healthy_samples['Bilirubin'].values,
    'Cirrhosis 1': df[df['Stage'] == 1]['Bilirubin'].dropna().values,
    'Cirrhosis 2': df[df['Stage'] == 2]['Bilirubin'].dropna().values,
    'Cirrhosis 3': df[df['Stage'] == 3]['Bilirubin'].dropna().values
}

plt.figure(figsize=(10, 6))

# Add boxplots for dataset values
for i, label in enumerate(stage_labels):
    if label in plot_data:
        print(i)
        print(label)
        print('\n')
        plt.boxplot(plot_data[label], positions=[i], widths=0.5)

# Add shaded predicted ranges
plt.fill_between([1 - 0.2, 1 + 0.2], pred_ranges[0][0], pred_ranges[0][1],
                     color='lightblue', alpha=0.5, label='NAFLD Predicted Range')

plt.fill_between([2 - 0.2, 2 + 0.2], pred_ranges[1][0], pred_ranges[1][1],
                     color='lightgreen', alpha=0.5, label='Fibrosis Predicted Range')

plt.axhline(y=1.3, color='red', linestyle='--', linewidth=1.5, label='Upper Limit (1.3 mg/dL)')

# Final styling
plt.xticks(ticks=x_positions, labels=stage_labels)
plt.ylabel('Bilirubin (mg/dL)')
plt.title('Predicted and Observed Bilirubin Ranges Across Liver Disease Stages')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
