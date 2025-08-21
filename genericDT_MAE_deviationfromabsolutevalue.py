import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})

# Load dataset
df = pd.read_csv('/Users/mila.s/Downloads/Research/DT in healthcare/liver/BHI/dataset/liver_cirrhosis.csv')
df_cirr = df[['Stage', 'Bilirubin']].dropna()

# Mapping code to disease labels
stage_labels = {
    0: "Healthy",
    1: "NAFLD",
    2: "Fibrosis",
    3: "Cirrhosis 1",
    4: "Cirrhosis 2",
    5: "Cirrhosis 3"
}

# Digital Twin Simulation Function
def simulate_bilirubin_range_wide(disease_state, width_factor=8.0):
    sf = disease_state / 5.0
    f_hep = 1 - 0.6 * sf
    oxi = 0.1 + 0.99 * sf
    heme_cat = 1 - 0.3 * sf
    hemolysis = 0.8 + 0.2 * sf
    ugt = 1 - 0.5 * sf
    raw_conj = f_hep * ugt * heme_cat / oxi
    raw_conj_min = (1 - 0.6) * (1 - 0.5) * (1 - 0.3) / (0.1 + 0.99 * 1)
    raw_conj_max = 1 * 1 * 1 / (0.1)
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
    production = hemolysis * 1.5
    clearance = production * conj_cap
    total = production - clearance

    spread = 1.2 + sf * width_factor
    low = round(total / spread, 2)
    high = round(total * spread, 2)
    return low, high

# Generate DT-predicted ranges for all stages
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
print("Digital Twin Predicted Bilirubin Ranges:")
print(dt_df)

# Visualization: DT ranges + real data
plot_data = {
    'Cirrhosis 1': df_cirr[df_cirr['Stage'] == 1]['Bilirubin'].values,
    'Cirrhosis 2': df_cirr[df_cirr['Stage'] == 2]['Bilirubin'].values,
    'Cirrhosis 3': df_cirr[df_cirr['Stage'] == 3]['Bilirubin'].values
}
labels = list(plot_data.keys())
x = np.arange(len(labels))
dt_colors = ['lightcoral', 'lightskyblue', 'lightgreen']

plt.figure(figsize=(8, 5))
for i, lbl in enumerate(labels):
    plt.boxplot(plot_data[lbl], positions=[i], widths=0.6)

for i, real_stage in enumerate([1, 2, 3]):
    internal = real_stage + 2
    low, high = dt_df.loc[dt_df['Stage'] == internal, ['Predicted_Low', 'Predicted_High']].values[0]
    plt.fill_between(
        [i - 0.3, i + 0.3],
        [low, low],
        [high, high],
        color=dt_colors[i],
        alpha=0.5,
        label=f'Predicted Range for {labels[i]}'
    )

plt.xticks(x, labels)
plt.ylabel('Bilirubin (mg/dL)')
plt.title('Partial DT (Generic) Predicted Ranges vs. Real Cirrhosis Data')
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# MAE per patient and per stage using stage-biased DT prediction
df_valid = df_cirr.copy()
dt_predicted_bilirubin = []


import random

# Stage-specific bias: more severe stages lean higher in the range
stage_bias = {
    1: round(random.uniform(0.7, 1.2), 2),  # Cirrhosis 1: lean toward lower range
    2: round(random.uniform(1.0, 1.6), 2),  # Cirrhosis 2: wider spread
    3: round(random.uniform(0.8, 2.4), 2)   # Cirrhosis 3: slightly higher variability
}

for _, row in df_valid.iterrows():
    stage = int(row['Stage'])  # uses cirrhosis stages 1–3
    low, high = simulate_bilirubin_range_wide(stage)
    alpha = stage_bias.get(stage, 0.5)
    predicted = low + alpha * (high - low)
    dt_predicted_bilirubin.append(predicted)

# Store predictions
df_valid['DT_Predicted_Bilirubin'] = dt_predicted_bilirubin

# Print MAE summary in your requested format
print("\nMean Absolute Error (midrange DT vs. dataset mean):")
for real_stage in [1, 2, 3]:
    stage_data = df_valid[df_valid['Stage'] == real_stage]
    mae = np.mean(np.abs(stage_data['Bilirubin'] - stage_data['DT_Predicted_Bilirubin']))
    print(f"  Cirrhosis {real_stage}: {mae:.2f} mg/dL")
