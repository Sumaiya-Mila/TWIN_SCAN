import pandas as pd  
import numpy as np   
from sklearn.decomposition import PCA  
from sklearn.preprocessing import StandardScaler  
import matplotlib.pyplot as plt  
import seaborn as sns


# Load the dataset
df = pd.read_csv("C:/Users/mila.s/Downloads/Research/DT in healthcare/liver/BHI/dataset/liver_cirrhosis.csv")
print("Dataset loaded. Shape:", df.shape)

# Separate features and target
#  'Stage' is the target (cirrhosis stage)
features = df.drop(columns=['Stage'])  # Remove the target
numeric_features = features.select_dtypes(include=[np.number])  # Keeping only numeric features
print("Numeric features used for PCA:", numeric_features.columns.tolist())

#  Standardize the features
#  scale features to have mean=0 and std=1
scaler = StandardScaler()
scaled_features = scaler.fit_transform(numeric_features)

#  Apply PCA
pca = PCA()  # We'll extract all components
pca_result = pca.fit_transform(scaled_features)

# Create the loadings matrix
loadings = pd.DataFrame(pca.components_.T,
                        index=numeric_features.columns,
                        columns=[f'PC{i+1}' for i in range(pca.n_components_)])

# Get top 3 contributors for PC1 to PC5
top_n = 3
top_contributors = {}

for pc in loadings.columns[:5]:  # For PC1 to PC5
    top_features = loadings[pc].abs().sort_values(ascending=False).head(top_n).index.tolist()
    top_contributors[pc] = top_features

# Convert to matrix-like DataFrame (PCs as columns, top features as rows)
top_contributors_df = pd.DataFrame(top_contributors).T
top_contributors_df.columns = [f"Top {i+1}" for i in range(top_n)]

# Display result
print("Top 3 Contributing Features to Each Principal Component (PC1 to PC5):\n")
print(top_contributors_df)

# Prepare data for plotting
plot_df = top_contributors_df.copy()

# Convert wide format to long format for seaborn heatmap
plot_matrix = plot_df.apply(lambda row: [f"{row['Top 1']}", f"{row['Top 2']}", f"{row['Top 3']}"], axis=1).tolist()
plot_matrix = pd.DataFrame(plot_matrix, index=plot_df.index, columns=['Top 1', 'Top 2', 'Top 3'])

# Create a categorical heatmap-style plot
plt.figure(figsize=(10, 4))
sns.heatmap([[1]*3]*5, annot=plot_matrix.values, fmt='', cmap='Blues',
            cbar=False, linewidths=0.5, linecolor='gray', xticklabels=plot_matrix.columns, yticklabels=plot_matrix.index)

plt.title("PCA Summary of Liver Biomarker Patterns")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()




# Prepare a visually enhanced heatmap of top contributors

# Create a matrix of categorical values (feature names)
plot_matrix = top_contributors_df.copy()

# Create a dummy numerical matrix for heatmap coloring
color_matrix = [[3, 2, 1]] * plot_matrix.shape[0]  # arbitrary gradient for color

# Plot heatmap
plt.figure(figsize=(10, 5))
sns.heatmap(color_matrix,
            annot=plot_matrix.values,
            fmt='',
            cmap='YlOrBr',  # visually appealing gradient
            cbar=False,
            linewidths=0.8,
            linecolor='Black',
            xticklabels=plot_matrix.columns,
            yticklabels=plot_matrix.index)

plt.title("PCA Summary of Liver Biomarker Patterns", fontsize=14)
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# Calculate explained variance ratios for all principal components
explained_variance_ratio = pca.explained_variance_ratio_

# Create a DataFrame to show variance explained by each PC
variance_df = pd.DataFrame({
    'Principal Component': [f'PC{i+1}' for i in range(len(explained_variance_ratio))],
    'Explained Variance Ratio': explained_variance_ratio
})

# Highlight the top components
top_variance_df = variance_df.head(5)
print(top_variance_df)
