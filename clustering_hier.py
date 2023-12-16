import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset into a DataFrame
df = pd.read_excel("/home/tofi-machine/Documents/DataMining/DataMining/movies.xlsx")

# Select relevant features for clustering (including 'Year')
features_for_clustering = [
    'Year', 'Rotten Tomatoes  critics', 'Metacritic  critics', 'Average critics ',
    'Rotten Tomatoes Audience ', 'Metacritic Audience ', 'Average audience ',
    'Audience vs Critics deviance ', 'Domestic gross ($million)',
    'Foreign Gross ($million)', 'Worldwide Gross ($million)', 'Budget ($million)',
    'one-hot encoding ScriptType', 'one-hot encoding Genre', 'one-hot encoding Oscar Winners'
]

# Replace non-numeric values with NaN in 'Foreign Gross' and 'Worldwide Gross'
df['Foreign Gross'] = pd.to_numeric(df['Foreign Gross'], errors='coerce')
df['Worldwide Gross'] = pd.to_numeric(df['Worldwide Gross'], errors='coerce')

# Convert the columns to numeric
df['Foreign Gross'] = pd.to_numeric(df['Foreign Gross'])
df['Worldwide Gross'] = pd.to_numeric(df['Worldwide Gross'])

# Select only relevant features for clustering
df_for_clustering = df[features_for_clustering]

# Standardize the features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_for_clustering)

# Hierarchical clustering
linkage_matrix = linkage(df_scaled, method='ward', metric='euclidean')

# Plot the dendrogram
plt.figure(figsize=(15, 8))
dendrogram(linkage_matrix, p=5, truncate_mode='level', orientation='top', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Film Index')
plt.ylabel('Distance')
plt.show()
