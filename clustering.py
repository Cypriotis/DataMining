import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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

# Determine the optimal number of clusters using the Elbow Method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()

# Based on the Elbow Method, choose the optimal number of clusters
k_optimal = 6  # Adjust based on the plot

# Apply K-Means clustering
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
df['Cluster_KMeans'] = kmeans.fit_predict(df_scaled)

# Characterize each cluster
cluster_characteristics = df.groupby('Cluster_KMeans').mean()
print(cluster_characteristics)

# Scatter plots focusing on Oscar Winners
sns.set(style="whitegrid")
plt.figure(figsize=(16, 10))

# Scatter plots for different comparisons with more focus on Oscar Winners
plt.subplot(2, 3, 1)
sns.scatterplot(x='Rotten Tomatoes  critics', y='Metacritic  critics',
                hue='one-hot encoding Oscar Winners', data=df, palette='Set1')
plt.title('Oscar Winners vs Critics Ratings')

plt.subplot(2, 3, 2)
sns.scatterplot(x='Rotten Tomatoes Audience ', y='Metacritic Audience ',
                hue='one-hot encoding Oscar Winners', data=df, palette='Set1')
plt.title('Oscar Winners vs Audience Ratings')

plt.subplot(2, 3, 3)
sns.scatterplot(x='Domestic gross ($million)', y='Foreign Gross ($million)',
                hue='one-hot encoding Oscar Winners', data=df, palette='Set1')
plt.title('Oscar Winners vs Gross Revenue')

plt.subplot(2, 3, 4)
sns.scatterplot(x='Average critics ', y='Average audience ',
                hue='one-hot encoding Oscar Winners', data=df, palette='Set1')
plt.title('Oscar Winners vs Average Ratings')

plt.subplot(2, 3, 5)
sns.scatterplot(x='one-hot encoding Oscar Winners', y='Rotten Tomatoes  critics',
                hue='Cluster_KMeans', data=df, palette='Set1')
plt.title('Oscar Winners vs Rotten Tomatoes Critics (Clustered)')

plt.tight_layout()
plt.show()
