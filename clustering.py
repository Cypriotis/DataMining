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
    'one-hot encoding Script Type', 'one-hot encoding Genre', 'one-hot encoding Oscar Winners'
]

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
k_optimal = 2  # Adjust based on the plot

# Apply K-Means clustering
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
df['Cluster_KMeans'] = kmeans.fit_predict(df_scaled)
df.to_excel('/home/tofi-machine/Documents/DataMining/DataMining/moviesclustered.xlsx', index=False)  # Replace with your file path


# Characterize each cluster, focusing on Oscar Wins
oscar_columns = [
    'one-hot encoding Oscar Winners',
    'Rotten Tomatoes  critics', 'Metacritic  critics', 'Average critics ',
    'Rotten Tomatoes Audience ', 'Metacritic Audience ', 'Average audience ',
    'Audience vs Critics deviance '
]

# Select only numeric columns related to Oscar Wins
numeric_oscar_columns = [col for col in oscar_columns if col in df.columns]

cluster_characteristics_oscar = df.groupby('Cluster_KMeans')[numeric_oscar_columns].mean()
print(cluster_characteristics_oscar)

# Scatter plot for the first three features, focusing on Oscar Wins
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))

# Scatter plot for 'Year' vs 'Rotten Tomatoes critics'
plt.subplot(2, 2, 1)
sns.scatterplot(x='Year', y='Rotten Tomatoes  critics', hue='Cluster_KMeans', data=df, palette='Set1')
plt.title('Year vs Rotten Tomatoes Critics (Oscar Wins)')

# Scatter plot for 'Year' vs 'Metacritic critics'
plt.subplot(2, 2, 2)
sns.scatterplot(x='Year', y='Metacritic  critics', hue='Cluster_KMeans', data=df, palette='Set1')
plt.title('Year vs Metacritic Critics (Oscar Wins)')

# Scatter plot for 'Rotten Tomatoes critics' vs 'Metacritic critics'
plt.subplot(2, 2, 3)
sns.scatterplot(x='Rotten Tomatoes  critics', y='Metacritic  critics', hue='Cluster_KMeans', data=df, palette='Set1')
plt.title('Rotten Tomatoes Critics vs Metacritic Critics (Oscar Wins)')

# Remove the empty subplot in the last position
plt.subplot(2, 2, 4).axis('off')

plt.tight_layout()
plt.show()
