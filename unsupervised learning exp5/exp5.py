import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load the Wholesale Customers Dataset
data = pd.read_csv('wholesale.csv')

# Select only the numeric columns for clustering
data_numeric = data  # If 'CHANNEL' and 'REGION' exist in your dataset, use this line.
# If 'CHANNEL' and 'REGION' don't exist, simply use: data_numeric = data

# Standardize the data to have mean=0 and variance=1
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric)

# Determine the optimal number of clusters using the Elbow Method
wcss = []  # Within-Cluster Sum of Squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph to find the optimal number of clusters
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Based on the Elbow Method, choose the optimal number of clusters (e.g., 4)
optimal_clusters = 4

# Apply K-Means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
cluster_labels = kmeans.fit_predict(data_scaled)

# Add the cluster labels to the original dataset
data['Cluster'] = cluster_labels

# Reduce dimensionality for visualization (you can skip this step if you prefer)
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)
data['PCA1'] = data_pca[:, 0]
data['PCA2'] = data_pca[:, 1]

# Visualize the clustered data using PCA
plt.figure(figsize=(10, 8))
scatter = plt.scatter(data['PCA1'], data['PCA2'], c=data['Cluster'], cmap='viridis')
plt.title('Clustering of Wholesale Customers (PCA)')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(*scatter.legend_elements(), title='Clusters')
plt.show()

# You can now analyze the clusters and draw conclusions based on the results.
