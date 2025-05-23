import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np
import warnings

# Suppress runtime warnings (like divide by zero, overflow, invalid value)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Load the dataset
df = pd.read_csv('../datasets/wholesale.csv')

# Drop the 'Channel' and 'Region' columns
df = df.drop(columns=['Channel', 'Region'])

# Preview the original data
print("Original Data (first 5 rows):")
print(df.head())

# Make a copy of the original (not-normalized) data
df_not_normalized = df.copy()

# Initialize Min-Max Scaler
scaler = MinMaxScaler()

# Apply Min-Max normalization
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Combine min and max values into one table
min_max_table = pd.DataFrame({
    'Min': df_normalized.min(),
    'Max': df_normalized.max()
})

print("\nMin-Max Table:")
print(min_max_table)

# Apply KMeans clustering with 2 clusters and random seed 12345
kmeans = KMeans(n_clusters=2, random_state=12345)
df_normalized['Cluster'] = kmeans.fit_predict(df_normalized)

# Preview the clustered data
print("\nClustered Data For Cluster (first 5 rows):")
print(df_normalized.head())

# Count the number of records in each cluster
cluster_counts = df_normalized['Cluster'].value_counts().sort_index()

print("\nNumber of records in each cluster:")
print(cluster_counts)

# Get centroids in normalized space
centroids_normalized = kmeans.cluster_centers_

# Reverse the Min-Max scaling to original scale
centroids_original = scaler.inverse_transform(centroids_normalized)

# Create a DataFrame for centroids with original feature names
centroids_df = pd.DataFrame(centroids_original, columns=df.columns)

print("\nCluster Centroids (in original feature space):")
print(centroids_df)

# Add cluster labels to original data
df_not_normalized['Cluster'] = df_normalized['Cluster'].values

# Create a smaller pairplot by reducing subplot size
sns.pairplot(df_not_normalized, hue='Cluster', diag_kind='kde', palette='Set1', height=1.5)

plt.suptitle('Scatter Plot Matrix of Wholesale Customers Dataset by Cluster', y=1.02)

# Compute silhouette scores for each sample (on normalized data used for clustering)
silhouette_vals = silhouette_samples(df_normalized.drop(columns=['Cluster']), df_normalized['Cluster'])

# Add silhouette scores to the dataframe
df_normalized['Silhouette'] = silhouette_vals

# Mean silhouette score overall
mean_silhouette_overall = silhouette_score(df_normalized.drop(columns=['Cluster', 'Silhouette']), df_normalized['Cluster'])

print(f"\nMean Silhouette Coefficient (overall): {mean_silhouette_overall:.4f}")

# Mean silhouette score per cluster
mean_silhouette_per_cluster = df_normalized.groupby('Cluster')['Silhouette'].mean()

print("\nMean Silhouette Coefficient (per cluster):")
print(mean_silhouette_per_cluster)

print("\nMean Silhouette Coefficient for different values of K:")

results = []

for k in range(2, 6):
    kmeans = KMeans(n_clusters=k, random_state=12345)
    labels = kmeans.fit_predict(df_normalized.drop(columns=['Cluster', 'Silhouette'], errors='ignore'))

    overall_score = silhouette_score(df_normalized.drop(columns=['Cluster', 'Silhouette'], errors='ignore'), labels)
    sample_silhouette_vals = silhouette_samples(df_normalized.drop(columns=['Cluster', 'Silhouette'], errors='ignore'), labels)

    temp_df = pd.DataFrame({'Cluster': labels, 'Silhouette': sample_silhouette_vals})
    mean_per_cluster = temp_df.groupby('Cluster')['Silhouette'].mean()

    # Prepare a row dict with cluster means
    row = {'K': k}
    for cluster_id, cluster_silhouette in mean_per_cluster.items():
        row[f'Cluster {cluster_id}'] = cluster_silhouette
    # Add overall at the end after clusters
    row['Overall'] = overall_score

    results.append(row)

results_df = pd.DataFrame(results)

# Reorder columns to put Overall at the end (after all cluster columns)
cols = [col for col in results_df.columns if col != 'Overall'] + ['Overall']
results_df = results_df[cols]

print(results_df)


plt.show()
