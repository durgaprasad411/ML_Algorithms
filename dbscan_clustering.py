import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from scipy.spatial import ConvexHull

# Load the dataset
data = pd.read_csv('C:/Durgaprasad/DATASETS/Mall_Customers.csv')

# Select relevant features for clustering
customer_data = data[['Annual Income (k$)', 'Spending Score (1-100)']].copy()

# Standardize the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_data)

# Convert back to a DataFrame for easier handling
scaled_data = pd.DataFrame(scaled_data, columns=['AnnualIncome', 'SpendingScore'])

# Grid search parameters
eps_values = np.arange(0.1, 1.1, 0.1)
min_samples_values = range(3, 10)
best_eps = None
best_min_samples = None
best_score = -1

# Perform grid search
for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(scaled_data)

        # Skip if the clustering resulted in only one cluster or all points as noise
        if len(set(clusters)) > 1 and len(set(clusters)) < len(scaled_data):
            try:
                score = silhouette_score(scaled_data, clusters)
                if score > best_score:
                    best_score = score
                    best_eps = eps
                    best_min_samples = min_samples
            except:
                pass

print(f'Best eps: {best_eps}, Best min_samples: {best_min_samples}')

# Apply DBSCAN with the best parameters found
dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
clusters = dbscan.fit_predict(scaled_data)

customer_data.loc[:, 'Cluster'] = clusters

# Display the first few rows with cluster labels
print(customer_data.head())

# Visualize the clusters using a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(customer_data['Annual Income (k$)'], customer_data['Spending Score (1-100)'], c=customer_data['Cluster'],
            cmap='viridis', marker='o')
plt.title('DBSCAN Clustering of Customers')
plt.xlabel('Annual Income (k)')
plt.ylabel('Spending Score (1-100)')
plt.colorbar(label='Cluster')

# Draw convex hulls around the clusters
unique_clusters = np.unique(clusters)
for i in unique_clusters:
    if i != -1:  # Skip noise points
        cluster_points = customer_data[customer_data['Cluster'] == i][
            ['Annual Income (k$)', 'Spending Score (1-100)']].values
        if len(cluster_points) > 2:  # ConvexHull needs at least 3 points
            hull = ConvexHull(cluster_points)
            for simplex in hull.simplices:
                plt.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], 'k-')##k for black line

plt.show()


from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(scaled_data, customer_data['Cluster'])
print(f'Silhouette Score: {silhouette_avg}')
