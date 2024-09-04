import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')

# Load the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv'
data = pd.read_csv(url)

print(data.head)
print(data.columns)

# Standardize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

# Apply Mean Shift clustering
mean_shift = MeanShift()
mean_shift.fit(data_scaled)
labels = mean_shift.labels_

print('Hi')
print(labels)


# Add the cluster labels to the original dataframe
data['Cluster'] = labels

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Fresh', y='Milk', hue='Cluster', palette='Set1')
plt.title('Mean Shift Clustering of Wholesale Customers')
plt.xlabel('Fresh')
plt.ylabel('Milk')
plt.show()

# Display the cluster centers
cluster_centers = mean_shift.cluster_centers_
print("Cluster Centers:")
print(cluster_centers)

# Cluster centers in original scale
cluster_centers_original = scaler.inverse_transform(cluster_centers)
print("Cluster Centers in Original Scale:")
print(pd.DataFrame(cluster_centers_original, columns=data.columns[:-1]))





# Analyze each cluster by calculating the mean of each feature within each cluster
cluster_centers = pd.DataFrame(mean_shift.cluster_centers_, columns=data.columns[:-1])
print("\nCluster centers (mean values of features for each cluster):")
print(cluster_centers)

# Add the original scale back to the cluster centers
cluster_centers_original_scale = scaler.inverse_transform(mean_shift.cluster_centers_)
cluster_centers_original = pd.DataFrame(cluster_centers_original_scale, columns=data.columns[:-1])
#print("\nCluster centers in original scale:")
#print(cluster_centers_original)

# Interpretation of clusters
for cluster_num in range(len(cluster_centers_original)):
    print(f"\nCluster {cluster_num}:")
    print(cluster_centers_original.iloc[cluster_num])
    if cluster_centers_original.iloc[cluster_num]['Fresh'] > cluster_centers_original['Fresh'].mean():
        print("This cluster spends a lot on Fresh products.")
    if cluster_centers_original.iloc[cluster_num]['Grocery'] > cluster_centers_original['Grocery'].mean():
        print("This cluster spends a lot on Grocery.")
    if cluster_centers_original.iloc[cluster_num]['Detergents_Paper'] > cluster_centers_original['Detergents_Paper'].mean():
        print("This cluster is likely to be a retailer.")
    if cluster_centers_original.iloc[cluster_num]['Milk'] > cluster_centers_original['Milk'].mean():
        print("This cluster spends a lot on Milk.")


from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(data_scaled, data['Cluster'])
print(f'Silhouette Score: {silhouette_avg}')
