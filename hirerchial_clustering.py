import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the dataset
data = pd.read_csv('C:/Durgaprasad/DATASETS/Online_Retail.csv', encoding='ISO-8859-1')

# Display the first few rows of the dataset
print(data.head())

# Drop rows with missing CustomerID
data = data.dropna(subset=['CustomerID'])

# Create a 'TotalSpend' feature
data['TotalSpend'] = data['Quantity'] * data['UnitPrice']

# Group the data by CustomerID to create features for clustering
customer_data = data.groupby('CustomerID').agg({
    'InvoiceNo': 'nunique',   # Frequency (Number of unique invoices)
    'TotalSpend': 'sum',      # Monetary (Total money spent)
    'InvoiceDate': 'max'      # Recency (Last purchase date)
}).reset_index()

#  correct date format
customer_data['InvoiceDate'] = pd.to_datetime(customer_data['InvoiceDate'], format='%m/%d/%y %H:%M')


customer_data['Recency'] = (customer_data['InvoiceDate'].max() - customer_data['InvoiceDate']).dt.days

# Drop 'InvoiceDate' as it's no longer needed
customer_data = customer_data.drop('InvoiceDate', axis=1)

# Display the cleaned and feature-engineered data
print(customer_data.head())

# Standardize the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_data[['InvoiceNo', 'TotalSpend', 'Recency']])

# Convert back to a DataFrame for easier handling
scaled_data = pd.DataFrame(scaled_data, columns=['Frequency', 'Monetary', 'Recency'])

# Generate the linkage matrix
linked = linkage(scaled_data, method='ward')


# Plot the dendrogram to visualize the clusters
plt.figure(figsize=(8, 7))
dendrogram(linked, orientation='top', labels=customer_data['CustomerID'].values, distance_sort='ascending', show_leaf_counts=False)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Customer ID')
plt.ylabel('Euclidean distances')
plt.show()

# Taking k value from the Dendrogram
optimal_k = 3
hierarchical_clustering = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
customer_data['Cluster'] = hierarchical_clustering.fit_predict(scaled_data)

# Display the first few rows with the cluster labels
print(customer_data.head())

# Visualize the clusters using a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(customer_data['InvoiceNo'], customer_data['TotalSpend'], c=customer_data['Cluster'], cmap='viridis')
plt.title('Customer Segments')
plt.xlabel('Frequency (Number of Purchases)')
plt.ylabel('Monetary (Total Spend)')
plt.show()

from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(scaled_data, customer_data['Cluster'])
print(f'Silhouette Score: {silhouette_avg}')
