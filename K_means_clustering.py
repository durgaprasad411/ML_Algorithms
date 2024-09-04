import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('C:/Durgaprasad/DATASETS/Online_Retail.csv', encoding='ISO-8859-1')

print(data.shape)

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

# Specify the correct date format
customer_data['InvoiceDate'] = pd.to_datetime(customer_data['InvoiceDate'], format='%m/%d/%y %H:%M')
print(customer_data['InvoiceDate'])
# Create the 'Recency' feature (days since last purchase)
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

#  Elbow method for optimal number of clusters
wcss = []  # Within-cluster sum of squares

# Test k values from 1 to 10
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# Plot the Elbow method results
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='*')
plt.title('Elbow Method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.show()

# Choose the optimal k
optimal_k = 4


kmeans = KMeans(n_clusters=optimal_k, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(scaled_data)

# Display the first few rows with the cluster labels
print(customer_data.head())
print(customer_data)

print(customer_data['Cluster'])

print("HI")


# Visualize the clusters using a scatter plot

plt.figure(figsize=(10, 6))
scatter = plt.scatter(customer_data['InvoiceNo'], customer_data['TotalSpend'], c=customer_data['Cluster'], cmap='viridis')
plt.title('Customer Segments')
plt.xlabel('Frequency (Number of Purchases)')
plt.ylabel('Monetary (Total Spend)')
handles, labels = scatter.legend_elements()
plt.legend(handles, labels, title="Clusters")
plt.show()

from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(scaled_data, customer_data['Cluster'])
print(f'Silhouette Score: {silhouette_avg}')
