import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib
import pickle

# Load dataset
data=pd.read_csv('C:\Durgaprasad\DATASETS\weight-height.csv')
# Display the first few rows of the dataset
print(data.head())


# Checking the column names
print(data.columns)

# Checking for null data
print(data.isnull().sum())

rq = data[data.isin(['?']).any(axis=1)]
print("\nRows containing '?':")
print(rq)


# Encode the target variable 'Gender'
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])


# Separate features and target
X = data[['Height', 'Weight']]
y = data['Gender']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling  the data
scaler = StandardScaler()
print("Before scaling: ",X_train)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("After scaling: ",X_train)

# Range of k values to try
k_values = range(1, 100)
accuracy_scores = []

# Loop over the range of k values
for k in k_values:
    # Initialize KNN with the current k
    knn = KNeighborsClassifier(n_neighbors=k)

    # Train the model
    knn.fit(X_train, y_train)

    # Predict on the test set
    y_pred = knn.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

# Find the best k value
best_k = k_values[np.argmax(accuracy_scores)]
print(f"Best k value: {best_k} with accuracy: {max(accuracy_scores)}")

# Plot accuracy vs. k
plt.plot(k_values, accuracy_scores, marker='*')
plt.title('KNN: Accuracy vs. k')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.show()

# Initialize KNN
knn = KNeighborsClassifier(n_neighbors=best_k)

# Train the model
knn.fit(X_train, y_train)

# Predict and evaluate
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


print(f"Accuracy: {accuracy}")




# Display the first few predictions
print("First few test values: ",y_test[:5])
print("First few predictions:")
print(y_pred[:5])


##saving the model
with open('knn_model.pkl','wb') as file:
    pickle.dump(knn,file)
with open('knn_model.pkl','rb') as file:
    saved_model=pickle.load(file)

print(saved_model)
h=float(input("Enter the height: "))
w=float(input("Enter the weight: "))
print(saved_model.predict([[h,w]]))





