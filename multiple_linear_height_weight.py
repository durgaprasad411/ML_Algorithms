import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# Loading the dataset
x = pd.read_csv('C:\Durgaprasad\DATASETS\weight-height.csv')
print(x.head())

# Checking the column names
print(x.columns)

# Checking for null data
print(x.isnull().sum())

# Splitting the data into independent (X) and dependent (y) variables
y = x['Weight']
X = x[['Gender', 'Height']]

# Encoding the 'Gender' column using OneHotEncoder
X = pd.get_dummies(X, columns=['Gender'], drop_first=True)
print(X.head())

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Creating and training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting the model by test data
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualizing predictions vs actual values
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.title('Predicted vs Actual Weights')
plt.xlabel('Actual Weights')
plt.ylabel('Predicted Weights')
plt.show()

