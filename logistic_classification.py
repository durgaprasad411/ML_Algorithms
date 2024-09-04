import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import joblib

import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(r'C:\Durgaprasad\DATASETS\adult.csv')

# Display the first few rows of the dataset
print(data.head())

rq = data[data.isin(['?']).any(axis=1)]##column wise checking
print("\nRows containing '?':")
print(rq)

# Drop these rows from the dataset
data_cleaned = data.drop(rq.index)

print(data_cleaned.isnull().sum())

print(data_cleaned.shape)

print("Before cleaning : ",data.shape)
print("After cleaning: ",data_cleaned.shape)
data=data_cleaned

# Handle missing values
imputer = SimpleImputer(strategy='most_frequent')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Separate features and target
X = data_imputed.drop('income', axis=1)##column wise drop
y = data_imputed['income']

# Label encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Transforms '<=50K' to 0 and '>50K' to 1
print("Target data after label Encoding: ")
print(y)

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

print("The Categorial data columns are: ")
print(categorical_cols)
print()
print("The numerical data columns are : ")
print(numerical_cols)
print()
# Preprocess the categorical and numerical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)

# Apply preprocessing
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Create and train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

print('The test values are  ',y_test)
print("The predicted values are : ",y_pred)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)


print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)





