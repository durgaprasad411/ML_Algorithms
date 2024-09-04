import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import pickle

# Loading the dataset
x = pd.read_csv('C:\Durgaprasad\DATASETS\Student_Performance.csv')
print(x.head())

# Checking the column names
print(x.columns)

# Checking for null data
print(x.isnull().sum())

# Splitting the data into independent (X) and dependent (y) variables
y = x['Performance Index']
X = x['Previous Scores'].values.reshape(-1, 1)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

# Creating and training the Linear Regression model
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
plt.title('Predicted vs Actual Performance Index')
plt.xlabel('Actual Performance Index')
plt.ylabel('Predicted Performance Index')
plt.show()

##saving the model
with open('Linear_model_pic.pkl','wb') as file:
    pickle.dump(model,file)
with open('Linear_model_pic.pkl','rb') as file:
    saved_model=pickle.load(file)

a=int(input("Enter ther previous score: "))
Assum_res={99:92,51:45,96:85,77:74}
pred_user = saved_model.predict([[a]])
print("Hi")
print("Performance index for the {0} is {1}".format(a,pred_user))
pre_res=Assum_res[a]
print(pre_res)
print(mean_squared_error([pre_res],[pred_user]))
