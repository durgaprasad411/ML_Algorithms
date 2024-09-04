import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
data=pd.read_csv('C:\Durgaprasad\DATASETS\housing_california.csv')
print(data.head())


##checking no.of columns and rows
print(data)
rw,cl=data.shape
print("The no.of rows in the given data is: ",rw)
print("The no.of columns in the data: ",cl)


##Checking for missing data
print(data.isnull().sum())

##checking for the missing colum
# Print rows where 'total_bedrooms' has null values
null_rows = data[data['total_bedrooms'].isnull()]
# Display the rows
print(null_rows['total_bedrooms'])

##Dealing with null values
#Dropping null rows
cl_data=data.dropna(subset=['total_bedrooms'])
print(cl_data)


##Taking independent and the dependent variables
x=cl_data['total_rooms']
y=cl_data['median_house_value']
print(y)
##Again checking null data
print(x.isnull().sum())
print(y.isnull().sum())

##splitting the data into training and testing data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

x_train = x_train.values.reshape(-1, 1)
x_test = x_test.values.reshape(-1, 1)


#Taking the model
model=LinearRegression()
model.fit(x_train,y_train)

#Predicting the test set results
y_pred=model.predict(x_test)

# Visualizing the Training set results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, model.predict(x_train), color = 'blue')
plt.title('Housing Price vs no.of rooms (Training set)')
plt.xlabel('No.of rooms')
plt.ylabel('Housing Price')
plt.show()

# Visualizing the Test set results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, model.predict(x_train), color = 'blue')
plt.title('Housing Price vs no.of rooms (Test set)')
plt.xlabel('No.of rooms')
plt.ylabel('Housing Price')
plt.show()


from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

# Calculating and printing MSE
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: ", mse)

# Calculating and printing RMSE
rmse = sqrt(mse)
print("Root Mean Squared Error: ", rmse)




