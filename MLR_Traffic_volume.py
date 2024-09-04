##importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import pickle
import joblib

x=pd.read_csv('C:\Durgaprasad\DATASETS\Metro_Interstate_Traffic_Volume.csv')
print(x.head())
##Column name
print(x.columns)
##Dropping unwanted columns
dt=x.drop(['holiday','clouds_all','weather_main','weather_description','date_time'],axis=1)
print(dt)
##checking no.of rows and columns in data
rows,colum=dt.shape
print("The no.of rows: ",rows)
print("The no.of columns: ",colum)
##checking fot null data
print(dt.isnull().sum())
##Splitting the data into independent and dependent variables
y=dt['traffic_volume']
x=dt[['temp','rain_1h','snow_1h']]
print(y.head())
print(" ******** ")
print(x)

## splitting the data into the training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)



##Taking the model
model=LinearRegression()
##Training the model
model.fit(X_train,y_train)

print(model)

## Predicting the model by test data
y_pred=model.predict(X_test)

##performance Evaluation
# Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Root Mean Squared Error
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

# R-squared score
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")


plt.scatter(y_test, y_pred)
plt.xlabel("Actual Traffic Volume")
plt.ylabel("Predicted Traffic Volume")
plt.title("Actual vs Predicted Traffic Volume")
plt.show()


####saving the model
with open('mlr_model.pkl','wb') as file:
    pickle.dump(model,file)
with open('mlr_model.pkl','rb') as file:
    saved_model=pickle.load(file)





user_data = pd.DataFrame({
    'temp': [285.0, 290.0, 275.0],     # Temperatures in Kelvin
    'rain_1h': [0.0, 0.2, 0.0],        # Rainfall in the last hour (in mm)
    'snow_1h': [0.0, 0.0, 1.0]         # Snowfall in the last hour (in mm)
})

# Predicting the traffic volume using the saved model
user_predictions = saved_model.predict(user_data)
print("Predicted Traffic Volumes:", user_predictions)

# If actual traffic volumes for the user data are available
actual_user_traffic = [2000, 2500, 1500]

# Calculating evaluation metrics for user data
user_mse = mean_squared_error(actual_user_traffic, user_predictions)
user_rmse = np.sqrt(user_mse)
user_r2 = r2_score(actual_user_traffic, user_predictions)

print(f"User Data Mean Squared Error: {user_mse}")
print(f"User Data Root Mean Squared Error: {user_rmse}")
print(f"User Data R-squared: {user_r2}")
