import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import joblib
import pickle
data=pd.read_csv('C:\Durgaprasad\DATASETS\huge_1M_titanic.csv')

print(data.head())
print(data.shape)


data.drop(['Cabin'], axis=1, inplace=True)

print(data.columns)

##converting categorical data into the numerical data(oneHotEncoding)
data=pd.get_dummies(data,columns=['Sex','Embarked'],drop_first=True)
print(data)
print(data.columns)

##selecting features and target varibables
x=data[['Pclass','Age','SibSp','Parch','Fare','Sex_male','Embarked_Q','Embarked_S']]
y=data['Survived']

##splitting the data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)


##Initializing the model
model=DecisionTreeClassifier(criterion='gini',max_depth=8,random_state=5)

##Training the model
model.fit(x_train,y_train)

##Predicting
y_pred=model.predict(x_test)

##Evaluating the model

accu=accuracy_score(y_test,y_pred)
print("Accuracy is : {}%".format(accu*100.0))

##Confusion matrix

import seaborn as sns
cfm = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix using seaborn
sns.heatmap(cfm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

##plotting the decision tree
plt.figure(figsize=(10,10))
plot_tree(
    model,
    feature_names=x.columns,
    class_names=['Not Survived', 'Survived'],
    filled=True,
    rounded=True,
    fontsize=12
)
plt.title("Decision Tree for Titanic Survival Prediction")
plt.show()

import pickle
from sklearn.metrics import accuracy_score

# Save the model
with open('Decisio_tree.pkl', 'wb') as file:
    pickle.dump(model, file)

# Load the saved model
with open('Decisio_tree.pkl', 'rb') as file:
    saved_model = pickle.load(file)

##['Pclass','Age','SibSp','Parch','Fare','Sex_male','Embarked_Q','Embarked_S','result']
user_data = [
    [3, 30, 1, 0, 1, 0, 1, 0, 1],  # Target: Survived (1)
    [1, 22, 1, 0, 2, 1, 0, 1, 1],  # Target: Survived (0)
    [3, 20, 0, 0, 1, 0, 1, 0, 1],  # Target: Survived (1)
    [2, 20, 1, 0, 1, 0, 1, 0, 1],  # Target: Survived (1)
    [1, 24, 1, 0, 1, 1, 0, 1, 0],  # Target: Not Survived (0)
    [0, 20, 0, 0, 1, 0, 1, 0, 1],   # Target: Survived (1)
    [3, 25, 0, 0, 1, 0, 0, 1, 1],   #Targe:  survived(1)
]

X_test = [row[:-1] for row in user_data]
y_true = [row[-1] for row in user_data]

# Predict the output using the saved model
y_pred = saved_model.predict(X_test)

print("Hi")
print(y_pred)

# Compare the predicted output with the actual output
for i, (pred, true) in enumerate(zip(y_pred, y_true)):
    result = "survived" if pred == 1 else "not survived"
    actual_result = "survived" if true == 1 else "not survived"
    print(f"Sample {i+1}: Predicted: {result}, Actual: {actual_result}")

# Calculate the accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Model accuracy: {accuracy:.2f}")
