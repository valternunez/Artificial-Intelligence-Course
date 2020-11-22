# Author: Valter Nunez
# Credits: Scikit Learn and Scott Robinson from Stack Abuse: https://stackabuse.com/linear-regression-in-python-with-scikit-learn/
# Dataset: Swedish Auto Insurance Dataset: https://www.math.muni.cz/~kolacek/docs/frvs/M7222/data/AutoInsurSweden.txt

#Imports
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


dataset = pd.read_csv('insurance.csv') #Import data from CSV file
print(dataset.shape) #Show how many rows and columns the dataset has
print(dataset.head()) #Show the first values of the dataset to understand how it is the dataset structured
print(dataset.describe()) #extra information to undestant dataset


X = dataset.iloc[:, :-1].values #allocation of values x
y = dataset.iloc[:, 1].values #allocation of values y


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 42) #Split testing and training sets. 30 Test, 70 Training. 42 as random state for LOLZ

regressor = LinearRegression() #Call linear model from scikit
regressor.fit(X_train, y_train) #Train model with training data.

y_pred = regressor.predict(X_test) #Do the predictions to see how good it was.

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}) #Get Actual vs Predicted
print(df) #Show it

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) #To compare with by-hand method, obtain RMSE
