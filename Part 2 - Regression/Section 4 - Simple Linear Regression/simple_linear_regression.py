# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
#One third of training data-10 -> test data


#Fitting simple linear regression to the training set
#CREATING MODEL
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()      #instance 
regressor.fit(X_train, y_train)     #fits to training data

#PREDICTING TEST RESULTS USING MODEL
y_pred = regressor.predict(X_test) #build vector of predictions of dependent attr.

#visualizing training set results using CHARTS
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary VS Experience(Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#visualizing test set results using CHARTS
plt.scatter(X_test, y_test, color = 'red')
#same regression line even with test
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary VS Experience(Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()