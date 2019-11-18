# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'D:\Udemy Machine Learning A-Z\Part 1 - Data Preprocessing\Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = 'np.nan', strategy = 'mean', axis = 0)  #mean is the default value for strategy
imputer = imputer.fit(X[:, 1:3]) #columns with missing data
X[:, 1:3] = imputer.transform(X[:, 1:3]) #replaces
 
        #scikit learn -