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
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)  #‘NaN’ was replaced by ‘np.nan’

''' columns with missing data '''
missingvalues = missingvalues.fit(X[:, 1:3])    
'''1:3 => 3 not included'''
''' replaces missing data with mean '''
X[:, 1:3]=missingvalues.transform(X[:, 1:3])

        #scikit learn -