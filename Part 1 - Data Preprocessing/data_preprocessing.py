# Data Preprocessing 

# Importing the libraries
import numpy as np #for mathematical func.
import matplotlib.pyplot as plt
import pandas as pd #for import/management of datasets

# Importing the dataset
dataset = pd.read_csv(r'D:\Udemy Machine Learning A-Z\Part 1 - Data Preprocessing\Data.csv')
        #r to produce a raw string
        
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

