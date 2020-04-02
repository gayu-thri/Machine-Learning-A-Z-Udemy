# Data Preprocessing 

# Importing the libraries
import numpy as np #for mathematical func.
import matplotlib.pyplot as plt
import pandas as pd #for import/management of datasets

# Importing the dataset
dataset = pd.read_csv(r'D:\Udemy Machine Learning A-Z\Part 1 - Data Preprocessing\Data.csv')
#r to produce a raw string in read_csv
        
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

# Encoding categorical data

# Encoding the Independent Variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
# Encoding Y data
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)
    
    
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))