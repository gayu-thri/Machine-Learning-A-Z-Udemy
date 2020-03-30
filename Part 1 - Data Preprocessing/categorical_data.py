# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'D:\Github\Machine-Learning-A-Z-Udemy\Part 1 - Data Preprocessing\Data.csv')

''' all rows;all columns except last one ~ dependent attributes '''
X = dataset.iloc[:, :-1].values   
''' last column alone ~ independent attribute '''
y = dataset.iloc[:, 3].values   

# Taking care of missing data
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)  #‘NaN’ was replaced by ‘np.nan’

''' columns with missing data '''
missingvalues = missingvalues.fit(X[:, 1:3])    
'''1:3 => 3 not included'''
''' replaces missing data with mean '''
X[:, 1:3]=missingvalues.transform(X[:, 1:3])

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
# Encoding Y data
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)

'''
#OLDER VERSION
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
#takes all entries of first column
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
'''