import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# import the train data
df = pd.read_csv('train.csv')

# inspect the data
df.info()
df.describe()
df.head()

# set predictors and response
X = df.drop(['Id', 'SalePrice'], axis = 1).values
y = df['SalePrice'].values

# check for missing values
df.isna().sum().sum()

# encode dummies
X = pd.get_dummies(X)

# impute missing values
imp = SimpleImputer(missing_values=np.nan, strategy='mean') # instantiate imputer
imp.fit(X)
X = imp.transform(X) # impute means






