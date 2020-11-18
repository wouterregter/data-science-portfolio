import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

# import the train data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# inspect the data
df_train.info()
df_train.describe()
df_train.head()

# get dummies for categorical variables
df_train = pd.get_dummies(df)

# set predictors and response
X_train = df_train.drop(['Id', 'SalePrice'], axis = 1).values
y_train = df_train['SalePrice'].values
X_test = df_test.drop(['Id', 'SalePrice'], axis = 1).values
y_test = df_test['SalePrice'].values

# check for missing values
df.isna().sum().sum()

# impute missing values
imp = SimpleImputer(missing_values=np.nan, strategy='mean') # instantiate imputer
imp.fit(X) # fit imputer
X_train = imp.transform(X_train) # impute means

reg = LinearRegression()
reg.fit()







