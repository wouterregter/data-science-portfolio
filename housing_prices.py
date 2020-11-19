import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

## import data and merge train and test for preprocessing
df_train = pd.read_csv('train.csv') # import train data into df
df_test = pd.read_csv('test.csv') # import test data into df

# save train and test ids so they can be dropped
id_train = df_train["Id"]
id_test = df_test["Id"]

# drop the id columns
df_train = df_train.drop("Id", axis = 1)
df_test = df_test.drop("Id", axis = 1)

# save the length of the train and test dfs for later reconstruction
n_train = df_train.shape[0]
n_test = df_test.shape[0]

y_train = df_train["SalePrice"] # set y_train so it can be dropped
df_full = pd.concat([df_train,df_test], axis = 0) # merge train and test to df_full

## inspect the dataset
df_full.columns
df_full.info()
df_full.iloc.describe()
df_full.head()

## inspect features

# MSSubClass
df_full['MSSubClass'].value_counts()
# convert to strings because the variable is categorical
df_full['MSSubClass'] = df_full['MSSubClass'].apply(str)

# MSZoning
df_full['MSZoning'].value_counts()

# LotFrontage
df_full['LotFrontage'].describe()

# LotArea
df_full['LotArea'].describe()

# Street
df_full['LotArea'].describe()


# check for missing values
df.isna().sum().sum()

# impute missing values
imp = SimpleImputer(missing_values=np.nan, strategy='mean') # instantiate imputer
imp.fit(X) # fit imputer
X_train = imp.transform(X_train) # impute means

reg = LinearRegression()
reg.fit()

# get dummies for categorical variables
df_train = pd.get_dummies(df)

# set predictors and response
X_train = df_train.drop(['Id', 'SalePrice'], axis = 1).values
y_train = df_train['SalePrice'].values
X_test = df_test.drop(['Id', 'SalePrice'], axis = 1).values
y_test = df_test['SalePrice'].values





