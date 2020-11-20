## Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge

### EDA

## Import the data and merge train and test for preprocessing

# Import import train and test data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# Save train and test ids so they can be dropped
id_train = df_train["Id"]
id_test = df_test["Id"]

# Drop the id columns
df_train = df_train.drop("Id", axis = 1)
df_test = df_test.drop("Id", axis = 1)

# Save the length of the train and test dfs for later reconstruction
n_train = df_train.shape[0]
n_test = df_test.shape[0]

y_train = df_train["SalePrice"] # set y_train so it can be dropped
df_full = pd.concat([df_train,df_test], axis = 0) # merge train and test to df_full
# drop SalePrice from df_full so it can still be used in EDA of df_train
df_full = df_full.drop('SalePrice', axis = 1)

## Inspect the dataset

# Inspect structure
df_train.columns
df_train.info()
df_train.describe()
df_train.head()

# Plot correlation heatmap
corrmat = df_train.corr()
f, ax = plt.subplots(figsize = (12, 9))
sns.heatmap(corrmat, vmax = 0.85);
plt.show()

# Dropping variables due to high multicollinearity
# Dropping GarageCars because of high multicollinearity with GarageArea
df_full = df_full.drop("GarageCars", axis = 1)
# Dropping GarageYrBlt because of high multicollinearity with YearBuilt
df_full = df_full.drop("GarageYrBlt", axis = 1)
# Dropping TotRmsAbvGrd because of high multicollinearity with GrLivArea
df_full = df_full.drop("TotRmsAbvGrd", axis = 1)
# Dropping 1stFlrSF because of high multicollinearity with TotalBsmtSF
df_full = df_full.drop("1stFlrSF", axis = 1)

# Convert some categorical features to strings that are numerical in the data
df_full['MSSubClass'] = df_full['MSSubClass'].apply(str)
df_full['OverallQual'] = df_full['OverallQual'].apply(str)
df_full['OverallCond'] = df_full['OverallCond'].apply(str)
df_full['MoSold'] = df_full['MoSold'].apply(str)
df_full['YrSold'] = df_full['YrSold'].apply(str)

## Missing data

# Check for missing values
df_full.isna().sum().sum()
# Impute missing values
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent') # instantiate imputer
imp.fit(df_full) # fit imputer
df_full = imp.transform(df_full) # impute means
df_full = pd.DataFrame(df_full)

## Final steps

# Get dummies for categorical variables
df_full = pd.get_dummies(df_full)

# Reconstruct train and test
X_train = df_full[:n_train].values
X_test = df_full[n_train:].values

### Analysis

#OLS Regression
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

#
ridge = Ridge()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

my_submission = pd.DataFrame({'Id': id_test, 'SalePrice': y_pred})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)