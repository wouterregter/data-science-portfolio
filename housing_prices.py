## Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler


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
# drop SalePrice from df_full instead of df_train so it can still be used in EDA of df_train
df_full = df_full.drop('SalePrice', axis = 1)

## Inspect the train dataset

# Inspect target
sns.distplot(y_train)
plt.style.use('ggplot')
plt.show()

# Inspect structure
df_train.columns
#df_train.info()
df_train.describe()
df_train.head()

# Plot correlation heatmap
corrmat = df_train.corr()
f, ax = plt.subplots(figsize = (12, 9))
sns.heatmap(corrmat, vmax = 0.85);

# Print highest correlations with target
df_train.corr()["SalePrice"].sort_values(ascending = False).head(20)

# Log-transforming the target variable for normality
y_train = np.log1p(y_train)

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
rel = (df_full.isnull().sum()/df_full.values.shape[0])
rel.sort_values(ascending = False).head(20)

# Drop variables with > 20% missing
df_full = df_full.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'], axis = 1)

# Check for missing values
df_full.isna().sum().sum()

# Set the string and numerical columns
string_cols = df_full.select_dtypes(include='object').columns
num_cols = df_full.select_dtypes(exclude='object').columns
df_full_cols = df_full.columns

# Impute categorical values with most frequent
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent') # instantiate imputer
imp.fit(df_full[string_cols]) # fit imputer
df_full[string_cols] = imp.transform(df_full[string_cols])# impute values

# Impute numerical values using IterativeImputer
it_imp = IterativeImputer(random_state=0) # instantiate imputer
it_imp.fit(df_full[num_cols]) # fit imputer
df_full[num_cols] = it_imp.transform(df_full[num_cols])# impute values

# Put the array back in a df using the correct column names
df_full = pd.DataFrame(df_full, columns=df_full_cols)

## Final steps

# Scale and center
scaler = StandardScaler()
scaler.fit(df_full[num_cols])
df_full[num_cols] = scaler.transform(df_full[num_cols])

# Get dummies for categorical variables
df_full = pd.get_dummies(df_full)

# Reconstruct train and test
X_train = df_full[:n_train].values
X_test = df_full[n_train:].values


### Analysis


# OLS Regression
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# Ridge Regression
ridge = Ridge()
ridge.fit(X_train, y_train)
# y_pred = reg.predict(X_test)

my_submission = pd.DataFrame({'Id': id_test, 'SalePrice': y_pred})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)