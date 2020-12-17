# Import Libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import RobustScaler

# Preprocess the data using the defined preprocess function
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
drop_cols = ['GarageArea', 'GarageYrBlt', 'TotRmsAbvGrd', '1stFlrSF'] # High multicolinearity
num_conv_cols = ['MSSubClass', 'OverallQual', 'OverallCond', 'MoSold', 'YrSold']
X_train, y_train, X_test = preprocess(training_set=df_train, test_set=df_test, log_y=False,
                                      drop_cols=drop_list, num_conv_cols=num_conv_cols)

# OLS Regression
reg = LinearRegression()
reg_cv_scores = cross_val_score(reg, X_train, y_train, cv = 3)

# Ridge Regression
ridge = Ridge()
ridge_pipe = make_pipeline(RobustScaler(), ridge)
ridge_cv_scores = cross_val_score(ridge_pipe, X_train, y_train, cv = 3)

# Tune hyperparams
ridge = Ridge()
ridge_params = {'alpha':[0.1, 1, 5, 10, 20, 60, 80, 100, 150, 180]}
search = GridSearchCV(ridge, param_grid=ridge_params, cv=3)
search.fit(X_train, y_train)
search.best_params_
search.best_score_
search.get_params()

# Get score with
ridge = Ridge(alpha=5)
pipe = make_pipeline(RobustScaler(), ridge)
cross_val_score(pipe, X_train, y_train, cv = 3)


my_submission = pd.DataFrame({'Id': id_test, 'SalePrice': y_pred})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)

lasso_params = {'alpha':[0.02, 0.024, 0.025, 0.026, 0.03]}