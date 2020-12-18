# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import xgboost as xgb


# Preprocess the data using the defined preprocess function
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
drop_cols = ['GarageArea', 'GarageYrBlt', 'TotRmsAbvGrd', '1stFlrSF'] # High multicolinearity
makestr_cols = ['MSSubClass', 'OverallQual', 'OverallCond', 'MoSold', 'YrSold'] # Need to be string
X_train, y_train, X_test, df_columns = preprocess(training_set=df_train, test_set=df_test, log_y=True,
                                                  drop_cols=drop_cols, makestr_cols=makestr_cols)

## OLS Regression

reg = LinearRegression()
reg_cv_scores = cross_val_score(reg, X_train, y_train, cv = 3)
reg_cv_scores.mean()

## Lasso Regression

# Reverse log transform of y_train
y_train = np.expm1(y_train)
# Set up the regressor pipeline
lasso_pipe = make_pipeline(RobustScaler(), Lasso())
# Set the paramater space for tuning
lasso_params = {'lasso__alpha': [100, 130, 160]}
# Tune alpha using GridSearchCV
lasso_gs = GridSearchCV(lasso_pipe, lasso_params, cv=3)
# Fit to the training data
lasso_gs.fit(X_train, y_train)
# Evaluate performance
lasso_gs.cv_results_
lasso_gs.best_params_
lasso_gs.best_score_
# Make tuned pipeline for faster use
lasso_tuned = make_pipeline(RobustScaler(), Lasso(alpha = 100))
lasso_tuned_score = cross_val_score(lasso_tuned,X_train,y_train,cv=3).mean()

# Visualize first 10 features (just for practice)
lasso_tuned.fit(X_train, y_train)
lasso_coef = lasso_tuned[1].coef_
plt.plot(range(10), lasso_coef[:10])
plt.xticks(range(10), df_columns[:10], rotation=60)
plt.margins(0.02)
plt.show()

# Log transform y_train again for use in other models
y_train = np.log1p(y_train)

## Ridge Regression

# Set up the regressor pipeline
ridge_pipe = make_pipeline(RobustScaler(), Ridge())
# Set the paramater space for tuning
ridge_params = {'ridge__alpha': [0.1, 1, 5, 10, 20, 60, 80, 100, 150, 180]}
# Tune alpha using GridSearchCV
ridge_gs = GridSearchCV(ridge_pipe, ridge_params, cv=3)
# Fit to the training data
ridge_gs.fit(X_train, y_train)
# Evaluate performance
ridge_gs.cv_results_
ridge_gs.best_params_
ridge_gs.best_score_
# Define tuned pipeline for faster use
ridge_tuned = make_pipeline(RobustScaler(), Ridge(alpha=5))
ridge_tuned_score = cross_val_score(ridge_tuned, X_train, y_train, cv=5).mean()
# Predict X_test
ridge_tuned.fit(X_train, y_train)
y_pred = ridge_tuned.predict(X_test)

## Decision Tree
dt = DecisionTreeRegressor(max_depth=5)
cross_val_score(dt, X_train, y_train, scoring='explained_variance', cv=5).mean()

dt.fit(X_train, y_train)
plt.style.use('ggplot')
plot_tree(dt)
plt.show()

## Extreme Gradient Boosting

xgb_reg = xgb.XGBRegressor(max_depth=3, n_estimators=200)
xgb_reg_score = cross_val_score(xg_reg, X_train, y_train, cv=3).mean()

## Support Vector Regression

svr_pipe = make_pipeline(RobustScaler(), SVR())
svr_score = cross_val_score(svr_pipe, X_train, y_train, cv=4).mean()


# Plot the first tree
xgb_reg.fit(X_train, y_train)
xgb.plot_tree(xgb_reg, num_trees=20)
plt.show()



# Reverse log transform if y_train was log transformed
y_pred = np.expm1(y_pred)
# Write a submission file
my_submission = pd.DataFrame({'Id': id_test, 'SalePrice': y_pred})
my_submission.to_csv('submission.csv', index=False)

