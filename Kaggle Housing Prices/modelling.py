# Import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
import xgboost as xgb
import graphviz

# Preprocess the data using the defined preprocess function
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
drop_cols = ['GarageArea', 'GarageYrBlt', 'TotRmsAbvGrd', '1stFlrSF'] # High multicolinearity
makestr_cols = ['MSSubClass', 'OverallQual', 'OverallCond', 'MoSold', 'YrSold'] # Need to converted to string
X_train, y_train, X_test, df_columns = preprocess(training_set=df_train, test_set=df_test, log_y=True,
                                                  drop_cols=drop_cols, makestr_cols=makestr_cols)


## OLS Regression


# Instantiate, fit and evaluate
reg = LinearRegression()
reg_cv_scores = cross_val_score(reg, X_train, y_train, cv = 3)
reg_cv_scores.mean()


## Lasso Regression


# Set up the regressor pipeline
lasso_pipe = make_pipeline(RobustScaler(), Lasso())
# Set the paramater space for tuning
lasso_param_grid = {'lasso__alpha': [0.0006, 0.0005, 0.0004]}
# Tune alpha using GridSearchCV
lasso_gs = GridSearchCV(lasso_pipe, lasso_param_grid, cv=3)
# Fit to the training data
lasso_gs.fit(X_train, y_train)
# Evaluate performance
lasso_gs.cv_results_
lasso_gs.best_params_
lasso_gs.best_score_
# Make tuned pipeline for faster use
lasso_tuned = make_pipeline(RobustScaler(), Lasso(alpha = 0.0005))
lasso_tuned.fit(X_train, y_train)
lasso_tuned_score = cross_val_score(lasso_tuned,X_train,y_train,
                                    cv=3,scoring='neg_root_mean_squared_error').mean()
y_pred = lasso_tuned.predict(X_test)

# Visualize first 10 features (just for practice)
lasso_coef = lasso_tuned[1].coef_
plt.plot(range(10), lasso_coef[:10])
plt.xticks(range(10), df_columns[:10], rotation=60)
plt.margins(0.02)
plt.show()

# Log-transform y_train again for use in other models
y_train = np.log1p(y_train)


## Ridge Regression


# Set up the regressor pipeline
ridge_pipe = make_pipeline(RobustScaler(), Ridge())
# Set the paramater space for tuning
ridge_param_grid = {'ridge__alpha': [0.1, 1, 5, 10, 20, 60, 80, 100, 150, 180]}
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
ridge_tuned_score = cross_val_score(ridge_tuned, X_train, y_train, scoring='neg_root_mean_squared_error', cv=5).mean()
# Predict X_test
ridge_tuned.fit(X_train, y_train)
y_pred = ridge_tuned.predict(X_test)


## Decision Tree


# Instantiate, fit and evaluate
dt = DecisionTreeRegressor(max_depth=5)
cross_val_score(dt, X_train, y_train, scoring='explained_variance', cv=5).mean()
# Plotting
dt.fit(X_train, y_train)
plt.style.use('ggplot')
plot_tree(dt)
plt.show()


## Bagging

# Instantiate, fit and evaluate
br = BaggingRegressor(base_estimator=dt, n_estimators=200, oob_score=True, n_jobs=-1)
br.fit(X_train, y_train)
br.oob_score_


## Random Forest


# Instantiate the regressor
rf = RandomForestRegressor()
# Tune hyperparameters
rf_param_grid = {'n_estimators': [100,350,500],
                 'max_features': ['log2','auto','sqrt'],
                 'min_samples_leaf': [2,10,30]}
rf_search = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv=3, scoring='neg_root_mean_squared_error',
                         verbose=1, n_jobs=-1)
rf_search.fit(X_train, y_train)
rf_search.best_params_
rf_search.best_score_
# Instantiate tuned regressor for faster use
rf_tuned = RandomForestRegressor(n_estimators=500, max_features='auto', min_samples_leaf=2)
cross_val_score(rf_tuned, X_train, y_train, cv=3, scoring='neg_root_mean_squared_error').mean()


## Adaboost

# Instantiate, fit and evaluate
ada = AdaBoostRegressor(base_estimator=dt, n_estimators=180, random_state=1)
cross_val_score(ada, X_train, y_train, cv=3, scoring='neg_root_mean_squared_error').mean()


## Gradient Boosting

# Instantiate, fit and evaluate
gb = GradientBoostingRegressor(max_depth=5, subsample=0.9, max_features=0.75, n_estimators=250)
cross_val_score(gb, X_train, y_train, cv=3, scoring='neg_root_mean_squared_error').mean()


## Extreme Gradient Boosting


# Build the XGB pipeline
xgb_pipe = make_pipeline(RobustScaler(), xgb.XGBRegressor(objective='reg:squarederror'))
# Set the parameters for tuning
xgb_params = {
    'xgbregressor__learning_rate': np.arange(0.05, 1, 0.05),
    'xgbregressor__max_depth': np.arange(3, 10, 1),
    'xgbregressor__n_estimators': np.arange(50, 200, 50)
}
# Perform a Randomized Search
xgb_search = RandomizedSearchCV(estimator=xgb_pipe, param_distributions=xgb_params, n_iter=50, cv=3,
                                scoring='neg_root_mean_squared_error', verbose=1)
xgb_search.fit(X_train, y_train)
# Evaluate
xgb_search.best_score_
xgb_search.best_params_
# Instantiate tuned pipeline for faster use
xgb_tuned = make_pipeline(RobustScaler(), xgb.XGBRegressor(objective='reg:squarederror', learning_rate=0.1, max_depth=3,
                                                          n_estimators=150))
xgb_score = cross_val_score(xgb_tuned, X_train, y_train, cv=3, scoring='neg_root_mean_squared_error').mean()
xgb_tuned.fit(X_train, y_train)
# Predict the test set
y_pred = xgb_tuned.predict(X_test)

# Plot the first tree
xgb.plot_tree(xgb_tuned[1], num_trees=0)
plt.show()
# Plot feature importance
xgb.plot_importance(xgb_tuned[1], max_num_features=10)
plt.show()


## Support Vector Regression


# Instantiate, fit and evaluate
svr_pipe = make_pipeline(RobustScaler(), SVR())
svr_score = cross_val_score(svr_pipe, X_train, y_train, cv=4).mean()


## Submission to Kaggle


# Reverse log transform if y_train was log transformed
y_pred = np.expm1(y_pred)
# Write a submission file
id_test = df_test['Id'].values
my_submission = pd.DataFrame({'Id': id_test, 'SalePrice': y_pred})
my_submission.to_csv('submission.csv', index=False)

