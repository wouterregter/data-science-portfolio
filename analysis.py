

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score, GridSearchCV

### Analysis


# OLS Regression
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
reg_cv_scores = cross_val_score(reg, X_train, y_train, cv = 3)

# Ridge Regression
ridge = Ridge()
ridge.fit(X_train, y_train)
ridge_params = {'alpha':[0.1, 10, 60, 80, 100, 150, 180,200, 230, 250]}
ridge_cv = GridSearchCV(ridge, param_grid=ridge_params, cv=5)
ridge_cv.fit(X_train,y_train)
ridge_cv.best_score_


# y_pred = reg.predict(X_test)


my_submission = pd.DataFrame({'Id': id_test, 'SalePrice': y_pred})
# you could use any filename. We choose submission here
my_submission.to_csv('submission1.csv', index=False)

lasso_params = {'alpha':[0.02, 0.024, 0.025, 0.026, 0.03]}