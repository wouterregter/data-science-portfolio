import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def preprocess(training_set, test_set, log_y=False, drop_cols=[], makestr_cols=[]):
    # Drop the id columns
    df_train = training_set.drop('Id', axis=1)
    df_test = test_set.drop('Id', axis=1)
    # Deleting outliers
    df_train = df_train.drop(df_train[(df_train.GrLivArea > 4000) & (df_train.SalePrice < 200000)].index)
    # Save the length of the train and test dfs for later reconstruction
    n_train = df_train.shape[0]
    n_test = df_test.shape[0]
    # Set target to y_train
    y_train = df_train['SalePrice']
    df_full = pd.concat([df_train, df_test], axis=0)
    # Drop target
    df_full = df_full.drop('SalePrice', axis=1)
    # Log-transforming the target variable for normality
    if log_y:
        y_train = np.log1p(y_train)
    # Dropping variables in drop_list
    df_full = df_full.drop(drop_cols, axis=1)
    # Convert some categorical features to strings that are numerical in the data
    for col in makestr_cols:
        df_full[col] = df_full[col].apply(str)
    # Changing NA's of features that are described in the data information as 'None'
    ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
    for feature in none_features:
        df_full[feature] = df_full[feature].fillna("None")
    # Set categorical and non categorical columns
    df_full_cols = df_full.columns
    categorical_columns = df_full.columns[df_full.dtypes == object]
    non_categorical_columns = df_full.columns[~(df_full.dtypes == object)]
    # Impute categorical values with most frequent
    imp1 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')  # instantiate imputer
    df_full[categorical_columns] = imp1.fit_transform(df_full[categorical_columns])  # impute values
    # Impute numerical values with median
    imp2 = SimpleImputer(missing_values=np.nan, strategy='median')  # instantiate imputer
    df_full[non_categorical_columns] = imp2.fit_transform(df_full[non_categorical_columns])  # impute values
    # Put the results back into a dataframe
    df_full = pd.DataFrame(df_full, columns=df_full_cols)
    # Get dummies for categorical variables
    df_full = pd.get_dummies(df_full)
    df_columns = df_full.columns
    # Reconstruct train and test
    X_train = df_full[:n_train].values
    X_test = df_full[n_train:].values
    return X_train, y_train, X_test, df_columns