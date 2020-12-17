import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def preprocess(training_set, test_set, log_y=False, drop_cols=[], num_conv_cols=[]):
    # Drop the id columns
    df_train = training_set.drop('Id', axis=1)
    df_test = test_set.drop('Id', axis=1)
    # Save the length of the train and test dfs for later reconstruction
    n_train = df_train.shape[0]
    n_test = df_test.shape[0]
    y_train = df_train['SalePrice'] # set y_train so it can be dropped
    df_full = pd.concat([df_train, df_test], axis=0)  # merge train and test to df_full
    # drop SalePrice from df_full instead of df_train so it can still be used in EDA of df_train
    df_full = df_full.drop('SalePrice', axis=1)
    # Log-transforming the target variable for normality
    if log_y:
        y_train = np.log1p(y_train)
    # Dropping variables in drop_list
    df_full = df_full.drop(drop_list, axis=1)
    # Convert some categorical features to strings that are numerical in the data
    for col in num_conv_cols:
        df_full[col] = df_full[col].apply(str)
    # Missing Data
    rel = (df_full.isnull().sum() / df_full.values.shape[0])
    morethan20 = rel[rel > 0.20].index
    # Drop variables with > 20% missing values
    df_full = df_full.drop(morethan20, axis=1)
    df_full_cols = df_full.columns
    # Impute categorical values with most frequent
    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')  # instantiate imputer
    imp.fit(df_full)  # fit imputer
    df_full = imp.transform(df_full)  # impute values
    df_full = pd.DataFrame(df_full, columns=df_full_cols)
    # Get dummies for categorical variables
    df_full = pd.get_dummies(df_full)
    # Reconstruct train and test
    X_train = df_full[:n_train].values
    X_test = df_full[n_train:].values
    return X_train, y_train, X_test