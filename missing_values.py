## Missing data

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

# Impute categorical values with most frequent
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent') # instantiate imputer
imp.fit(df_full[string_cols]) # fit imputer
df_full[string_cols] = imp.transform(df_full[string_cols])# impute values

# Impute numerical values using IterativeImputer
it_imp = IterativeImputer(random_state=0) # instantiate imputer
it_imp.fit(df_full[num_cols]) # fit imputer
df_full[num_cols] = it_imp.transform(df_full[num_cols])# impute values