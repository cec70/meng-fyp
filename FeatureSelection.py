import xarray as xr
from sklearn.ensemble import RandomForestClassifier

# Load data with SSW index and labels
data_path = "Preprocessing/preprocessed_data_with_labels.nc"  # Test subset
data = xr.open_dataset(data_path)

# Extract features and time
u10 = data['u10']
t_polar = data['t_polar']
ssw_labels = data['ssw_labels']  # Binary labels (1 = SSW, 0 = non-SSW)
time = data['time']

# Function to add lagged features
def add_lagged_features(data, variables, lags=[1, 7, 14]):
    """
    Add lagged features to the dataset for specified variables.
    
    Parameters:
    - data (xarray.Dataset): xarray.Dataset with variables (e.g., u10, t_polar).
    - variables (list): List of variable names to lag (e.g., ['u10', 't_polar']).
    - lags (list): List of lag days (e.g., [1, 7, 14]).
    
    Returns:
    - xarray.Dataset with original and lagged features.
    """

    lagged_data = data.copy()
    for var in variables:
        for lag in lags:
            # Shift the variable by the specified lag and rename it
            lagged_var = data[var].shift(time=lag).rename(f"{var}_lag{lag}")
            # Add the lagged variable to the dataset
            lagged_data = lagged_data.assign({f"{var}_lag{lag}": lagged_var})
    return lagged_data

# Add lagged features for u10 and t_polar
variables_to_lag = ['u10', 't_polar']
lags = [1, 7, 14]  # Days to look back (1 day, 1 week, 2 weeks)
data_with_lags = add_lagged_features(data, variables_to_lag, lags)

# Drop NaNs introduced by shifting (earliest days will have missing lags)
data_with_lags = data_with_lags.dropna(dim='time', how='any')

# Convert to DataFrame for ML
features = [
    'u10', 't_polar',
    'u10_lag1', 'u10_lag7', 'u10_lag14',
    't_polar_lag1', 't_polar_lag7', 't_polar_lag14'
]
X = data_with_lags[features].to_dataframe()
y = data_with_lags['ssw_labels'].to_series()

# Check label distribution
print("\nSSW Label Distribution:")
print(y.value_counts())

# Feature Importance with Random Forest
def assess_feature_importance(X, y):
    """
    Use Random Forest to assess feature importance.
    
    Parameters:
    - X: DataFrame with features.
    - y: Series with labels.
    
    Returns:
    - Feature importance dictionary.
    """

    # Initialize and train the Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Extract feature importances and map them to feature names
    importance = dict(zip(X.columns, rf.feature_importances_))
    return importance

# Run feature importance
importance = assess_feature_importance(X, y)
print("\nFeature Importance:")
for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    print(f"{feature}: {score:.4f}")

# Save dataset with lagged features
output_path = "Preprocessing/preprocessed_data_with_lags.nc"
data_with_lags.to_netcdf(output_path)
print(f"Saved data with lagged features to {output_path}")

# Print sample for verification
print("\nSample features (first 5 rows):")
print(X.head())
print("\nSample labels (first 5):")
print(y.head())