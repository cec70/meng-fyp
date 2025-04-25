import xarray as xr
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline
import joblib

# Load preprocessed data with lags
data_path = "Preprocessing/preprocessed_data_with_lags.nc"
data = xr.open_dataset(data_path)

# Convert to DataFrame for ML
features = [
    'u10', 't_polar',
    'u10_lag1', 'u10_lag7', 'u10_lag14',
    't_polar_lag1', 't_polar_lag7', 't_polar_lag14'
]
X = data[features].to_dataframe()
y = data['ssw_index'].to_series()  # Continuous SSW index as target

# Ensure time is the index
X.index = pd.to_datetime(data['time'].values)
y.index = pd.to_datetime(data['time'].values)

# Check for NaNs and drop if any
X = X.dropna()
y = y.loc[X.index]  # Align y with X after dropping NaNs

# Split data into training, validation, and test sets (time series, no shuffling)
train_val, test = train_test_split(X, test_size=0.2, shuffle=False)  # 80% train+val, 20% test
train, val = train_test_split(train_val, test_size=0.25, shuffle=False)  # 60% train, 20% val

X_train = train[features]
y_train = y.loc[X_train.index]
X_val = val[features]
y_val = y.loc[X_val.index]
X_test = test[features]
y_test = y.loc[X_test.index]

# Function to evaluate model across lead times
def evaluate_model(model, X_test, y_test, lead_times=[1, 7, 14, 20, 25, 30, 35, 40, 45, 47]):
    """
    Evaluate a predictive model's performance over multiple lead times using Mean Absolute Error (MAE).

    Parameters:
        model (object): Predictive model.
        X_test (pd.DataFrame): Test set features with a datetime index.
        y_test (pd.Series): True target values corresponding to X_test, with a datetime index.
        lead_times (list): A list of lead times (in days) to evaluate the model's performance (Defaults: [1, 7, 14, 20, 25, 30, 35, 40, 45, 47]).

    Returns:
        dict: A dictionary where keys are lead times and values are the corresponding MAE scores. 
    """

    mae_scores = {}

    # Iterate over the specified lead times
    for lead in lead_times:
        # Shift the y_test series by the lead time (negative shift for future prediction)
        y_test_shifted = y_test.shift(-lead, freq='D')
        
        # Drop NaN values resulting from the shift
        y_test_shifted = y_test_shifted.dropna()
        
        # Find the common index between X_test and the shifted y_test
        common_index = X_test.index.intersection(y_test_shifted.index)
        
        # Align X_test and y_test_shifted to the common index
        X_test_shifted = X_test.loc[common_index]
        y_test_shifted = y_test_shifted.loc[common_index]
        
        # Make predictions and calculate MAE
        if len(X_test_shifted) > 0:
            y_pred = model.predict(X_test_shifted)
            mae = mean_absolute_error(y_test_shifted, y_pred)
            mae_scores[lead] = mae
        else:
            # Assign NaN for the current lead time
            mae_scores[lead] = np.nan
    
    # Return the dictionary of MAE scores for each lead time
    return mae_scores

# Time-series cross-validation setup
tscv = TimeSeriesSplit(n_splits=5)

### 1. Random Forest Model ###

# Define the parameter grid for hyperparameter tuning
rf_param_grid = {
    'n_estimators': [100, 200], 
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2'] 
}

# Perform grid search with time-series cross-validation
rf_model = GridSearchCV(RandomForestRegressor(random_state=42), rf_param_grid, cv=tscv, 
                        scoring='neg_mean_absolute_error', n_jobs=-1)

# Fit the model on the training data
rf_model.fit(X_train, y_train)

# Retrieve the best estimator from the grid search
rf_best = rf_model.best_estimator_
print("Best Random Forest Parameters:", rf_model.best_params_)

# Evaluate the model's performance across lead times
rf_mae = evaluate_model(rf_best, X_test, y_test)
print("Random Forest MAE by Lead Time:", rf_mae)

### 2. Multilayer Perceptron (MLP) Model ###

# Define a pipeline for the MLP model with a scaler and MLPRegressor
mlp_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(random_state=42, max_iter=1000, early_stopping=True))
])

# Define the parameter grid for hyperparameter tuning
mlp_param_grid = {
    'mlp__hidden_layer_sizes': [(64, 32), (128, 64, 32), (128, 128)],
    'mlp__learning_rate_init': [0.001, 0.005],
    'mlp__alpha': [0.0001, 0.001]
}

# Perform grid search with time-series cross-validation
mlp_model = GridSearchCV(mlp_pipeline, mlp_param_grid, cv=tscv, 
                         scoring='neg_mean_absolute_error', n_jobs=-1)

# Fit the model on the training data
mlp_model.fit(X_train, y_train)

# Retrieve the best estimator from the grid search
mlp_best = mlp_model.best_estimator_
print("Best MLP Parameters:", mlp_model.best_params_)

# Evaluate the model's performance across lead times
mlp_mae = evaluate_model(mlp_best, X_test, y_test)
print("MLP MAE by Lead Time:", mlp_mae)

### 3. Gradient Boosting Model ###

# Define the parameter grid for hyperparameter tuning
gb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.05],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0]
}
# Perform grid search with time-series cross-validation
gb_model = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_param_grid, cv=tscv, 
                        scoring='neg_mean_absolute_error', n_jobs=-1)

# Fit the model on the training data
gb_model.fit(X_train, y_train)

# Retrieve the best estimator from the grid search
gb_best = gb_model.best_estimator_
print("Best Gradient Boosting Parameters:", gb_model.best_params_)

# Evaluate the model's performance across lead times
gb_mae = evaluate_model(gb_best, X_test, y_test)
print("Gradient Boosting MAE by Lead Time:", gb_mae)

### 4. Support Vector Machine (SVM) Model ###

# Define a pipeline for the SVM model with a scaler and SVR (Support Vector Regressor)
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVR(kernel='rbf'))
])

# Define the parameter grid for hyperparameter tuning
svm_param_grid = {
    'svm__C': [0.5, 1.0, 5.0],
    'svm__epsilon': [0.05, 0.1],
    'svm__gamma': ['scale', 0.1] 
}

# Perform grid search with time-series cross-validation
svm_model = GridSearchCV(svm_pipeline, svm_param_grid, cv=tscv, 
                         scoring='neg_mean_absolute_error', n_jobs=-1)

# Fit the model on the training data
svm_model.fit(X_train, y_train)

# Retrieve the best estimator from the grid search
svm_best = svm_model.best_estimator_
print("Best SVM Parameters:", svm_model.best_params_)

# Evaluate the model's performance across lead times
svm_mae = evaluate_model(svm_best, X_test, y_test)
print("SVM MAE by Lead Time:", svm_mae)

### 5. Linear Regression Model (using Ridge for regularization) ###

# Define the parameter grid for hyperparameter tuning for Ridge regression
lr_param_grid = {
    'alpha': [0.01, 0.1, 1.0],
    'fit_intercept': [True, False]
}

# Perform grid search with time-series cross-validation
lr_model = GridSearchCV(Ridge(), lr_param_grid, cv=tscv, 
                        scoring='neg_mean_absolute_error', n_jobs=-1)

# Fit the Ridge regression model on the training data
lr_model.fit(X_train, y_train)

# Retrieve the best estimator from the grid search
lr_best = lr_model.best_estimator_
print("Best Ridge Parameters:", lr_model.best_params_)

# Evaluate the Ridge regression model's performance across lead times
lr_mae = evaluate_model(lr_best, X_test, y_test)
print("Ridge MAE by Lead Time:", lr_mae)

# Set plotting style
plt.style.use('default')
sns.set_palette("muted")

# Set font properties
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

# Extract lead times from the Random Forest MAE dictionary
lead_times = list(rf_mae.keys())

# Generate a smooth range of lead times for interpolation
lead_times_smooth = np.linspace(min(lead_times), max(lead_times), 300)

# Smooth the MAE values for each model using cubic spline interpolation
rf_smooth = make_interp_spline(lead_times, [rf_mae[lt] for lt in lead_times])(lead_times_smooth)
mlp_smooth = make_interp_spline(lead_times, [mlp_mae[lt] for lt in lead_times])(lead_times_smooth)
gb_smooth = make_interp_spline(lead_times, [gb_mae[lt] for lt in lead_times])(lead_times_smooth)
svm_smooth = make_interp_spline(lead_times, [svm_mae[lt] for lt in lead_times])(lead_times_smooth)
lr_smooth = make_interp_spline(lead_times, [lr_mae[lt] for lt in lead_times])(lead_times_smooth)

# Plot the smoothed MAE values for each model
plt.plot(lead_times_smooth, rf_smooth, label='Random Forest')
plt.plot(lead_times_smooth, mlp_smooth, label='MLP')
plt.plot(lead_times_smooth, gb_smooth, label='Gradient Boosting')
plt.plot(lead_times_smooth, svm_smooth, label='SVM')
plt.plot(lead_times_smooth, lr_smooth, label='Linear Regression')

# Plot formatting
plt.xlabel('Lead Time (Days)')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.grid(False)
plt.show()

# Save best models as .plk files
best_models = {rf_best, mlp_best, gb_best, svm_best, lr_best}
for i, model in enumerate(best_models):
    model_name = ['RandomForest', 'MLP', 'GradientBoosting', 'SVM', 'LinearRegression'][i]
    model_path = f"Models/{model_name}.pkl"
    joblib.dump(model, model_path)

# Save X_test and y_test as .pkl files
joblib.dump(X_test, "Models/X_test.pkl")
joblib.dump(y_test, "Models/y_test.pkl")

# Save MAE results
mae_results = {'rf': rf_mae, 'mlp': mlp_mae, 'gb': gb_mae, 'svm': svm_mae, 'lr': lr_mae}
np.save("Models/mae_results.npy", mae_results)