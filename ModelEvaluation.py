import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wilcoxon, spearmanr
import joblib

# Set plotting style
plt.style.use('default')
sns.set_palette("muted")

# Set font properties
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

def scatter_pred_vs_actual(X_test, y_test, models, model_names, lead_time=7, save_path=None):
    """Scatter plot of predicted vs actual SSW index for a given lead time, with correlation coefficients."""

    # Shift the test data based on lead time and align indices
    y_test_shifted = y_test.shift(-lead_time, freq='D').dropna()
    common_index = X_test.index.intersection(y_test_shifted.index)
    X_test_shifted = X_test.loc[common_index]
    y_test_shifted = y_test_shifted.loc[common_index]

    # Create subplots for scatter plots of predicted vs actual values for each model
    fig, axes = plt.subplots(1, len(models), figsize=(12, 3.5), sharey=True)
    for ax, model, name in zip(axes, models, model_names):
        # Predict values using the model
        y_pred = model.predict(X_test_shifted)
        
        # Scatter plot of actual vs predicted values
        ax.scatter(y_test_shifted, y_pred, alpha=0.5)
        
        # Add a reference line (y = x) for perfect predictions
        ax.plot([y_test_shifted.min(), y_test_shifted.max()], 
                [y_test_shifted.min(), y_test_shifted.max()], 'r--', lw=2)
        
        # Set axis labels and title for the subplot
        ax.set_xlabel('Actual SSW Index')
        ax.set_title(name)

        # Calculate and display Pearson correlation coefficient
        corr, _ = pearsonr(y_test_shifted, y_pred)
        ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
        
    # Set the y-axis label for the first subplot
    axes[0].set_ylabel('Predicted SSW Index')
    
    # Plot formatting
    plt.suptitle(f'Scatter Plot: Predicted vs Actual (Lead Time = {lead_time} days)')
    plt.tight_layout()

    # Save Plot
    if save_path:
        plt.savefig(save_path)
    plt.show()

def time_series_forecast_vs_actual(X_test, y_test, models, model_names, lead_time=7, save_path=None, ssw_event_dates=None):
    """Time-series plot of predicted vs actual SSW index with optional SSW event markers."""

    # Shift the test data based on lead time and align indices
    y_test_shifted = y_test.shift(-lead_time, freq='D').dropna()
    common_index = X_test.index.intersection(y_test_shifted.index)
    X_test_shifted = X_test.loc[common_index]
    y_test_shifted = y_test_shifted.loc[common_index]

    # Create the time-series plot
    plt.figure(figsize=(10, 6))
    
    # Plot the actual SSW index values
    plt.plot(y_test_shifted.index, y_test_shifted, label='Actual', color='black', lw=1.75)
    
    # Plot the predicted SSW index values for each model
    for model, name in zip(models, model_names):
        y_pred = model.predict(X_test_shifted)
        plt.plot(y_test_shifted.index, y_pred, label=name, alpha=0.7, lw=1)

    # Mark known SSW event dates on the plot
    if ssw_event_dates:
        for event_date in ssw_event_dates:
            plt.axvline(pd.Timestamp(event_date), color='black', linestyle='--', alpha=0.5, lw=1, label='Known SSW Event Date' if event_date == ssw_event_dates[0] else "")

    # Plot formatting
    plt.xlabel('Time')
    plt.ylabel('SSW Index')
    plt.title(f'Time Series: Predicted vs Actual (Lead Time = {lead_time} days)')
    plt.legend()

    # Save Plot
    if save_path:
        plt.savefig(save_path)
    plt.show()

def residual_plot(X_test, y_test, models, model_names, lead_time=7, save_path=None):
    """Time-series plot of residuals (actual - predicted) for each model."""

    # Shift the test data based on lead time and align indices
    y_test_shifted = y_test.shift(-lead_time, freq='D').dropna()
    common_index = X_test.index.intersection(y_test_shifted.index)
    X_test_shifted = X_test.loc[common_index]
    y_test_shifted = y_test_shifted.loc[common_index]

    # Create the time-series plot
    plt.figure(figsize=(10, 6))

    # Plot the residuals for each model
    for model, name in zip(models, model_names):
        y_pred = model.predict(X_test_shifted)
        residuals = y_test_shifted - y_pred  # Actual - Predicted
        plt.plot(y_test_shifted.index, residuals, label=f'{name} Residuals', alpha=0.7)
    
    # Add a zero line for reference
    plt.axhline(0, color='black', linestyle='--', lw=1)

    # Plot formatting
    plt.xlabel('Time')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title(f'Residuals Over Time (Lead Time = {lead_time} days)')
    plt.legend()

    # Save Plot
    if save_path:
        plt.savefig(save_path)
    plt.show()

def statistical_evaluation_rf_vs_svm(X_test, y_test, rf_model, svm_model, mae_rf, mae_svm, lead_times, save_path=None):
    """Statistical comparison of Random Forest and SVM."""

    # Calculate the MAE difference between RF and SVM across lead times
    mae_diff = [mae_rf[lt] - mae_svm[lt] for lt in lead_times]
    median_diff = np.median(mae_diff)  # Compute the median of the MAE differences
    print(f"Median MAE Difference (RF - SVM): {median_diff:.4f}")

    # Perform the Wilcoxon Signed-Rank Test to compare the MAE distributions of RF and SVM
    wilcoxon_stat, wilcoxon_p = wilcoxon(list(mae_rf.values()), list(mae_svm.values()))
    print(f"Wilcoxon Signed-Rank Test: Statistic = {wilcoxon_stat:.2f}, p-value = {wilcoxon_p:.4f}")

    # Select a specific lead time for detailed error analysis
    lead_time = 47

    # Shift the test data based on the lead time and align indices
    y_test_shifted = y_test.shift(-lead_time, freq='D').dropna()
    common_index = X_test.index.intersection(y_test_shifted.index)
    X_test_shifted = X_test.loc[common_index]
    y_test_shifted = y_test_shifted.loc[common_index]

    # Predict values using RF and SVM models
    rf_pred = rf_model.predict(X_test_shifted)
    svm_pred = svm_model.predict(X_test_shifted)

    # Calculate absolute errors for RF and SVM predictions
    rf_errors = np.abs(rf_pred - y_test_shifted)
    svm_errors = np.abs(svm_pred - y_test_shifted)

    # Plot histograms of absolute errors for RF and SVM
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(rf_errors, bins=20, alpha=0.7, color='darkblue', label='RF Errors')
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title(f'Random Forest Error Distribution (Lead Time = {lead_time} days)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(svm_errors, bins=20, alpha=0.7, color='darkred', label='SVM Errors')
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title(f'SVM Error Distribution (Lead Time = {lead_time} days)')
    plt.legend()
    plt.tight_layout()

    # Save the histogram plot
    if save_path:
        plt.savefig(save_path.replace('.png', '_histograms.png'))
    plt.show()

    # Calculate Spearman correlation between actual and predicted values for RF and SVM
    spearman_corr_rf, spearman_p_rf = spearmanr(y_test_shifted, rf_pred)
    spearman_corr_svm, spearman_p_svm = spearmanr(y_test_shifted, svm_pred)
    print(f"Spearman Correlation (Lead Time = {lead_time} days):")
    print(f"  RF: Corr = {spearman_corr_rf:.3f}, p-value = {spearman_p_rf:.4f}")
    print(f"  SVM: Corr = {spearman_corr_svm:.3f}, p-value = {spearman_p_svm:.4f}")

if __name__ == "__main__":
    # Load saved data and models
    X_test = joblib.load('Models/X_test.pkl')
    y_test = joblib.load('Models/y_test.pkl')
    rf_best = joblib.load('Models/RandomForest.pkl')
    mlp_best = joblib.load('Models/MLP.pkl')
    gb_best = joblib.load('Models/GradientBoosting.pkl')
    svm_best = joblib.load('Models/SVM.pkl')
    ridge_best = joblib.load('Models/LinearRegression.pkl')
    mae_results = np.load('Models/mae_results.npy', allow_pickle=True).item()
    feature_importance = np.load('Models/feature_importance.npy', allow_pickle=True).item()

    # Define model and feature lists
    models = [rf_best, mlp_best, gb_best, svm_best, ridge_best]
    model_names = ['Random Forest', 'MLP', 'Gradient Boosting', 'SVM', 'Linear Regression']
    features = ['u10', 't_polar', 'u10_lag1', 'u10_lag7', 'u10_lag14', 
                't_polar_lag1', 't_polar_lag7', 't_polar_lag14']
    lead_times = [1, 7, 14, 20, 25, 30, 35, 40, 45, 47]
    mae_dicts = [mae_results['rf'], mae_results['mlp'], mae_results['gb'], 
                 mae_results['svm'], mae_results['lr']]

    # Generate visualizations
    scatter_pred_vs_actual(X_test, y_test, models, model_names, lead_time=7, save_path='Plots/scatter_plot_7days.png')
    time_series_forecast_vs_actual(X_test, y_test, models, model_names, lead_time=7, save_path='Plots/time_series_7days.png', ssw_event_dates=['2019-01-02', '2021-01-05'])
    residual_plot(X_test, y_test, models, model_names, lead_time=7, save_path='Plots/residuals_plot_7days.png')

    # Statistical evaluation of RF vs SVM
    statistical_evaluation_rf_vs_svm(X_test, y_test, rf_best, svm_best, mae_results['rf'], mae_results['svm'], lead_times, save_path='Plots/rf_vs_svm_stats.png')
    
    print("Visualizations generated and saved.")
