import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from PIL import Image
from sklearn.inspection import permutation_importance
from lime import lime_tabular 

# Set plotting style
plt.style.use('default')
sns.set_palette("muted")

# Set font properties
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

# Load saved data and models
X_test = joblib.load('Models/X_test.pkl')
y_test = joblib.load('Models/y_test.pkl')
rf_best = joblib.load('Models/RandomForest.pkl')
mlp_best = joblib.load('Models/MLP.pkl')
gb_best = joblib.load('Models/GradientBoosting.pkl')
svm_best = joblib.load('Models/SVM.pkl')
ridge_best = joblib.load('Models/LinearRegression.pkl')

# Define model and feature lists
models = [rf_best, mlp_best, gb_best, svm_best, ridge_best]
model_names = ['Random Forest', 'MLP', 'Gradient Boosting', 'SVM', 'Linear Regression']
features = ['u10', 't_polar', 'u10_lag1', 'u10_lag7', 'u10_lag14', 
            't_polar_lag1', 't_polar_lag7', 't_polar_lag14']

# Function to shift data for a given lead time
def get_shifted_data(X_test, y_test, lead_time=7):
    y_test_shifted = y_test.shift(-lead_time, freq='D').dropna()
    common_index = X_test.index.intersection(y_test_shifted.index)
    X_test_shifted = X_test.loc[common_index]
    y_test_shifted = y_test_shifted.loc[common_index]
    return X_test_shifted, y_test_shifted

# Select a lead time for analysis
lead_time = 7
X_test_shifted, y_test_shifted = get_shifted_data(X_test, y_test, lead_time)

# Subset of data for faster computation
X_test_sample = X_test_shifted.sample(100, random_state=42) if len(X_test_shifted) > 100 else X_test_shifted
y_test_sample = y_test_shifted.loc[X_test_sample.index]

### 1. SHAP Analysis ###
def shap_analysis(models, model_names, X_test_sample, features, save_path=None, lead_time=None):
    """Perform SHAP analysis for all models and generate summary plots."""

    print("Running SHAP analysis for all models...")
    
    shap_values_dict = {}
    summary_images = []
    
    for idx, (model, model_name) in enumerate(zip(models, model_names)):
        # Handle Pipeline models
        if hasattr(model, 'named_steps'):
            # Extract the final predictor from the pipeline
            predictor = model.named_steps[list(model.named_steps.keys())[-1]]
            # Transform the input data using the pipeline's scaler
            X_input = model.named_steps['scaler'].transform(X_test_sample)
            # Use KernelExplainer for SHAP analysis
            explainer = shap.KernelExplainer(predictor.predict, X_input[:50])
            shap_values = explainer.shap_values(X_input, nsamples=100)
        else:
            # Use TreeExplainer for tree-based models, otherwise KernelExplainer
            if model_name in ['Random Forest', 'Gradient Boosting']:
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.KernelExplainer(model.predict, X_test_sample[:50])
            shap_values = explainer.shap_values(X_test_sample)
        
        # Store SHAP values in the dictionary
        shap_values_dict[model_name] = shap_values
        
        # Generate and save individual SHAP summary plot
        plt.figure(figsize=(10, 4))
        shap.summary_plot(shap_values, X_test_sample, feature_names=features, show=False)

        plt.title(f"SHAP Summary Plot ({model_name}, Lead Time = {lead_time} days)")

        # Save the plot to a temporary file
        plt.savefig(f"temp_summary_{idx}.png", bbox_inches='tight')
        summary_images.append(f"temp_summary_{idx}.png")
        plt.close()
    
    # Combine all individual summary plots into one image
    images = [Image.open(img) for img in summary_images]
    widths, heights = zip(*(i.size for i in images))
    total_height = sum(heights)
    max_width = max(widths)
    
    # Create a blank image with the combined dimensions
    combined_image = Image.new('RGB', (max_width, total_height))
    y_offset = 0
    for img in images:
        combined_image.paste(img, (0, y_offset))
        y_offset += img.size[1]
    
    # Save the combined image
    if save_path:
        combined_image.save(f"{save_path}_all_summary.png")
    
    # Clean up temporary files
    import os
    for img_file in summary_images:
        os.remove(img_file)
    
    return shap_values_dict

### 2. LIME Analysis ###
def lime_analysis(model, model_name, X_test_sample, features, num_samples=5, save_path=None):
    """Perform LIME analysis for a given model and generate explanations for a few instances."""

    print(f"Running LIME analysis for {model_name}...")
    
    # Handle Pipeline models
    if hasattr(model, 'named_steps'):
        # Use the full pipeline for prediction
        predictor = lambda x: model.predict(x)
    else:
        # Use the model's predict method directly
        predictor = model.predict

    # Initialize the LIME explainer
    explainer = lime_tabular.LimeTabularExplainer(
        X_test_sample.values, feature_names=features, mode='regression'
    )

    # Explain a few instances
    for i in range(min(num_samples, len(X_test_sample))):
        # Generate explanation for a single instance
        exp = explainer.explain_instance(X_test_sample.iloc[i].values, predictor, num_features=5)
        print(f"LIME Explanation for {model_name}, Instance {i+1}:")
        
        # Extract feature explanations as a list of tuples (feature, weight)
        explanation_list = exp.as_list()
        print(explanation_list)

        # Format feature labels for better readability in the plot
        formatted_labels = []
        for feature, weight in explanation_list:
            formatted_feature = (
                feature.replace('<=', r'$\leq$').replace('>=', r'$\geq$').replace('>', r'$>$').replace('<', r'$<$'))
            formatted_labels.append(f"{formatted_feature}")  # Add formatted feature to the list
        
        # Plot the explanation
        plt.figure()
        fig = exp.as_pyplot_figure()

        # Set manually formatted labels for the y-axis
        ax = fig.axes[0]
        ax.set_yticklabels(formatted_labels[::-1], fontsize=10)
        
        # Adjust layout
        plt.title(f"LIME Explanation ({model_name}, Instance {i+1}, Lead Time = {lead_time} days)")
        fig.set_size_inches(9, 6)
        fig.tight_layout(pad=2.0)
        plt.xlabel('Feature Influence')
        
        # Save the plot
        if save_path:
            plt.savefig(f"{save_path}_instance_{i+1}.png", bbox_inches='tight')
        plt.show()

### 3. Permutation Importance ###
def permutation_importance_analysis(model, model_name, X_test_sample, y_test_sample, features, save_path=None):
    """Perform Permutation Importance analysis for a given model."""
    
    print(f"Running Permutation Importance for {model_name}...")
    
    # Compute permutation importance
    result = permutation_importance(model, X_test_sample, y_test_sample, n_repeats=10, random_state=42)
    
    # Create a pandas Series to store the mean importance values
    perm_importance = pd.Series(result.importances_mean, index=features).sort_values(ascending=False)

    # Generate bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=perm_importance.values, y=perm_importance.index, palette='mako')
    plt.xlabel('Importance (MAE Increase)')  # Label for x-axis
    plt.ylabel('Feature')  # Label for y-axis
    plt.title(f"Permutation Importance ({model_name}, Lead Time = {lead_time} days)")  # Plot title
    
    # Save the plot
    if save_path:
        plt.savefig(save_path)
    plt.show()

    # Return the permutation importance values
    return perm_importance

# Run XAI analyses for all models
if __name__ == "__main__":
    # SHAP Analysis
    shap_analysis(models, model_names, X_test_sample, features, save_path=f"shap_plots", lead_time=7)

    # LIME Analysis (only for Random Forest and SVM)
    lime_analysis(rf_best, 'Random Forest', X_test_sample, y_test_sample, features, num_samples=2, 
                  save_path='Plots/lime_rf')
    lime_analysis(svm_best, 'SVM', X_test_sample, y_test_sample, features, num_samples=2, 
                  save_path='Plots/lime_svm')

    # Permutation Importance
    perm_importances = {}
    for model, name in zip(models, model_names):
        perm_importances[name] = permutation_importance_analysis(model, name, X_test_sample, y_test_sample, features, 
                                                                 save_path=f"Plots/perm_{name.lower().replace(' ', '_')}.png")

    print("XAI analysis completed.")