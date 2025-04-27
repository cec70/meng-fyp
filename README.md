# MEng Final Year Project
### Machine Learning-based Prediction of Sudden Stratospheric Warmings

This repository contains a comprehensive pipeline for predicting and analysing Sudden Stratospheric Warming (SSW) events using Machine Learning (ML) models. The pipeline includes data preprocessing, feature engineering, model training, evaluation, and explainability analysis.

---

## **Table of Contents**
1. [Overview](#overview)
2. [Dependencies](#dependencies)
3. [Pipeline Steps](#pipeline-steps)
4. [Results](#results)
5. [Acknowledgments](#acknowledgments)

---

## **Overview**
Sudden Stratospheric Warming events are significant meteorological phenomena that can disrupt weather patterns. This project aims to:
- Predict SSW events using ML models.
- Analyse the importance of meteorological features in SSW prediction.
- Provide explainability for the predictions using SHAP, LIME, and Permutation Importance.

The pipeline processes ERA5 reanalysis data, extracts relevant features, trains Machine Learning models, and evaluates their performance across different lead times.

---

## **Dependencies**
The project requires the following Python libraries:
- NumPy (v1.26.0)
- Pandas (v2.1.1)
- Xarray (v2023.9.0)
- Matplotlib (v3.8.1)
- Seaborn (v0.12.2)
- Scikit-learn (v1.3.1)
- LIME (v0.2.0.1)
- SHAP (v0.42.1)
- Joblib (v1.3.2)
- Pillow (v10.0.1)
- SciPy (v1.11.2)

Install the dependencies using:  
```
pip install -r requirements.txt
```

---

## **Pipeline Steps**
### 1. Data Preprocessing
**Script**: ERA5Preprocessor.py  
**Description**: Processes ERA5 reanalysis data to compute zonal wind *U<sub>10</sub>* and polar temperature *T* anomalies. Outputs a NetCDF file with preprocessed data.  
**Output**: preprocessed_data.nc

### 2. Exploratory Data Analysis (EDA)
**Script**: ERA5EDA.py  
**Description**: Performs EDA on the preprocessed ERA5 reanalysis data for *U<sub>10</sub>* and *T*.  
**Output**: EDA plots (*.png).

### 3. SSW Indexing and Labelling
**Script**: SSWIndexLabel.py  
**Description**: Computes a continuous SSW index and binary labels based on predefined thresholds.  
**Output**: preprocessed_data_with_labels.nc

### 4. Feature Engineering
**Script**: FeatureSelection.py  
**Description**: Adds lagged features for *U<sub>10</sub>* and *T* to capture temporal dependencies. Outputs a dataset with lagged features.  
**Output**: preprocessed_data_with_lags.nc

### 5. Model Training
**Script**: MLModel.py  
**Description**: Trains ML models (Random Forest, MLP, Gradient Boosting, SVM, and Linear Regression) to predict the SSW index. Evaluates models across multiple lead times.  
**Output**: Trained models (*.pkl) and MAE results (mae_results.npy).

### 6. Model Evaluation
**Script**: ModelEvaluation.py  
**Description**: Evaluates model performance using scatter plots, time-series plots, and residual plots. Compares Random Forest and SVM using statistical tests.  
**Output**: Evaluation plots (*.png).

### 7. Explainability Analysis
**Script**: XAI.py  
**Description**: Uses SHAP, LIME, and Permutation Importance to explain model predictions and identify key features.  
**Output**: Explanation plots (*.png).

---

## **Results**
**Model Performance**: Random Forest and SVM models achieved the best performance across multiple lead times.  
**Feature Importance**: *U<sub>10</sub>* and its lagged features were identified as the most important predictors for SSW events.  
**Explainability**: SHAP, LIME, and Permutation Importance provided insights into how individual features influenced model predictions.

---

## **Acknowledgments**
This project uses ERA5 reanalysis data provided by the European Centre for Medium-Range Weather Forecasts (ECMWF). Special thanks to the Centre for Climate Adaptation & Environmental Research at the University of Bath for supporting this research.
