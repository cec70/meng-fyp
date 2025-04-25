import xarray as xr
import numpy as np
from scipy.ndimage import label as scipy_label  # For consecutive day detection

# Load preprocessed data
data_path = "Preprocessing/preprocessed_data.nc"  # Test subset
data = xr.open_dataset(data_path)

# Extract u10 and t_polar
u10 = data['u10']  # Zonal wind anomaly (55–75°N)
t_polar = data['t_polar']  # Temperature anomaly (60–90°N)
time = data['time']

# Normalize u10 and t_polar
u10_mean = np.mean(u10)
u10_std = np.std(u10)
t_polar_mean = np.mean(t_polar)
t_polar_std = np.std(t_polar)

u10_normalized = (u10 - u10_mean) / u10_std
t_polar_normalized = (t_polar - t_polar_mean) / t_polar_std

# Option 1: Continuous SSW Index 
def compute_continuous_ssw_index(u10, t_polar, w=0.1):
    """
    Compute a continuous SSW index as a weighted combination of -u10 and t_polar.
    Negative u10 (wind reversal) and positive t_polar (warming) increase the index.
    
    Parameters:
    - u10 (xarray.DataArray): Normalized zonal wind anomaly.
    - t_polar (xarray.DataArray): Normalized temperature anomaly.
    - w (float): Weight for t_polar (default: 0.1 to balance m/s and K scales).
    
    Returns:
    - ssw_index: xarray.DataArray with continuous SSW index.
    """

    # Weighted combination of u10 and t_polar
    ssw_index = -u10 + w * t_polar  # Higher values indicate stronger SSW likelihood
    return ssw_index.rename("ssw_index")

# Option 2: Binary SSW Labels (Standard Criteria)
def compute_binary_ssw_labels(u10, t_polar, min_days=7, u10_threshold=0, t_polar_threshold=0.5):
    """
    Compute binary SSW labels based on u10 < 0 and t_polar > threshold for >7 days.
    
    Parameters:
    - u10 (xarray.DataArray): Normalized zonal wind anomaly.
    - t_polar (xarray.DataArray): Normalized temperature anomaly.
    - min_days (int): Minimum consecutive days for SSW (default: 7).
    - u10_threshold (float): Threshold for u10 (default: 0).
    - t_polar_threshold (float): Threshold for t_polar warming (default: 0.5).
    
    Returns:
    - ssw_labels: xarray.DataArray with binary labels (1 = SSW, 0 = non-SSW).
    """

    # Condition: u10 < 0 (wind reversal) and t_polar > 0.5 (warming)
    ssw_condition = (u10 < u10_threshold) & (t_polar > t_polar_threshold)
    
    # Label consecutive periods
    ssw_binary = ssw_condition.astype(int).values
    labeled_array, num_features = scipy_label(ssw_binary)  # Identify consecutive runs
    
    # Filter for runs >= min_days
    ssw_labels = np.zeros_like(ssw_binary)
    for feature in range(1, num_features + 1):
        feature_mask = (labeled_array == feature)
        if feature_mask.sum() >= min_days:
            ssw_labels[feature_mask] = 1
    
    # Return the binary SSW labels with time coordinates
    return xr.DataArray(ssw_labels, coords={"time": u10["time"]}, dims=["time"], name="ssw_labels")

# Compute both options
ssw_index = compute_continuous_ssw_index(u10_normalized, t_polar_normalized, w=0.1)
ssw_labels = compute_binary_ssw_labels(u10_normalized, t_polar_normalized, min_days=7, u10_threshold=0, t_polar_threshold=0.5)

# Add to dataset
data_with_labels = data.assign(ssw_index=ssw_index, ssw_labels=ssw_labels)

# Save updated dataset
output_path = "Preprocessing/preprocessed_data_with_labels.nc"
data_with_labels.to_netcdf(output_path)
print(f"Saved data with SSW index and labels to {output_path}")

# Print samples for verification
print("\nSample u10 values:", u10_normalized.values[:5])
print("Sample t_polar values:", t_polar_normalized.values[:5])
print("Sample SSW index values:", ssw_index.values[:5])
print("Sample SSW labels:", ssw_labels.values[:5])