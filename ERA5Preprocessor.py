import xarray as xr
import numpy as np
import pandas as pd
import glob
import os
from datetime import datetime, timedelta
from pathlib import Path

class ERA5Preprocessor:
    """
    A class to preprocess multiple ERA5 daily files for SSW prediction.
    Handles files named like era5_YYYYdDDD.nc, using available variables.
    Processes u (U10 over 55–75°N) and t (T over 60–90°N) on L137 model levels.
    """
    
    # Mapping of standard variable and coordinate names to their aliases
    NAME_MAPPINGS = {
        'variables': {
            'u': ['u', 'zonal_wind'],  # Zonal wind
            't': ['t', 'temperature'],  # Temperature
            'v': ['v', 'meridional_wind']  # Meridional wind
        },
        'coords': {
            'level': ['level', 'model_level'],  # Model level coordinate
            'time': ['time', 'valid_time']  # Time coordinate
        }
    }
    
    def __init__(self, folder_path, file_pattern="era5_*.nc", variables=['u', 't'], model_level=30):
        """
        Parameters:
        - folder_path (str): Path to folder with ERA5 files.
        - file_pattern (str): Pattern to match files.
        - variables (list): Variables to process (default: ['u', 't']).
        - model_level (int): Model level (L137) approximating 10 hPa (default: 30).
        """

        # Initialize instance variables
        self.folder_path = folder_path
        self.file_pattern = file_pattern
        self.variables = variables
        self.model_level = model_level
        self.data = None

        # Load and preprocess the data
        self._load_data()
    
    def _parse_filename(self, filename):
        """Extract year and day of year from filename (e.g., era5_2003d018.nc -> (2003, 18))."""

        # Extract the base name of the file
        basename = os.path.basename(filename)
        try:
            year = int(basename[5:9])  # Extract year (YYYY)
            doy = int(basename[10:13])  # Extract day of year (DDD)
            
            # Convert year and doy to a datetime object
            date = datetime(year, 1, 1) + timedelta(days=doy - 1)
            
            # Return the parsed date and the original filename
            return (date, filename)
        except (ValueError, IndexError):
            # Raise an error if the filename format is invalid
            raise ValueError(f"Invalid filename format: {basename}. Expected 'era5_YYYYdDDD.nc'.")
    
    def _standardize_names(self, ds):
        """Rename variables and coordinates to standard names."""

        # Handle expver coordinate if present
        if 'expver' in ds.coords:
            try:
                expver_vals = ds['expver'].values
                if expver_vals.size > 1:
                    # Select the first valid expver value and drop the coordinate
                    first_expver = expver_vals[0] if not np.isnan(expver_vals[0]) else expver_vals[~np.isnan(expver_vals)][0]
                    ds = ds.sel(expver=first_expver).drop_vars('expver')
                else:
                    # If only one expver, squeeze and drop
                    ds = ds.squeeze('expver').drop_vars('expver')
            except Exception as e:
                print(f"Warning: Failed to process expver in {ds.attrs.get('source', 'unknown file')}: {e}")
                
                # Drop expver if processing fails
                ds = ds.drop_vars('expver', errors='ignore')
        
        # Standardize variable and coordinate names
        rename_dict = {}

        # Loop through aliases and map them to standard names if found in the dataset
        for std_name, aliases in self.NAME_MAPPINGS['variables'].items(): 
            for alias in aliases:
                if alias in ds.variables and std_name not in ds.variables:
                    rename_dict[alias] = std_name
        for std_name, aliases in self.NAME_MAPPINGS['coords'].items():
            for alias in aliases:
                if alias in ds.coords and std_name not in ds.coords:
                    rename_dict[alias] = std_name
        
        # Rename variables and coordinates in the dataset if any mappings were found
        if rename_dict:
            ds = ds.rename(rename_dict)
            print(f"Renamed variables/coords: {rename_dict}")
        
        # Return the dataset with only the required variables at the specified model level
        return ds[self.variables].sel(level=self.model_level)
    
    def _load_data(self):
        """Load and concatenate multiple ERA5 files, standardizing names."""

        try:
            # Get a list of files matching the pattern in the specified folder
            file_list = glob.glob(os.path.join(self.folder_path, self.file_pattern))
            if not file_list:
                raise ValueError(f"No files found in {self.folder_path} matching {self.file_pattern}")
            
            # Sort the files by their parsed datetime extracted from the filename
            sorted_files = sorted(file_list, key=lambda f: self._parse_filename(f)[0])
            
            # Open multiple files as a single dataset, applying preprocessing to standardize names
            ds = xr.open_mfdataset(
            sorted_files,
            combine='by_coords',
            parallel=True,
            preprocess=self._standardize_names
            )
            
            # Check if all required variables are present in the dataset
            missing_vars = [var for var in self.variables if var not in ds.variables]
            if missing_vars:
                raise ValueError(f"Variables {missing_vars} not found in dataset after standardization.")
            
            # Assign the loaded dataset to the instance variable
            self.data = ds
            print(f"Loaded {len(sorted_files)} files for {self.variables} at level {self.model_level}.")
        except Exception as e:
                # Raise an error if any issue occurs during the loading process
                raise ValueError(f"Error loading ERA5 data: {e}")
    
    def resample_daily(self):
        """Resample hourly data to daily means."""

        # Sort the data by time and resample it to daily means
        self.data = self.data.sortby('time').resample(time='1D').mean()
        print("Resampled to daily data.")
    
    def compute_anomalies(self, climatology_period=None):
        """Compute anomalies relative to climatology."""

        # Calculate the climatology for specified period or the entire dataset
        if climatology_period:
            start_year, end_year = climatology_period
            climatology = (self.data.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
                  .groupby('time.dayofyear').mean(dim='time'))
        else:
            climatology = self.data.groupby('time.dayofyear').mean(dim='time')
        
        # Compute anomalies by subtracting the climatology from the data
        self.data = self.data.groupby('time.dayofyear') - climatology
        print("Computed anomalies.")
    
    def compute_u10(self, lat_range=(55, 75)):
        """Compute U10: zonal wind averaged over 55-75°N."""

        # Select the subset of the data for the specified latitude range
        u_subset = self.data['u'].sel(latitude=slice(lat_range[1], lat_range[0]))
        
        # Compute weights based on the cosine of latitude 
        weights = np.cos(np.deg2rad(u_subset.latitude))
        
        # Apply weights to the zonal wind data
        u_weighted = xr.apply_ufunc(
            lambda x, w: x * w,  # Multiply data by weights
            u_subset,
            weights,
            input_core_dims=[['latitude'], ['latitude']],
            output_core_dims=[['latitude']],
            vectorize=True,
            dask='allowed'
        )
        
        # Compute the weighted average over latitude and longitude
        self.u10 = u_weighted.sum(dim=['latitude', 'longitude']) / weights.sum()
        
        print(f"Computed U10 over {lat_range}°N.")

    def compute_t_polar(self, lat_range=(60, 90)):
        """Compute T: temperature averaged over 60-90°N."""

        # Select temperature data for the specified latitude range
        t_subset = self.data['t'].sel(latitude=slice(lat_range[1], lat_range[0]))
        
        # Compute weights based on the cosine of latitude
        weights = np.cos(np.deg2rad(t_subset.latitude))
        
        # Apply weights to the temperature data
        t_weighted = xr.apply_ufunc(
            lambda x, w: x * w,
            t_subset,
            weights,
            input_core_dims=[['latitude'], ['latitude']],
            output_core_dims=[['latitude']],
            vectorize=True,
            dask='allowed'
        )
        
        # Compute the weighted average over latitude and longitude
        self.t_polar = t_weighted.sum(dim=['latitude', 'longitude']) / weights.sum()
      
        print(f"Computed T over {lat_range}°N.")
    
    def create_feature_dataset(self):
        """Create an xarray Dataset for U10 and T Polar features."""

        # Convert U10 and T to a pandas DataFrame
        features = pd.DataFrame(
            {
                'u10': self.u10.values,
                't_polar': self.t_polar.values
            },
            index=self.u10['time'].values
        )

        # Create an xarray Dataset from the features DataFrame
        self.features = xr.Dataset(
            {
                'u10': ('time', features['u10'].values),
                't_polar': ('time', features['t_polar'].values)
            },
            coords={'time': features.index}
        )
        
        print(f"Created feature dataset with variables: {list(features.columns)}")
    
    def get_preprocessed_data(self):
        """Return preprocessed features."""

        return self.features

# Run main
if __name__ == "__main__":
    base_folder_path = str(Path("/data3/ERA5"))
    
    all_files = []
    for year in range(2000, 2022 + 1):
        folder_path = os.path.join(base_folder_path, str(year))
        preprocessor = ERA5Preprocessor(folder_path, file_pattern="era5_*.nc", variables=['u', 't'], model_level=30)
        print(f"Loaded {year}")
        # Collect all files from the preprocessor
        all_files.append(preprocessor.data)
    
    # Combine all the data from the collected files
    combined_data = xr.concat(all_files, dim='time')
    
    # Ensure the time index is sorted
    combined_data = combined_data.sortby('time')
    
    # Continue with the rest of the processing
    preprocessor.data = combined_data
    preprocessor.resample_daily()
    preprocessor.compute_anomalies(climatology_period=(2000, 2022))
    preprocessor.compute_u10(lat_range=(55, 75))
    preprocessor.compute_t_polar(lat_range=(60, 90))
    preprocessor.create_feature_dataset()
    
    preprocessed_data = preprocessor.get_preprocessed_data()
    print(preprocessed_data)

    # Save
    output_path = "preprocessed_data.nc"
    preprocessed_data.to_netcdf(output_path)
    print(f"Saved preprocessed data to {output_path}")
