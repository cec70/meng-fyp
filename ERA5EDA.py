import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
plt.style.use('default')
sns.set_palette("muted")

# Set font properties
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

class ERA5EDA:
    """
    A class to perform Exploratory Data Analysis (EDA) on preprocessed ERA5 data
    for U10 (zonal wind) and T (polar temperature).
    """

    def __init__(self, preprocessed_data):
        """
        Parameters:
        - preprocessed_data (xr.Dataset): Preprocessed data from ERA5Preprocessor.
          The dataset must include:
          - 'time' (datetime64): Time dimension in ISO format.
          - 'u10' (float): Zonal wind at 10 hPa.
          - 't_polar' (float): Polar temperature at 10 hPa.
        """

        # Store the preprocessed data and extract relevant variables
        self.data = preprocessed_data
        self.time = pd.to_datetime(self.data['time'].values)
        self.u10 = self.data['u10'].values
        self.t_polar = self.data['t_polar'].values

        # Normalize u10 and t_polar
        self.u10_mean = np.mean(self.u10)
        self.u10_std = np.std(self.u10)
        self.t_polar_mean = np.mean(self.t_polar)
        self.t_polar_std = np.std(self.t_polar)

        self.u10_normalized = (self.u10 - self.u10_mean) / self.u10_std
        self.t_polar_normalized = (self.t_polar - self.t_polar_mean) / self.t_polar_std

        # Initialize SSW events as an empty list or add placeholder dates
        self.ssw_events = pd.to_datetime(['2000-03-20', '2001-02-11', '2001-12-31', '2002-02-18', '2003-01-18', '2004-01-05', '2006-01-21',
                                          '2007-02-24', '2008-02-22', '2009-01-24', '2010-02-09', '2010-03-24', '2013-01-06', '2018-02-12',
                                          '2019-01-02', '2021-01-05'])

    def plot_time_series(self):
        """Plot time series of normalized U10 and T with optional SSW event markers."""

        # Create a figure with two subplots for U10 and T Polar time series
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        # Plot normalized U10 and T Polar time series on the two subplots
        self._plot_u10(ax1)
        self._plot_t_polar(ax2)

        # Mark known SSW events on both subplots
        self._mark_ssw_events(ax1, ax2)

        # Adjust layout to prevent overlap and display the plots
        plt.tight_layout()
        plt.show()

    def _plot_u10(self, ax):
        """Helper method to plot normalized U10 time series."""

        # Plot the normalized U10 time series
        ax.plot(self.time, self.u10_normalized, label='$ U_{10} $ (Normalised)', color='darkblue')
        ax.set_title('Zonal Wind $ U_{10} $ at 10 hPa (Normalised)', fontsize=14)
        ax.set_ylabel('Normalised $ U_{10} $', fontsize=12)
        ax.legend()
        ax.grid(False)

    def _plot_t_polar(self, ax):
        """Helper method to plot normalized T Polar time series."""

        # Plot the normalized T Polar time series
        ax.plot(self.time, self.t_polar_normalized, label='$ T $ (Normalised)', color='darkred')
        ax.set_title('Polar Temperature $ T $ at 10 hPa (Normalised)', fontsize=14)
        ax.set_ylabel('Normalised $ T $', fontsize=12)
        ax.set_xlabel('Time', fontsize=12)
        ax.legend()
        ax.grid(False)

    def _mark_ssw_events(self, ax1, ax2):
        """Helper method to mark SSW events on the plots."""

        # Skip marking if no SSW events are provided
        if self.ssw_events.empty:
            return

        # Add vertical lines to mark SSW events on the plots
        label_set = False
        for ssw_date in self.ssw_events:
            # Add a legend label only for the first SSW event
            label = 'Known SSW Event' if not label_set else None
            ax1.axvline(ssw_date, color='black', linestyle='--', alpha=0.5, label=label)
            ax2.axvline(ssw_date, color='black', linestyle='--', alpha=0.5)
            label_set = True

        # Ensure the legend is displayed if SSW events are marked
        if not self.ssw_events.empty:
            ax1.legend()

    def plot_scatter(self):
        """Scatter plot of normalized U10 vs T to explore their relationship."""

        plt.figure(figsize=(8, 6))

        # Scatter plot
        plt.scatter(self.u10_normalized, self.t_polar_normalized, alpha=0.5, c='darkgray', label='Daily Data')

        # Highlight SSW events using U10 < 0 as a simple proxy
        ssw_indices = np.where((self.u10_normalized < -1) & (self.t_polar_normalized > 1))[0]
        if len(ssw_indices) > 0:
            plt.scatter(self.u10_normalized[ssw_indices], self.t_polar_normalized[ssw_indices], c='royalblue', marker='o', alpha=0.5, s=35, label='Potential SSW Event')

        # Highlight known SSW events in red
        label_set = False
        for ssw_date in self.ssw_events:
            ssw_index = np.where(self.time == ssw_date)[0]
            if len(ssw_index) > 0:
                label = 'Known SSW Event' if not label_set else None
                plt.scatter(self.u10_normalized[ssw_index], self.t_polar_normalized[ssw_index], c='red', marker='x', s=100, label=label)
                label_set = True

        # Plot formatting
        plt.xlabel('Normalised $ U_{10} $', fontsize=14)
        plt.ylabel('Normalised $ T $', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.show()

    def run_eda(self):
        """Run all EDA steps."""

        print("Starting Exploratory Data Analysis...")

        # Plot time series of normalized U10 and T with SSW event markers
        self.plot_time_series()
        # Create a scatter plot to explore the relationship between U10 and T
        self.plot_scatter()

        print("EDA completed.")

# Run EDA
if __name__ == "__main__":
    preprocessed_data_path = "Preprocessing/preprocessed_data.nc"
    preprocessed_data = xr.open_dataset(preprocessed_data_path)

    eda = ERA5EDA(preprocessed_data)
    eda.run_eda()