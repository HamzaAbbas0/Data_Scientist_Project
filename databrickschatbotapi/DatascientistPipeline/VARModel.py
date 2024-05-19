import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.api import VAR

class VARModel:
    def __init__(self, time_series):
        """
        Initialize the VAR model with a time series.

        Parameters:
        - time_series: pd.DataFrame or np.array
            The multivariate time series data.
        """
        self.time_series = time_series
        

    def adf_test(self):
        """
        Perform Augmented Dickey-Fuller test for stationarity.

        Returns:
        - p_values: pd.Series
            The p-values from the ADF test for each variable.
        """
        p_values = self.time_series.apply(lambda x: adfuller(x)[1])
        return p_values
    

    def difference_series(self, order=1):
        """
        Difference the time series to make it stationary.

        Parameters:
        - order: int, optional (default=1)
            The order of differencing.

        Returns:
        - differenced_series: pd.DataFrame
            The differenced time series.
        """
        differenced_series = self.time_series.diff(order).dropna()
        return differenced_series
    

    def plot_acf_pacf(self, lags=20):
        """
        Plot the autocorrelation function (ACF) and partial autocorrelation function (PACF) for diagnostic purposes.

        Parameters:
        - lags: int, optional (default=20)
            The number of lags to include in the plot.
        """
        fig, ax = plt.subplots(self.time_series.shape[1], 2, figsize=(12, 4 * self.time_series.shape[1]))
        for i, col in enumerate(self.time_series.columns):
            plot_acf(self.time_series[col], lags=lags, ax=ax[i, 0], title=f'ACF - {col}')
            plot_pacf(self.time_series[col], lags=lags, ax=ax[i, 1], title=f'PACF - {col}')
        plt.show()
        

    def train_test_split(self, test_size=0.2):
        """
        Split the time series into training and testing sets.

        Parameters:
        - test_size: float, optional (default=0.2)
            The proportion of the data to include in the test split.

        Returns:
        - train_set: pd.DataFrame
            The training set.
        - test_set: pd.DataFrame
            The testing set.
        """
        split_index = int(len(self.time_series) * (1 - test_size))
        train_set, test_set = self.time_series.iloc[:split_index], self.time_series.iloc[split_index:]
        return train_set, test_set
    

    def fit_var(self, train_set, lag_order=2):
        """
        Fit a VAR model to the multivariate time series.

        Parameters:
        - lag_order: int, optional (default=1)
            The lag order of the VAR model.

        Returns:
        - var_model: statsmodels.tsa.vector_ar.var_model.VAR
            The fitted VAR model.
        """
        var_model = VAR(train_set)
        var_model_instance = var_model.fit(maxlags=10)
        var_model_instance.save('saved_model/var_model.pkl')
        return var_model_instance

    
        
    def forecast(self, train_set, test_set, steps=10):
        """
        Forecast future values using the fitted VAR model.

        Parameters:
        - steps: int, optional (default=10)
            The number of steps to forecast into the future.

        Returns:
        - forecast_values: pd.DataFrame
            The forecasted values.
        """
        var_model_instance = self.fit_var(train_set)
        print(var_model_instance.summary())

        # Forecast future values
        forecast_values = var_model_instance.forecast(train_set.values, steps=steps)

        # Create a DatetimeIndex for the forecasted values
        # forecast_index = pd.date_range(start=self.time_series.index[-1], periods=steps + 1, freq=self.time_series.index.freq)[1:]

        
        forcasted_results = pd.DataFrame(forecast_values)
        forcasted_results.head(12)
        forcasted_results.index = test_set.index
        forcasted_results.head(12)
        forcasted_results.columns = test_set.columns
        return forcasted_results
        return forecast_values


    def evaluate_model(self, train_set, test_set):
        """
        Evaluate the VAR model on the test set.

        Parameters:
        - test_set: pd.DataFrame
            The testing set.

        Returns:
        - mae_scores: pd.Series
            Mean Absolute Error (MAE) for each variable.
        """
        # Forecast future values using the fitted VAR model
        forecast_values = self.forecast(train_set, test_set, steps=len(test_set))

        # Calculate Mean Absolute Error (MAE) for each variable
        mae_scores = test_set.subtract(forecast_values).abs().mean()

        return mae_scores
    
    
    def plot_forecasted(self, test_set, forecast_values):
        # Plotting individual plots for each variable
        for variable in test_set.columns:
            plt.figure(figsize=(10, 5))

            # Plot actual values
            plt.plot(test_set.index, test_set[variable], label=f'Actual {variable}', color='blue')

            # Plot forecasted values
            plt.plot(forecast_values.index, forecast_values[variable], label=f'Forecasted {variable}', linestyle='dashed', color='red')

            plt.xlabel('Date')
            plt.ylabel('Values')
            plt.title(f'{variable} Forecasting')
            plt.legend()
            plt.show()
