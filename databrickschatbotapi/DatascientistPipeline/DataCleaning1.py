import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns
import pandas as pd
import missingno as msno
import numpy as np

class DataCleaner:
    def __init__(self, source, df, file_name,process_id, problem_type, type_column, dependent_var):
        self.source = source
        self.file_name = file_name
        self.data = df
        self.problem_type = problem_type
        self.type_column = type_column
        self.dependent_var = dependent_var
        self.problem_type_identify()
        self.process_id=process_id
        

            
    def problem_type_identify(self):
        try:
            print("I am in problem type identifier function")
            # Call the appropriate EDA method based on problem type
            if self.problem_type.lower() == "time series":
                self.time_series()
            elif self.problem_type.lower() == "categorical":
                self.categorical()
            elif self.problem_type.lower() == "numerical":
                self.numerical()
            else:
                raise ValueError("Invalid problem type. Supported types are 'time series', 'categorical', and 'numerical'.")
        except Exception as e:
            print(f"Error occurred in problem type identification: {e}")

    def numerical(self):
#         print("cleaning")
#         self.handle_missing_values(method='drop')
#         self.handle_duplicates()
#         self.remove_high_cardinality_and_constant_columns()
#         self.handle_outliers(threshold_percentage=2)
#         self.reset_index()
#         self.msno_plot()
        
        print("cleaning")
        self.remove_high_cardinality_and_constant_columns()
        self.handle_outliers(threshold_percentage=1)
        self.data = self.identify_and_remove_outliers(k=1.5)
        self.drop_columns()
        # handling categorical missing values
        self.handle_cat_missing_values(method = 'fillna_others')
        # handling numerical missing values
        self.handle_missing_values(method= 'fillna_mean')
        self.handle_duplicates()
        self.reset_index()
        self.msno_plot()
        self.reset_index()
        #removing the datetime columns
#         self.remove_datetime_columns()
#         self.data.info()
        
        
    def categorical(self):
#         self.remove_unamed()
        print("cleaning")
        self.remove_high_cardinality_and_constant_columns()
        self.handle_outliers(threshold_percentage=1)
#         self.data = self.identify_and_remove_outliers(k=1.5)
        self.drop_columns()
        # handling categorical missing values
        self.handle_cat_missing_values(method = 'fillna_others')
        # handling numerical missing values
        self.handle_missing_values(method= 'fillna_mean')
        self.handle_duplicates()
        self.reset_index()
        self.msno_plot()
        self.reset_index()
        self.remove_datetime_columns()


    def get_data(self):
        return self.data    
    
    def time_series(self):
#         self.plot_resampled_time_series()
#         self.data = self.identify_and_remove_outliers(k=1.5)
        self.drop_columns()
#         print("____ Missing Numbers after Outliers Treatment_____")
#         self.msno_plot()
        if self.data.isnull().any().any():
            self.data.fillna(method='ffill', inplace = True)
            self.data.fillna(method='bfill', inplace=True)
            self.data.fillna(self.data.mean(), inplace=True)
        
        self.handle_duplicates()
#         self.handle_datetime(format='%Y-%m-%d %H:%M:%S')
#         self.reset_index()
        print("____ Dataset after missing values Treatment_____")
        self.msno_plot() 
        self.remove_high_cardinality_and_constant_columns()

             

    def remove_high_cardinality_and_constant_columns(self, threshold=90):
        """
        Remove high-cardinality categorical columns and constant columns.

        Parameters:
        - df: pandas DataFrame
            The input DataFrame.
        - threshold: int, optional (default=90)
            The threshold for the number of unique values. Columns with more than
            this number of unique values will be removed.

        Returns:
        - df_filtered: pandas DataFrame
            The DataFrame with high-cardinality categorical columns and constant columns removed.
        """
        # Identify categorical columns
        categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns

        # Filter columns with high cardinality
        high_cardinality_columns = [col for col in categorical_columns if  self.data[col].nunique() > threshold]

        # Identify constant columns
        constant_columns = [col for col in self.data.columns if self.data[col].nunique() == 1]

        # Combine columns to remove
        columns_to_remove = set(high_cardinality_columns + constant_columns)

        # Drop columns
        df_filtered =  self.data.drop(columns=columns_to_remove)
        
        self.data = df_filtered
        
        
    def remove_datetime_columns(self):
        """
        Removes datetime columns from a DataFrame.

        Parameters:
        - df (DataFrame): Input DataFrame containing datetime columns.

        Returns:
        - df (DataFrame): DataFrame with datetime columns removed.
        """
        # Identify datetime columns
        datetime_columns = []
        for column in self.data.columns:
            try:
                if self.data[column].dtype == 'datetime64[ns]':
                    pd.to_datetime(self.data[column])
                    datetime_columns.append(column)
            except ValueError:
                pass

        # Remove datetime columns from DataFrame
        self.data.drop(columns=datetime_columns,inplace=True,errors='ignore')
        return self.data
        
    
    
    def drop_columns(self, threshold=0.5):
        """
        Drop columns with missing values exceeding a specified threshold.

        Parameters:
        - threshold (float): Threshold for the percentage of missing values.
        """
        cols_to_drop = self.data.columns[self.data.isnull().mean() > threshold]
        self.data.drop(columns=cols_to_drop, inplace=True)
        
        id_patterns = ['id', 'ID', 'Id', '_id', 'Identifier', 'identifier']
    
        # Find column names that contain any of the patterns.
        id_columns = [col for col in self.data.columns if any(pattern in col for pattern in id_patterns)]
        
            # Drop the identified columns from the DataFrame.
        self.data.drop(columns=id_columns,inplace=True, errors='ignore')  # errors='ignore' allows dropping non-existing columns without error
    
    def identify_and_remove_outliers(self, k=1.5):
        """
        Identify and remove outliers from a time series using the Interquartile Range (IQR) method.

        Parameters:
        - time_series_data (pd.Series): The time series data to process.
        - k (float): The multiplier for determining the outlier threshold. Default is 1.5.

        Returns:
        - pd.Series: The time series data with outliers removed.
        """
        time_series_data = self.data
        # Calculate quartiles and IQR
        Q1 = time_series_data.quantile(0.25)
        Q3 = time_series_data.quantile(0.75)
        IQR = Q3 - Q1

        # Define the outlier threshold range
        lower_bound = Q1 - k * IQR
        upper_bound = Q3 + k * IQR

        # Identify outliers
        outliers = (time_series_data < lower_bound) | (time_series_data > upper_bound)

        # Replace outliers with NaN or choose another method (e.g., imputation)
        time_series_data[outliers] = np.nan

        # Alternatively, you can remove the outliers by filtering
        # filtered_data = time_series_data[~outliers]
        self.data = time_series_data

        return time_series_data

    def handle_outliers(self, threshold_percentage = 1):
        # Separate numerical and categorical columns
        numerical_columns = self.data.select_dtypes(include=np.number).columns
        categorical_columns = self.data.select_dtypes(include='object').columns

        # Identify and handle outliers for numerical columns
        for col in numerical_columns:
            # Identify outliers using IQR
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Handle outliers by capping extreme values
            self.data[col] = self.data[col].clip(lower=lower_bound, upper=upper_bound)

        # Identify and handle outliers for categorical columns
        for col in categorical_columns:
            # Identify rare categories based on a dynamic threshold
            threshold_count = len(self.data) * threshold_percentage / 100
            category_counts = self.data[col].value_counts()
            rare_categories = category_counts[category_counts < threshold_count].index

            # Handle rare categories by replacing them with a common label or removing them
            self.data[col] = self.data[col].replace(rare_categories, 'Other')

        # Visualize the impact of outlier handling for numerical columns
        plt.figure(figsize=(15, 6))
        sns.boxplot(data=self.data[numerical_columns])
        plt.xticks(rotation=90)
        plt.title('Boxplot After Handling Numerical Outliers')
        plt.show()

        # Visualize the impact of outlier handling for categorical columns
        for col in categorical_columns:
            plt.figure(figsize=(10, 6))
            sns.countplot(x=col, data=self.data)
            plt.xticks(rotation=90)
            plt.title(f'Count Plot for {col} After Handling Outliers')
            plt.show()
    
    
    def msno_plot(self):
        msno.bar(self.data) # you can see pandas-profilin count part
        plt.title('Count of Values per Column in Dataset for Missing value Analysis', size=16)
        plt.show()
    
    def handle_missing_values(self, method= 'drop'):
        """
        Handle missing values in the dataset.

        Parameters:
        - method (str): Method for handling missing values. Options: 'drop', 'fillna_mean', 'fillna_median'.
        """
        if method == 'drop':
            self.data.dropna(inplace=True)
        elif method == 'fillna_mean':
            self.data.fillna(self.data.mean(), inplace=True)
        elif method == 'fillna_median':
            self.data.fillna(self.data.median(), inplace=True)
        elif method == 'linear':
            self.data.interpolate(method='linear', inplace = True)
        else:
            raise ValueError("Invalid method. Use 'drop', 'fillna_mean', 'fillna_median' or 'linear'.")

#         df_filled = self.data.fillna(method='ffill')
        return self.data

    def handle_duplicates(self):
        """
        Remove duplicate rows from the dataset.
        """
        self.data.drop_duplicates(inplace=True)


#     def handle_outliers(self, method='z-score', threshold=3):
#         """
#         Handle outliers in the dataset.

#         Parameters:
#         - method (str): Method for handling outliers. Options: 'z-score', 'IQR'.
#         - threshold (float): Threshold for identifying outliers.
#         """
#         numeric_columns = self.data.select_dtypes(include=[np.number]).columns

#         for col in numeric_columns:
#             if method == 'z-score':
#                 z_scores = (self.data[col] - self.data[col].mean()) / self.data[col].std()
#                 self.data = self.data[(np.abs(z_scores) < threshold)]
#             elif method == 'IQR':
#                 Q1 = self.data[col].quantile(0.25)
#                 Q3 = self.data[col].quantile(0.75)
#                 IQR = Q3 - Q1
#                 self.data = self.data[~((self.data[col] < (Q1 - 1.5 * IQR)) | (self.data[col] > (Q3 + 1.5 * IQR)))]
#             else:
#                 raise ValueError("Invalid method. Use 'z-score' or 'IQR'.")


    def handle_datetime(self, format='%Y-%m-%d %H:%M:%S'):
        """
        Convert the first datetime column found to datetime format.

        Parameters:
        - format (str): Format of the datetime column.
        """
        for col in self.data.columns:
            try:
                self.data[col] = pd.to_datetime(self.data[col], format=format)
                break  # Stop after the first datetime column is found and converted
            except (TypeError, ValueError):
                pass  # Continue if the conversion fails for the current column
        else:
            print("No datetime column found in the dataset.")

    def reset_index(self):
        """
        Reset the index of the DataFrame.
        """
        self.data.reset_index(drop=True, inplace=True)
        
        
    def handle_cat_missing_values(self, method='fillna_mode', fill_value = "others" ):
        for col in self.data.select_dtypes(include='object').columns:
            if method == 'drop':
                self.data.dropna(inplace=True)
            elif method == 'fillna_mode':
                self.data.fillna(self.data.mode().iloc[0], inplace=True) # Using iloc[0] to handle cases where mode returns multiple values
            
            elif method == 'fillna_others':
                self.data[col].fillna(fill_value, inplace=True)

    def plot_resampled_time_series(self, date_col='date', value_col='value'):
        """
        Plot time series graph after automatic resampling based on data's time range.

        Parameters:
        - data (DataFrame): The input time series data.
        - date_col (str): The name of the column containing date values.
        - value_col (str): The name of the column containing numeric values.

        Returns:
        - None
        """
        df = self.data
        datetime_column = self.type_column[0]
        # Convert the datetime column to a pandas datetime object
        df[datetime_column] = pd.to_datetime(df[datetime_column])

        # Set the datetime column as the index
        df.set_index(datetime_column, inplace=True)

        # Downsample with Mean
        resampled_df = df.resample('D').mean()

        # Plot original and resampled data
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df[self.dependent_var], label='Original Data', marker='o')
        plt.plot(resampled_df.index, resampled_df[self.dependent_var], label='Resampled Data', marker='o')
        plt.title('Original vs Resampled Time Series Data')
        plt.xlabel('Date')
        plt.ylabel('Your Column')
        plt.legend()
        plt.show()
        
        self.data = df