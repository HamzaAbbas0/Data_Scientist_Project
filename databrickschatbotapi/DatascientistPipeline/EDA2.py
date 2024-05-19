import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns
import pandas as pd
import missingno as msno
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder
from databrickschatbotapi.DatascientistPipeline.JSON_creator import MyDictionaryManager
import geopandas as gpd

plt.rcParams['agg.path.chunksize'] = 20000

class EDA:
    def __init__(self, source, df, file_name,process_id, problem_type, type_column, dependent_var,date_index):
#         self.json_filename = f'{problem_type}/json/{file_name}.json'
        self.source = source
        self.file_name = file_name
        self.data = df
        
        self.problem_type = problem_type
        self.type_column = type_column
        self.dependent_var = dependent_var
        self.date_index = date_index
        self.process_id=process_id
        
        
        # JSON Innitialization
        self.dict_manager = MyDictionaryManager(source, file_name,process_id, problem_type)
#         self.dict_manager.update_value('File name', self.file_name)
#         self.dict_manager.update_value('Problem type', self.problem_type)
#         self.dict_manager.save_dictionary()
        self.problem_type_identify1()


    def problem_type_identify1(self):
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
            
    def time_series(self):
        self.summary_statistics()
#         self.missing_values()
        print("____ Missing Numbers before Outliers Treatment_____")
        self.msno_plot()
#         self.visualize_distribution()
        self.visualize_trends(self.dependent_var,self.date_index)
#         self.plot_choropleth_map()
        self.plot_correlation()
        self.dict_manager.save_dictionary()
#         self.correlation_analysis()

            
    def get_data(self):
        return self.data
        
    def numerical(self):
        self.summary_statistics()
        print("____ Missing Numbers before Outliers Treatment_____")
        self.msno_plot()
        self.visualize_distribution()
        self.pairwise_plots()
        self.plot_histogram_within_categories()
        self.plot_bar_chart()
        self.plot_correlation()
#         self.plot_stacked_bar_chart()
#         self.dict_manager.save_dictionary()
        
    def categorical(self):
#         frequency_dist = self.frequency_distribution()
#         print(frequency_dist)
        self.summary_statistics()
        self.msno_plot()
        print("_________ Exploratory Data Analysis of Categorical Variables ________")
        self.plot_bar_chart()
#         self.plot_stacked_bar_chart()
#         self.plot_box_plot(df)
        self.create_cross_tab()
        self.perform_chi_square_test()
        self.plot_histogram_within_categories()
        print("_________ Exploratory Data Analysis of Numerical Variables ________")
        self.visualize_distribution()
        self.plot_correlation()
#         self.dict_manager.save_dictionary()
        
#     def mixed(self):
#         self.numerical()
#         self.categorical()


# class TimeSeries:
#     def __init__(self, df, problem_type, type_column):
#         summary_statistics(self)
            
            
        
        
                
# class Numerical:
# #     def __init__(self, data, problem_type):
        
    
# class Categorical:
    
    
# class Mixed:


    
# Independent functions
    
# def __init__(self, data):
#     self.data = data



    def summary_statistics(self):
        try:
            # Display basic statistics of the dataset
            print("Summary Statistics:")
            print(self.data.describe())
    #         self.data.describe().to_csv(f'LSTM_result_csv/data_description.csv')

            path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/csv/summary_statistics.csv'
            self.dict_manager.update_value('Summary Statistics ', path)
            self.data.describe().to_csv(path)
        except Exception as e:
            self.logger.error(f"An error occurred in summary_statistics method: {e}")
            

    def frequency_distribution(self):
        try:
            return self.data.apply(lambda column: column.value_counts())
        except Exception as e:
            self.logger.error(f"An error occurred in frequency_distribution method: {e}")
            

    def plot_bar_chart(self):
        try:
            path_for_json = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/graphs/bar_chart'
            self.dict_manager.update_value(f'Bar charts ', path_for_json)
            for column in self.data.select_dtypes(include='object'):
                if len(self.data[column].unique()) > 20 and column != self.dependent_var:
                    print(f"Skipping {column} due to more than 20 categories.")
                    continue
                sns.countplot(x=column, data=self.data)
                plt.xticks(rotation=90)
                plt.title(f'Bar Chart for {column}')
                path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/graphs/bar_chart_{column}.png'
                plt.tight_layout()
                plt.savefig(path)
                plt.show()
        except Exception as e:
            self.logger.error(f"An error occurred in plot_bar_chart method: {e}")
            

    def plot_stacked_bar_chart(self):
        try:
            path_for_json = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/graphs/stacked_column_chart'
            self.dict_manager.update_value(f'Stacked column charts for two columns ', path_for_json)
            for x_column in self.data.select_dtypes(include='object'):
                for hue_column in self.data.select_dtypes(include='object'):
                    if x_column != hue_column:
                        sns.countplot(x=x_column, hue=hue_column, data=self.data)
                        plt.xticks(rotation=90)
                        plt.title(f'Stacked Column Chart for {x_column} by {hue_column}')
                        path = f'Knowledge/{problem_type}/{self.process_id}/graphs/stacked_column_chart_{x_column}_vs_{hue_column}.png'

                        plt.savefig(path)
                        plt.show()
        except Exception as e:
            self.logger.error(f"An error occurred in plot_stacked_bar_chart method: {e}")
            

    def create_cross_tab(self):
        try:
            path_for_json = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/csv/Crosstab'
            self.dict_manager.update_value(f'Crosstabs of two columns ', path_for_json)
            path_for_json = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/graphs/Cross_tabulation'
            self.dict_manager.update_value(f'Cross tabulation graph of two columns ', path_for_json)
            for column1 in self.data.select_dtypes(include='object'):
                for column2 in self.data.select_dtypes(include='object'):
                    if column1 != column2:
                        cross_tab = pd.crosstab(self.data[column1], self.data[column2])
                        path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/csv/Crosstab_{column1}_vs_{column2}.csv'

                        cross_tab.to_csv(path)
    #                     print(crosstab)
                        # Plotting the cross-tabulation as a heatmap
                        sns.heatmap(cross_tab, annot=True, fmt='d', cmap='viridis')
                        plt.title(f'Cross-Tabulation between {column1} and {column2}')
                        path = f'Knowledge/{self.problem_type}/{self.source}/{self.file_name}/graphs/Cross_tabulation_{column1}_vs_{column2}.png'
                        plt.savefig(path)
                        plt.show()
        except Exception as e:
            self.logger.error(f"An error occurred in create_cross_tab method: {e}")
            

    def perform_chi_square_test(self):
        try:
            chi_columns=['Column1', 'Column2', 'chi_value', 'P-value']
            chi_results = pd.DataFrame(columns = chi_columns)
            for column1 in self.data.select_dtypes(include='object'):
                for column2 in self.data.select_dtypes(include='object'):
                    if column1 != column2:
                        crosstab_result = pd.crosstab(self.data[column1], self.data[column2])
                        chi2, p, dof, expected = chi2_contingency(crosstab_result)
                        chi_results = chi_results.append({'Column1': column1, 'Column2': column2, 'chi_value': chi2, 'P-value': p}, ignore_index=True)
                        print(f"Chi-square statistic for {column1} and {column2}: {chi2}, p-value: {p}")
            path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/csv/chi_square_statistics.csv'
            self.dict_manager.update_value(f'Chi-Square statistics ', path)
            chi_results.to_csv(path, index=False)
        except Exception as e:
            self.logger.error(f"An error occurred in perform_chi_square_test method: {e}")
            

    def plot_histogram_within_categories(self):
        try:
            path_for_json = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/graphs/histogram'
            self.dict_manager.update_value(f'Histogram of two columns ', path_for_json)

            for x_column in self.data.select_dtypes(include='number'):
                for hue_column in self.data.select_dtypes(include='object'):
                    # Check if the column has more than 20 categories and is not the dependent variable
                    if len(self.data[hue_column].unique()) > 20 and hue_column != self.dependent_var:
                        print(f"Skipping {hue_column} due to more than 20 categories.")
                        continue

                    sns.histplot(x=x_column, hue=hue_column, data=self.data, multiple='stack')
                    plt.title(f'Histogram for {x_column} within Categories of {hue_column}')
                    path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/graphs/histogram_{x_column}_vs_{hue_column}.png'

                    plt.savefig(path)
                    plt.show()
        except Exception as e:
            self.logger.error(f"An error occurred in plot_histogram_within_categories method: {e}")

    
#     Not Used
#     def missing_values(self):
#         # Display the count of missing values in each column
#         if self.data.isnull().sum().any():
#             print("\nMissing Values:")
#             print(self.data.isnull().sum())
#             return True
#         else:
#             print("No missing values found")
#             return False
        
            
    def msno_plot(self):
        try:
            msno.bar(self.data) # you can see pandas-profilin count part
            plt.title('Count of Values per Column in Dataset for Missing value Analysis', size=16)
            path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/graphs/mising_number_plot.png'
            self.dict_manager.update_value(f'Missing number plot before cleaning ', path)
            plt.savefig(path)
            plt.show()
        except Exception as e:
            self.logger.error(f"An error occurred in msno_plot method: {e}")


    # For Time Series
#     def visualize_trends(self, dependent_var=None, date_index=None):
        
#         numerical_columns = self.data.columns
#         print("!!!!!!",numerical_columns)

#         df = self.data
#         df.set_index(date_index, inplace=True)  # Set the date to the index

#         plt.figure(figsize=(22, 10))
#         df[dependent_var].plot()
#         plt.xlabel('Time')
#         plt.ylabel('Value')
#         plt.title(f"{dependent_var} Time Series Data Trend")
#         plt.legend()
#         path = f'Knowledge/{self.problem_type}/{self.source}/{self.file_name}/graphs/{dependent_var}_trend_graph.png'
#         self.dict_manager.update_value('Trend graph', path)
#         plt.savefig(path)
#         plt.show()
        
        
    def visualize_trends(self, dependent_var=None, date_index=None):
        try:
            numerical_columns = self.data.select_dtypes(include=['float64', 'int64']).columns
            print(numerical_columns)

            df = self.data
            if date_index is not None:
                df.set_index(date_index, inplace=True)  # Set the date to the index

            for variable_name in df[numerical_columns].columns:
                plt.plot(df.index, df[variable_name], label=variable_name)
                plt.xlabel('Time')
                plt.ylabel('Value')

                if dependent_var is not None:
                    plt.title(f'Time Series Data Trend for {variable_name} with {dependent_var}')
                    path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/graphs/{variable_name}_with_{dependent_var}_trend_graph.png'
                    self.dict_manager.update_value(f'Trend graph for {variable_name} with {dependent_var}', path)
                else:
                    plt.title(f'Time Series Data Trend for {variable_name}')
                    path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/graphs/{variable_name}_trend_graph.png'
                    self.dict_manager.update_value(f'Trend graph for {variable_name}', path)

                plt.legend()
                plt.savefig(path)
                plt.show()
        except Exception as e:
            self.logger.error(f"An error occurred in visualize_trends method: {e}")
            

    def identify_and_remove_outliers(self, k=1.5):
        try:
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
        except Exception as e:
            self.logger.error(f"An error occurred in identify_and_remove_outliers method: {e}")

    
#     def seasonal_decompose()   
#         result = seasonal_decompose(df['value_column'], model='additive', period=seasonal_period)
#         result.plot()
#         plt.show()

        
        
    def plot_correlation(self):
        try:
            """
            Calculate the correlation matrix for all columns in a DataFrame.

            Parameters:
            - data_frame (pd.DataFrame): The input DataFrame.

            Returns:
            - pd.DataFrame: The correlation matrix.
            """
            data_frame = self.data
            correlation_matrix = data_frame.corr()
        #         return correlation_matrix

            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
            plt.title('Correlation Heatmap')
            plt.show()
        except Exception as e:
            self.logger.error(f"An error occurred in plot_correlation method: {e}")

    def visualize_distribution(self, num_cols=3):
        try:
            # Visualize the distribution of numerical columns
            numerical_columns = self.data.select_dtypes(include=['float64', 'int64']).columns

            num_columns = len(numerical_columns)

            if num_columns < num_cols:
                # If there are fewer columns than num_cols, plot each histogram separately
                fig, axes = plt.subplots(nrows=1, ncols=num_columns, figsize=(5 * num_columns, 5))

                for idx, column in enumerate(numerical_columns):
                    # Check if there is only one subplot
                    if num_columns == 1:
                        sns.histplot(self.data[column], kde=True, ax=axes)
                        axes.set_title(f'Distribution of {column}')
                    else:
                        sns.histplot(self.data[column], kde=True, ax=axes[idx])
                        axes[idx].set_title(f'Distribution of {column}')

        #                 sns.histplot(self.data[column], kde=True, ax=axes[idx])
        #                 axes[idx].set_title(f'Distribution of {column}')
            else:
                # Use a layout if there are enough columns
                num_rows = -(-num_columns // num_cols)  # Ceiling division to determine the number of rows
                fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5 * num_rows))

                for idx, column in enumerate(numerical_columns):
                    row_idx = idx // num_cols
                    col_idx = idx % num_cols
                    sns.histplot(self.data[column], kde=True, ax=axes[row_idx, col_idx])
                    axes[row_idx, col_idx].set_title(f'Distribution of {column}')

                # Remove empty subplots if the number of plots is not a perfect multiple of num_cols
        #             for idx in range(num_columns, num_rows * num_cols):
        #                 fig.delaxes(axes.flatten()[idx])

                # Remove empty subplots if the number of plots is not a perfect multiple of num_cols
                for idx in range(num_columns, num_rows * num_cols):
                    axes.flatten()[idx].axis('off')



            plt.tight_layout()
            path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/graphs/probability_distributions.png'
            self.dict_manager.update_value(f'Probability distributions ', path)
            plt.savefig(path)
            plt.show()
        except Exception as e:
            self.logger.error(f"An error occurred in visualize_distribution method: {e}")




        
        
#     def visualize_distribution(self):
#         # Visualize the distribution of numerical columns
#         numerical_columns = self.data.select_dtypes(include=['float64', 'int64']).columns
#         for column in numerical_columns:
#             plt.figure(figsize=(8, 6))
#             sns.histplot(self.data[column], kde=True)
#             plt.title(f'Distribution of {column}')
#             plt.show()

#     def visualize_distribution_selected(self,list_col = None):
#         # Visualize the distribution of numerical columns
#         if list_col is not None:
#             data_cur = self.data[list_col]
#             numerical_columns = data_cur.select_dtypes(include=['float64', 'int64']).columns
#             for column in numerical_columns:
#                 plt.figure(figsize=(8, 6))
#                 sns.histplot(self.data[column], kde=True)
#                 plt.title(f'Distribution of {column}')
#                 plt.show()
#         if list_col == []:
#             print("going to else")
#             self.visualize_distribution()

#     Not used
#     def plot_distribution_of_categorical(self, column_name):
#         # Count the occurrences of each category in the specified column
#         try:
#             column_counts = self.data[column_name].value_counts()

#             # Plot the distribution using seaborn
#             plt.figure(figsize=(10, 6))
#             sns.barplot(x=column_counts.index, y=column_counts.values, palette="viridis")
#             plt.title(f'Distribution of {column_name}')
#             plt.xlabel(column_name)
#             plt.ylabel('Count')
#             plt.show()
#         except:
#             pass


#     def visualize_pairplot(self):
#         sns.pairplot(self.data)
#         plt.savefig(f'Knowledge/{self.problem_type}/{self.source}/{self.file_name}/graphs/pairwise__plots.png')
#         plt.show()

    def pairwise_plots(self):
        try:
            sns.pairplot(self.data)
            path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/graphs/pairwise_plots.png'
            self.dict_manager.update_value(f'Pairwise plots ', path)
            plt.savefig(path)
            plt.show()
        except Exception as e:
            self.logger.error(f"An error occurred in pairwise_plots method: {e}")

    
    
#     def plot_time_series(self, dependent_variable, data, title='Time Series Plot', xlabel='Time'):

#         # Select the column corresponding to the specified timestamp

#     #         # Clean up the MONTH_YEAR column
#     #         data['MONTH_YEAR'] = data['MONTH_YEAR'].str.replace(r'\s*,\s*', ',', regex=True)

#     #         # Convert to datetime
#     #         data['MONTH_YEAR'] = pd.to_datetime(data['MONTH_YEAR'], format='%B,%Y')

#         x = data.select_dtypes(include='datetime64[ns]')
#         if x is None:
#             x = data[:,0]

#         # Convert object columns to numeric
#         data = data.apply(pd.to_numeric, errors='coerce')


#         plt.figure(figsize=(12, 6))
#         plt.plot(x, data[dependent_variable])
#         plt.title(title)
#         plt.xlabel(xlabel)
#         plt.ylabel(dependent_variable)
#         plt.grid(True)
#         plt.show()


    #     def plot_time_series(self, data, title='Time Series Plot', xlabel='Time'):

    #         # Select the column corresponding to the specified timestamp
    #         x = data.select_dtypes(include='datetime64[ns]')

    #         # Convert object columns to numeric
    #         data = data.apply(pd.to_numeric, errors='coerce')

    #         # Select numerical columns
    #         numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns

    #         # Generate arrays for numerical columns
    #         numerical_arrays = [data[col].values for col in numerical_columns]
    #         print("Numerical arrays", numerical_arrays)

    #         # Display the result
    #         for col, arr in zip(numerical_columns, numerical_arrays):
    #             print(f"{col}: {arr}")

    #             print("X column: ", x)
    #             print("Y column: ", arr)

    #             plt.figure(figsize=(12, 6))
    #             plt.plot(x, arr)
    #             plt.title(title)
    #             plt.xlabel(xlabel)
    #             plt.ylabel(col)
    #             plt.grid(True)
    #             plt.show()


    
    
#     Not used
#     def plot_comparison(df, variable1, variable2, title='Comparison Plot'):
#         """
#         Generate a comparison plot of two variables over time using the DataFrame index.

#         Parameters:
#         - df: Pandas DataFrame with a datetime index.
#         - variable1: Column name for variable 1.
#         - variable2: Column name for variable 2.
#         - variable1_label: Label for variable 1 (default: 'Variable 1').
#         - variable2_label: Label for variable 2 (default: 'Variable 2').
#         - title: Plot title (default: 'Comparison Plot').
#         """
#         plt.figure(figsize=(10, 6))
#         plt.plot(df.index, df[variable1], label=variable1)
#         plt.plot(df.index, df[variable2], label=variable2)
#         plt.xlabel('Time')
#         plt.ylabel('Values')
#         plt.title(title)
#         plt.legend()
#         plt.grid(True)
#         plt.show()


    def plot_seasonal_decomposition(self, dependent_variable, freq='D'):
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose

            date_col = None
            for column in self.data.columns:
                try:
                    self.data[column] = pd.to_datetime(self.data[column])
                    date_col = column
                    break  # Stop loop if conversion is successful for any column
                except ValueError:
                    pass  # Ignore errors and continue to the next column

            print(date_col)

            for column in dependent_variable:
                time_series = self.data.set_index(date_col)[column]
                decomposition = seasonal_decompose(time_series, period=freq)
                trend = decomposition.trend
                seasonal = decomposition.seasonal
                residual = decomposition.resid

                plt.figure(figsize=(12, 8))

                plt.subplot(411)
                plt.plot(column, label='Original')
                plt.legend(loc='upper left')
                plt.title('Original Time Series')

                plt.subplot(412)
                plt.plot(trend, label='Trend')
                plt.legend(loc='upper left')
                plt.title('Trend Component')

                plt.subplot(413)
                plt.plot(seasonal, label='Seasonality')
                plt.legend(loc='upper left')
                plt.title('Seasonality Component')

                plt.subplot(414)
                plt.plot(residual, label='Residuals')
                plt.legend(loc='upper left')
                plt.title('Residuals')

                plt.tight_layout()
                plt.show()
        except Exception as e:
            self.logger.error(f"An error occurred in plot_seasonal_decomposition method: {e}")

    def correlation_analysis(self):
        try:
            # Display a heatmap to visualize the correlation matrix
            features = self.data[1:]
            correlation_matrix = features.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title('Correlation Matrix')
            path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/graphs/correlation_heatmaps.png'
            self.dict_manager.update_value(f'Correlation Heatmap ', path)
            plt.savefig(path)
            plt.show()
            print(correlation_matrix)
            return correlation_matrix
        except Exception as e:
            self.logger.error(f"An error occurred in correlation_analysis method: {e}")

    
    def most_correlated_features(self, correlation_matrix, threshold=0.4):
        try:
            """
            Get the most correlated features from a correlation matrix.

            Parameters:
            - correlation_matrix: pandas DataFrame or 2D array
              The input correlation matrix.
            - threshold: float, optional (default=0.8)
              The correlation coefficient threshold to consider for high correlation.

            Returns:
            - List of tuples, where each tuple contains the names of the most correlated features
              along with their correlation coefficient.
            """
            if not isinstance(correlation_matrix, pd.DataFrame):
                correlation_matrix = pd.DataFrame(np.array(correlation_matrix))

                # Get the upper triangle of the correlation matrix (excluding the diagonal)
                upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))

                # Find pairs of highly correlated features based on the threshold
                correlated_features = [(col1, col2, upper_triangle.loc[col1, col2])
                                       for col1 in upper_triangle.columns
                                       for col2 in upper_triangle.columns
                                       if abs(upper_triangle.loc[col1, col2]) > threshold]

                print("Most correlated features:")
                for feature_pair in correlated_features:
                    print(f"{feature_pair[0]} and {feature_pair[1]} with correlation: {feature_pair[2]}")

                json_filename = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/json/correlated_features.json'

                # Create a dictionary to hold the correlated features
                correlated_dict = [{"Feature1": feature_pair[0], "Feature2": feature_pair[1], "Correlation": feature_pair[2]} for feature_pair in correlated_features]

                # Write the correlated features to the JSON file
                with open(json_filename, 'w') as jsonfile:
                    json.dump(correlated_dict, jsonfile, indent=2)
        except Exception as e:
            self.logger.error(f"An error occurred in most_correlated_features method: {e}")

    def plot_autocorrelation_for_float_columns(self, df, dependent_variable, lags=None):
        try:
            """
            Plots the autocorrelation function (ACF) for all float columns in a DataFrame.

            Parameters:
            - df: pandas DataFrame
              The DataFrame containing the time series data.
            - lags: int or None, optional (default=None)
              The number of lags to include in the plot. If None, all lags will be included.

            Returns:
            - None
            """
            float_columns = self.data.select_dtypes(include=['float64', 'int64']).columns
            lag_values = {}

            for column in dependent_variable:
                plt.figure(figsize=(12, 6))
                ax = plt.gca()
                acf_result = plot_acf(self.data[column], lags=lags, title=f'Autocorrelation Function (ACF) - {column}', zero=False, ax=ax)
                plt.xlabel('Lag')
                plt.ylabel('Autocorrelation')
                plt.show()
        except Exception as e:
            self.logger.error(f"An error occurred in plot_autocorrelation_for_float_columns method: {e}")

    def time_indexing(self, data, index_col):
        try:
            """
            Automatically set the datetime column as the index of the DataFrame.

            Returns:
            - indexed_df
            """
            indexed = data.set_index(index_col)
            return indexed
        except Exception as e:
            self.logger.error(f"An error occurred in time_indexing method: {e}")
    
    
    def convert_degrees(self, coord_str):
        """
        Convert coordinates in the format 'X.XXN' or 'X.XXS' to numeric values.
        """
        value, direction = float(coord_str[:-1]), coord_str[-1]
        return value if direction in ['N', 'E'] else -value

    def convert_degrees(self, coord_str):
        """
        Convert coordinates in the format 'X.XXN' or 'X.XXS' to numeric values.
        """
        value, direction = float(coord_str[:-1]), coord_str[-1]
        return value if direction in ['N', 'E'] else -value

    def plot_choropleth_map(self, cmap='viridis', title='Choropleth Map'):
        try:
            """
            Plot a choropleth map based on latitude and longitude columns in the dataset.

            Parameters:
            - cmap (str): The colormap to be used for the choropleth map.
            - title (str): The title of the plot.

            Returns:
            - None
            """
            latitude_col = None
            longitude_col = None

            for col in self.data.columns:
                if 'lat' in col.lower() or 'latitude' in col.lower():
                    latitude_col = col
                if 'lon' in col.lower() or 'longitude' in col.lower():
                    longitude_col = col

            # Identify latitude and longitude columns
            if latitude_col not in self.data.columns or longitude_col not in self.data.columns:
                print("Latitude and/or longitude columns not found in the dataset.")
                return

            # Check if latitude and longitude columns are numeric
            if pd.api.types.is_numeric_dtype(self.data[latitude_col]) and pd.api.types.is_numeric_dtype(self.data[longitude_col]):
                # If numeric, directly plot the data
                gdf = gpd.GeoDataFrame(self.data, geometry=gpd.points_from_xy(self.data[longitude_col], self.data[latitude_col]))
            else:
                # If not numeric, convert coordinates to numeric values
                self.data[latitude_col] = self.data[latitude_col].apply(self.convert_degrees)
                self.data[longitude_col] = self.data[longitude_col].apply(self.convert_degrees)
                gdf = gpd.GeoDataFrame(self.data, geometry=gpd.points_from_xy(self.data[longitude_col], self.data[latitude_col]))

            # Set the coordinate reference system (CRS)
            gdf.crs = "EPSG:4326"  # WGS 84

            # Plot the choropleth map
            fig, ax = plt.subplots(figsize=(10, 8))

            # Ensure the world map is plotted for reference
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            world.plot(ax=ax, color='lightgrey')

            # Plot the data points
            gdf.plot(ax=ax, marker='o', color='red', markersize=50, alpha=0.5)

            plt.title(title)
            plt.show()
        except Exception as e:
            self.logger.error(f"An error occurred in plot_choropleth_map method: {e}")