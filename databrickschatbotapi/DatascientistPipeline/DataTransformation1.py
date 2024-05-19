import pandas as pd
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib
import numpy as np

class DataTransformer:
    def __init__(self, source, df, file_name,process_id, problem_type, type_column, dependent_var, date_index):
        self.source = source
        self.file_name = file_name
        self.data = df
        self.problem_type = problem_type
        self.type_column = date_index
        self.dependent_var = dependent_var
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.scaler_y = MinMaxScaler()
        self.label_encoder_y = LabelEncoder()
        self.transformed_data = None
        self.transformed_data_y = None
        self.numeric_data = None
        self.numeric_columns = None
        self.categorical_columns = None
        self.X = None
        self.y = None
        self.date_index = date_index
        self.transformed_data = None
        self.transformed_data_y =  None
        self.process_id=process_id

        if problem_type.lower() == "time series":
            self.time_series()

        elif problem_type.lower() == "numerical":
            self.numerical()

        elif problem_type.lower() == "categorical":
            self.categorical()


    def numerical(self):
        try:
            self.transformed_data, self.transformed_data_y = self.transform_data_for_numerical() 
        except Exception as e:
            print(f"Error occurred during numerical preprocessing: {e}")


    def categorical(self):
        try:
            self.transformed_data, self.transformed_data_y = self.transform_data(self.data)
        except Exception as e:
            print(f"Error occurred during categorical preprocessing: {e}")

    def time_series(self):
        try:
            scale_range = (0,1)
#             self.transformed_data = self.timeseries_data_transform(scale_range) 
            self.transformed_data = self.my_timeseries_data_transform(scale_range) 
        except Exception as e:
            print(f"Error occurred during time series preprocessing: {e}") 


    def get_data(self):
        return self.data

    def get_transformed_data(self):
        return self.transformed_data

    def get_filled_scaler(self):
        return self.scaler

    def get_filled_scaler_y(self):
        return self.scaler_y

    def get_filled_encoder(self):
        return self.label_encoder
    
        # Convert to sequences
    def create_sequences(self,data, sequence_length):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length, :]) # All features except the target
        return np.array(X)


    def timeseries_data_transform(self, scale_range=(0, 1), sequence_length=8):
        ts = self.data.copy()  # Ensure we are working with a copy of the original data
        

        datetime_columns = ts.select_dtypes(include=['datetime64', 'datetime', 'timedelta64']).columns

        if self.dependent_var in datetime_columns:
            datetime_columns = datetime_columns.drop(self.dependent_var)

        datetime_indices = ts.index  # Store datetime indices before dropping columns
        ts.drop(columns=datetime_columns, inplace=True)

        if 'Unnamed: 0' in ts.columns:
            ts.drop(['Unnamed: 0'], axis=1, inplace=True)

        self.categorical_columns = ts.select_dtypes(include=['object']).columns
        if not self.categorical_columns.empty:
            for column in self.categorical_columns:
                ts[column] = self.label_encoder.fit_transform(ts[column])

        # Initialize and fit Min-Max scaler
        scaler = MinMaxScaler(feature_range=scale_range)
        scaled_data = scaler.fit_transform(ts)

        # Convert scaled data back to a DataFrame
        scaled_df = pd.DataFrame(scaled_data, columns=ts.columns)

        # Reset datetime indices
        scaled_df.index = datetime_indices

        self.data = scaled_df

        return self.data
    
    def my_timeseries_data_transform(self, scale_range=(0, 1), sequence_length=1):
        ts = self.data.copy()
        if 'Unnamed: 0' in ts.columns:
            ts.drop(['Unnamed: 0'], axis=1, inplace=True)
        ts.set_index(self.date_index, inplace=True)
#         scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit(ts[[self.dependent_var]])  # Fit scaler to training data only
        transformed_data = self.scaler.transform(ts[[self.dependent_var]])
        return transformed_data
        


    def set_date_as_index1(self):
        df = self.data
        print("before Set Index",df)
        df= df.set_index(self.date_index, inplace = True) #Set the date to the index
        print("HHHHHHHHHHHHHHHHHHHHHHHH")
        df.head(12)
        #numeric_columns = self.data.select_dtypes(include=['int', 'float']).columns
        
        self.numeric_data = self.df
        return df

    def create_lags(self, dependent_variable, max_lag=2):
        """
        Create lag features for all numeric columns in the dataset.

        Parameters:
        - max_lag (int): Maximum lag value to be created for each column.
        """
        # numeric_columns = self.data.select_dtypes(include=['float64', 'int64']).columns
        # print("Num cols: ", numeric_columns)

        self.data =  self.data[dependent_variable]

        for col in self.data[dependent_variable]:
            print(col)
            for lag in range(1, max_lag + 1):
                self.data[f'{col}_lag_{lag}'] = self.data[col].shift(lag)


        self.data = self.data.dropna()
        print("After lags: ", self.data)


    def create_time_features(self):
        """
        Create time-based features from the first datetime column found in the dataset.
        """
        for col in self.data.columns:
            try:
                datetime_series = pd.to_datetime(self.data[col])
                self.data['year'] = datetime_series.dt.year
                self.data['month'] = datetime_series.dt.month
                self.data['day'] = datetime_series.dt.day
                self.data['hour'] = datetime_series.dt.hour
                self.data['weekday'] = datetime_series.dt.weekday
                break  # Stop after the first datetime column is found and features are created
            except (TypeError, ValueError):
                pass  # Continue if the conversion fails for the current column
        else:
            print("No datetime column found in the dataset.")


#     def scale_features(self):
#         """
#         Scale all numeric columns using StandardScaler.
#         """
#         numeric_columns = self.data.select_dtypes(include=['float64', 'int64']).columns

#         scaler = StandardScaler()
#         self.data[numeric_columns] = scaler.fit_transform(self.data[numeric_columns])


    def drop_columns(self, threshold=0.5):
        """
        Drop columns with missing values exceeding a specified threshold.

        Parameters:
        - threshold (float): Threshold for the percentage of missing values.
        """
        cols_to_drop = self.data.columns[self.data.isnull().mean() > threshold]
        self.data.drop(columns=cols_to_drop, inplace=True)


    def reset_index(self):
        """
        Reset the index of the DataFrame.
        """
        self.data.reset_index(drop=True, inplace=True)


    def set_datetime_as_index(self):
        """
        Automatically set the datetime column as the index of the DataFrame.

        Returns:
        - None
        """

        self.data.set_index(self.type_column, inplace=True)



    def transform_data_for_numerical(self):
        """
        Perform data transformation on the input_data.

        Parameters:
        - input_data: pandas DataFrame

        Returns:
        - transformed_data: pandas DataFrame
        """
        self.X = self.data.drop(columns=self.dependent_var)
        self.y = self.data[self.dependent_var]

        # Example: Scaling numerical columns using MinMaxScaler

        self.numeric_columns = self.X.select_dtypes(include=['float64', 'int64']).columns
#         scaler = MinMaxScaler()
        if not self.numeric_columns.empty:
            self.X[self.numeric_columns] = self.scaler.fit_transform(self.X[self.numeric_columns])

#         scaler_y = MinMaxScaler()
        self.y = self.scaler_y.fit_transform(self.y.values.reshape(-1, 1)).ravel()

        # Example: Encoding categorical columns using LabelEncoder
        self.categorical_columns = self.X.select_dtypes(include=['object']).columns
#         label_encoder = LabelEncoder()
        if not self.categorical_columns.empty:
            for column in self.categorical_columns:
                self.X[column] = self.label_encoder.fit_transform(self.X[column])

        # return self.X, self.y, self.scaler, self.scaler_y, self.label_encoder
        return self.X, self.y

#     def transform_categorical_features(self):
#         # Make a copy of the input data to avoid modifying the original DataFrame
#         df = self.data

#         # Iterate through each column and check if it's categorical
#         for column in df.columns:
#             if df[column].dtype == 'object':
#                 # Check the number of unique values in the column
#                 unique_values = df[column].nunique()

#                 # Use different encoding methods based on the number of unique values
#                 if unique_values <= 10:
#                     # Use Label Encoding for columns with 10 or fewer unique values
#                     le = LabelEncoder()
#                     df[column] = le.fit_transform(df[column])
#                 elif unique_values <= 50:
#                     # Use Binary Encoding for columns with 50 or fewer unique values
#                     encoder = ce.BinaryEncoder(cols=[column])
#                     df = encoder.fit_transform(df)
#                 else:
#                     # Use Target Encoding for columns with more than 50 unique values
#                     encoder = ce.TargetEncoder(cols=[column])
#                     df = encoder.fit_transform(df, y=df['target_column'])

#         return df


    def transform_data(self, df):
            """
            Perform data transformation on the input_data.

            Parameters:
            - df: pandas DataFrame

            Returns:
            - df_X: pandas DataFrame (features)
            - df_Y: pandas Series (target variable)
            """
            df_Y = df[self.dependent_var]
            df_X = df.drop(columns=self.dependent_var)

            # Example: Scaling numerical columns using MinMaxScaler
            numeric_columns = df_X.select_dtypes(include=['float64', 'int64']).columns
            df_X[numeric_columns] = self.scaler.fit_transform(df_X[numeric_columns])

            # Example: Encoding categorical columns using LabelEncoder
            categorical_columns = df_X.select_dtypes(include=['object']).columns
            df_X[categorical_columns] = df_X[categorical_columns].apply(self.label_encoder.fit_transform)

            # Encoding the target variable
            df_Y = self.label_encoder_y.fit_transform(df_Y)

            return df_X, df_Y



#########my transform_Data_work

#     def transform_data(self, df):
#         """
#         Perform data transformation on the input_data.

#         Parameters:
#         - input_data: pandas DataFrame

#         Returns:
#         - transformed_data: pandas DataFrame
#         """
#         df_Y = df[self.dependent_var]
#         df_X = df.drop(columns=self.dependent_var)

#         # Example: Scaling numerical columns using MinMaxScaler
#         self.numeric_columns = df_X.select_dtypes(include=['float64', 'int64']).columns
#         df_X[self.numeric_columns] = self.scaler.fit_transform(df_X[self.numeric_columns])

#         # Example: Encoding categorical columns using LabelEncoder
#         self.categorical_columns = df_X.select_dtypes(include=['object']).columns

#         # Encode the categorical attributes
#         df_X[self.categorical_columns] = df_X[self.categorical_columns].apply(lambda col: self.label_encoder.fit_transform(col))

#         # Ensure df_Y is a 1D array
#         df_Y = df_Y.iloc[:, 0]

#         # Encode the target variable using LabelEncoder
#         df_Y = self.label_encoder_y.fit_transform(df_Y)

#         return df_X, df_Y



#     def set_datetime_as_index(self):
#         """
#         Automatically set the datetime column as the index of the DataFrame.

#         Returns:
#         - None
#         """
#         # Check if any column has datetime dtype
#         datetime_columns = self.data.select_dtypes(include='datetime64[ns]').columns

#         if len(datetime_columns) == 0:
#             print("No datetime column found in the DataFrame.")

#         # If there are multiple datetime columns, select the first one
#         else:
#             datetime_column = datetime_columns[0]

#             # Set the datetime column as the index
#             self.data.set_index(datetime_column, inplace=True)