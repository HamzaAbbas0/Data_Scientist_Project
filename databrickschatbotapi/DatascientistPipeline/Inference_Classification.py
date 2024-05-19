from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

class Inference_Classification:
    def __init__(self, df, problem_type, source, file_name, target_variable):
        
        self.data = df
        self.problem_type = problem_type
        self.file_name = file_name
        self.source = source
        self.target_variable = target_variable
        self.result = ""
             

    def classification_model(self, input_data = None):
        
        if input_data is not None:
            model_path = f'Knowledge/{self.problem_type}/{self.source}/{self.file_name}/models/RFC_model_{self.target_variable}.joblib'
            scaler_x_path = f'Knowledge/{self.problem_type}/{self.source}/{self.file_name}/models/scaler_x_{self.target_variable}.pkl'
            x_encoder_path = f'Knowledge/{self.problem_type}/{self.source}/{self.file_name}/models/x_encoder_{self.target_variable}.pkl'
            y_encoder_path = f'Knowledge/{self.problem_type}/{self.source}/{self.file_name}/models/y_encoder_{self.target_variable}.pkl'
            column_names_path = f'Knowledge/{self.problem_type}/{self.source}/{self.file_name}/models/column_names_{self.target_variable}.txt'

            model = joblib.load(model_path)
            scaler = joblib.load(scaler_x_path)
            label_encoder_y = joblib.load(y_encoder_path)
            label_encoder = joblib.load(x_encoder_path)
            try:
                with open(column_names_path, 'r') as file:
                    loaded_list = file.read().strip().split('\n')
            except Exception as e:
                print(f"An error occurred: {e}")

            # Convert the loaded list to a NumPy array if needed
            loaded_array = np.array(loaded_list)

            self.input_data = input_data[loaded_list]
            print(self.input_data)

            numeric_columns = self.input_data.select_dtypes(include=['float64', 'int64']).columns
            self.input_data[numeric_columns] = scaler.fit_transform(self.input_data[numeric_columns])

            for column in self.input_data.select_dtypes(include=['object', 'category']).columns:
                self.input_data[column] = label_encoder.fit_transform(self.input_data[column])

            new_data_processed = self.input_data
#             print(new_data_processed)

            # Make predictions using the loaded ANN model
            predictions = model.predict(new_data_processed)

            y_pred_decoded = label_encoder_y.inverse_transform(predictions)

            self.result = y_pred_decoded
#             print(f"The value of {self.target_variable } is: {self.result[0]}")
            return self.result