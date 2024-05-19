import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import joblib
import os

class LSTMForecaster:
#     def __init__(self, sequence_length=1, target_variable=None, time_series_column=None, prediction_date_input=None, new_data=None):
    def __init__(self,file_name, problem_type=None, source=None,target_variable=None, sequence_length=1):
        # Initialize attributes
        self.problem_type = problem_type
        self.source = source
        self.file_name = file_name
        self.target_variable = target_variable
#         self.time_series_column = time_series_column
#         self.prediction_date_input = prediction_date_input
#         self.new_data = new_data
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = None
        self.default_directory = f'Knowledge/{self.problem_type}/{self.source}/{self.file_name}/models/'
        self.load_model_and_scaler()
    
    def load_model_and_scaler(self):
        scaler_path = os.path.join(self.default_directory, f'scaler_{self.target_variable}.pkl')
        model_path = os.path.join(self.default_directory, f'LSTM_model_{self.target_variable}.h5')
        
        if os.path.exists(scaler_path) and os.path.exists(model_path):
            self.scaler = joblib.load(scaler_path)
            self.model = load_model(model_path)
        else:
            raise FileNotFoundError("Scaler or model file not found.")
    
    def prepare_data_for_lstm(self, data):
        X, Y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length), 0])
            Y.append(data[i + self.sequence_length, 0])
        return np.array(X), np.array(Y)
    
    def create_lstm_model(self, lstm_units=50):
        model = Sequential()
        model.add(LSTM(units=lstm_units, input_shape=(self.sequence_length, 1)))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model 
    
    def train_and_save_model(self, data, model_path, sequence_length=1, epochs=10, batch_size=1):
        X, Y = self.prepare_data_for_lstm(data)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        model = self.create_lstm_model()
        model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=2)
        model.save(model_path)
        self.model = model
    
    def scale_data(self, data):
        if self.scaler:
            return self.scaler.transform(data)
        else:
            raise ValueError("Scaler object is not loaded. Load scaler first.")
    
    def inverse_scale(self, scaled_data):
        if self.scaler:
            return self.scaler.inverse_transform(scaled_data)
        else:
            raise ValueError("Scaler object is not loaded. Load scaler first.")
    
    def predict(self, data, prediction_date):
        scaled_data = self.scale_data(data)
        forecasted_value = self.model.predict(np.reshape(scaled_data[-1], (1, self.sequence_length, 1)))
        return self.inverse_scale(forecasted_value)[0][0]