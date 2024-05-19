import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

class LSTM:
    def __init__(self, data, scaler, sequence_length, test_size=0.3, lstm_units=50, epochs=5):
        self.data = data
        self.sequence_length = sequence_length
#         self.target_column = target_column
        self.test_size = test_size
        self.lstm_units = lstm_units
        self.epochs = epochs
        self.scaler = scaler
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        
    
    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(units=self.lstm_units, input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        self.model.add(Dense(1, activation='relu'))  # Adjust the activation function as needed
        self.model.compile(optimizer='adam', loss='mean_squared_error')  # Adjust the loss function as needed
    
    def train_model(self):
        self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=32)
    
    def evaluate_model(self):
        loss = self.model.evaluate(self.X_test, self.y_test)
        print(f'Test Loss: {loss}')
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

# Example usage:
# Assuming 'df' is your dataset DataFrame