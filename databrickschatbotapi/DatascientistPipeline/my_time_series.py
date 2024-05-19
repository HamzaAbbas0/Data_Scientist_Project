import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from keras.backend import clear_session

class Modeling:
    def __init__(self, source, df, file_name, problem_type, dependent_var):
        self.source = source
        self.file_name = file_name
        self.data = df
        self.problem_type = problem_type
        self.dependent_var = dependent_var
        # LSTM related variables
        self.scaled_data = None
        self.num_features = None  
        self.sequence_length = None  
        # training related variables
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.test_size = 0.2
        self.lstm_units = 50
        self.epochs = 100
        self.batch_size = 32
        self.validation_data = None

    def run_modeling(self):
        try:
            if self.problem_type.lower() == "time series":
                self.time_series()
        except Exception as e:
            print(f"An error occurred while running the model: {str(e)}")
            print("Skipping model process.")

    def time_series(self):
        self.scale_data()
        self.X_data, self.Y_data = self.create_sequences(self.scaled_data)
        self.X_train, self.X_test, self.Y_train, self.Y_test = self.train_test_splitter(self.X_data, self.Y_data, test_size=self.test_size)
        self.build_lstm_model()
        self.train_lstm_model()
        self.evaluate_lstm_model()

    def scale_data(self):
        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(self.data)

    def create_sequences(self, dataset):
        X, Y = [], []
        for i in range(len(dataset) - self.sequence_length):
            X.append(dataset[i:(i + self.sequence_length), :-1])
            Y.append(dataset[i + self.sequence_length, -1])
        return np.array(X), np.array(Y)

    def train_test_splitter(self, X, Y, test_size=0.2):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=test_size, shuffle=False)
        return self.X_train, self.X_test, self.Y_train, self.Y_test

    def build_lstm_model(self):
        self.model = Sequential()
        self.model.add(LSTM(units=self.lstm_units, input_shape=(self.sequence_length, self.num_features)))
        self.model.add(Dense(units=1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train_lstm_model(self):
        if self.validation_data is None:
            self.validation_data = (self.X_test, self.Y_test)
        self.history = self.model.fit(self.X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch_size,
                                      validation_data=self.validation_data, verbose=1)

    def evaluate_lstm_model(self):
        test_loss = self.model.evaluate(self.X_test, self.Y_test)
        print(f'Test Loss: {test_loss}')
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.Y_test, y_pred)
        print(f'Mean Squared Error: {mse}')
        mae = mean_absolute_error(self.Y_test, y_pred)
        print(f'Mean Absolute Error: {mae}')

        plt.figure(figsize=(12, 6))
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss Over Epochs')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()

        clear_session()
