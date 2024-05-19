import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import json


class LSTMModel:
    def __init__(self, data, file_name, sequence_length = 8):
        self.file_name = file_name
        self.data = data
        self.sequence_length = sequence_length
        self.model = None
        self.history = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.num_col = data.shape[1]
#         self.target_col = target_col
#         self.feature_cols = feature_cols
#         self.look_back = look_back
        self.scaler = MinMaxScaler()
#         self.model = None

#         self.timestamp_col = self.identify_timestamp_column()
       
        
    
    def data_scaller(self, scale_range = (0,1)):
#         scaler = MinMaxScaler(feature_range=scale_range)
        scaled_data = self.scaler.fit_transform(self.data.values)
        return scaled_data

    def create_sequences(self, dataset):
        X, Y = [], []
        for i in range(len(dataset) - self.sequence_length):
            X.append(dataset[i:(i + self.sequence_length), :-1])
            Y.append(dataset[i + self.sequence_length, -1])
        return np.array(X), np.array(Y)
    
    def train_test_splitter(self, X, Y, test_size=0.2, random_state = 42):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size = test_size, random_state = random_state)
        
        return self.X_train, self.X_test, self.Y_train, self.Y_test
    
    def build_model(self, lstm_units = 50):
#         self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], 1, self.X_train.shape[1]))
#         self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], 1, self.X_test.shape[1]))
        self.model = Sequential()
        self.model.add(LSTM(units=lstm_units, input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        self.model.add(Dense(units=1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
    def train_model(self, epochs=50, batch_size=32, validation_data=None):
        if validation_data is None:
            validation_data = (self.X_test, self.Y_test)
        self.history = self.model.fit(self.X_train, self.Y_train, epochs=epochs, batch_size=batch_size,
                                      validation_data=validation_data, verbose=1)
        self.save_model(f'LSTM_models/{self.file_name}_LSTM_model.h5')
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss Over Epochs')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(f'LSTM_graphs/{self.file_name}_training_loss_graph.png')
        plt.show()
    
    def save_model(self, model_file_path):
        # Save the entire model (architecture, optimizer, and weights)
        self.model.save(model_file_path)
        print(f"Model saved to {model_file_path}")
    
    def evaluation(self, X_test=None, Y_test=None):
        if X_test is None:
            X_test = self.X_test
        if Y_test is None:
            Y_test = self.Y_test
        loss = self.model.evaluate(X_test, Y_test)
        print(f'Test Loss: {loss}')
        return loss
    
# ___Commented due to unusable and have some errors    
#     def evaluate_model(self):
#         predictions = self.model.predict(self.X_test)
#         predictions_inverse = self.scaler.inverse_transform(predictions)

#         y_pred = self.model.predict(self.X_test)
#         y_pred_inverse = self.scaler.inverse_transform(y_pred.reshape(-1, 1))
#         y_test_inverse = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))

#         mse = mean_squared_error(y_test_inverse, y_pred_inverse)
#         mae = mean_absolute_error(y_test_inverse, y_pred_inverse)

#         print(f'Mean Squared Error: {mse}')
#         print(f'Mean Absolute Error: {mae}')

#         plt.figure(figsize=(15, 6))
#         plt.plot(y_test_inverse, label='Actual Values', color='blue')
#         plt.plot(y_pred_inverse, label='Predicted Values', color='red')
#         plt.title('LSTM Forecast Evaluation')
#         plt.xlabel('Time')
#         plt.ylabel('Quantity')
#         plt.legend()
#         plt.show()

    def LSTM_inference(self, X_test=None, Y_test=None):
        if X_test is None:
            X_test = self.X_test
        if Y_test is None:
            Y_test = self.Y_test
            
        # Predict on test data
        y_pred = self.model.predict(X_test)

        # Assuming the original data had 9 features
        num_original_features = int(self.num_col)

        # Reconstruct the y_pred array to match the original number of features
        y_pred_reconstructed = np.zeros((len(y_pred), num_original_features))
        y_pred_reconstructed[:,0] = y_pred.ravel()  # Assuming the target variable is at index 0

        # Inverse transform predictions
        self.y_pred_inv = self.scaler.inverse_transform(y_pred_reconstructed)[:,0]

        # Similarly, reconstruct y_test if it was also scaled
        y_test_reconstructed = np.zeros((len(Y_test), num_original_features))
        y_test_reconstructed[:,0] = Y_test.ravel()
        self.y_test_inv = self.scaler.inverse_transform(y_test_reconstructed)[:,0]

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(self.y_test_inv, self.y_pred_inv))
        print(f"Test RMSE: {rmse}")

        # Calculate MSE and MAE
        mse = mean_squared_error(self.y_test_inv, self.y_pred_inv)
        mae = mean_absolute_error(self.y_test_inv, self.y_pred_inv)
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Mean Absolute Error (MAE): {mae}")

        # Create a dictionary with floating-point values
        metrics = {
            "Root Mean Squared Error (RMSE)": rmse,
            "Mean Squared Error (MSE)": mse,
            "Mean Absolute Error (MAE)": mae,
        }

        # Specify the file path where you want to save the JSON data
        file_path = f'LSTM_result_csv/{self.file_name}_metrics.json'

        # Write the dictionary to a JSON file
        with open(file_path, "w") as json_file:
            json.dump(metrics, json_file)

        print(f"Metrics saved to {file_path}.")


        
    def plot_inference(self):
        # Plot actual vs predicted values
        title='LSTM Model: Actual vs Predicted Test Samples Values'
        plt.figure(figsize=(15, 6))
        plt.plot(self.y_test_inv, label='Actual')
        plt.plot(self.y_pred_inv, label='Predicted')
        plt.title(title)
        plt.ylabel('Value')
        plt.xlabel('Sample Index')
        plt.legend()
        plt.savefig(f'LSTM_graphs/{self.file_name}_inference_plot.png')
        plt.show()
#         plt.savefig('/content/drive/MyDrive/MAL2324_CW_DataSet_Initial/Results/'+title+'graph.png')


        
    
    
    
    
