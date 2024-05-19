import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from databrickschatbotapi.DatascientistPipeline.JSON_creator import MyDictionaryManager
import json
from keras.backend import clear_session
import torch
import tensorflow as tf




class NeuralNetworkRegression:
    def __init__(self, source, X, y, target_variable, scaler, scaler_y, label_encoder, file_name, problem_type, epoch):
#         self.dependent_var = target_variable
        self.source = source
        self.dict_manager = MyDictionaryManager(source, file_name, problem_type)
        self.target_variable = target_variable
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = self._prepare_data()
        self.epoch = epoch
        self.mse_history = []
        self.scaler = scaler
        self.scaler_y = scaler_y
        self.label_encoder = label_encoder
        self.file_name = file_name
        self.problem_type = problem_type
        self.model = self._train_ann(use_gpu=False)
        
        
        
    def _prepare_data(self):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    
    def _train_ann(self, use_gpu=False):
        
        # Set CUDA_VISIBLE_DEVICES to disable GPU if use_gpu is False
#         if not use_gpu:
#             os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        model = Sequential()
        model.add(Dense(128, input_dim=self.X_train.shape[1], activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(4, activation='relu'))
        model.add(Dense(1, activation='linear'))  # Output layer for regression

        model.compile(optimizer='adam', loss='mean_squared_error')
        for epoch in range(self.epoch):
            history = model.fit(self.X_train, self.y_train, epochs=1, batch_size=32, verbose=1)
            mse = history.history['loss'][0]
            self.mse_history.append(mse)

        return model
    
    
    def plot_mse_over_epochs(self):
        # Plotting MSE over epochs
        plt.plot(np.arange(1, self.epoch + 1), self.mse_history, marker='o', linestyle='-')
        plt.title(f'Mean Squared Error over Epochs {self.target_variable[0]}')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.grid(True)
        plt.show()
        path = f'Knowledge/{self.problem_type}/{self.source}/{self.file_name}/graphs/loss_over_epochs_{self.target_variable[0]}.png'
        self.dict_manager.update_value(f"Neural regressor training loss graph ", path)
        plt.savefig(path)

        
    def _predict(self, new_data):
        # Scale numerical columns using the same scaler
        new_data = new_data[self.X.columns]
        
        numeric_columns = new_data.select_dtypes(include=['float64', 'int64']).columns
        print("numeric_columns: ", numeric_columns)
        new_data[numeric_columns] = self.scaler.transform(new_data[numeric_columns])

        # Encode categorical columns using the same label_encoder
        for column in new_data.select_dtypes(include=['object']).columns:
            new_data[column] = self.label_encoder.fit_transform(new_data[column])

        # Ensure the order of columns is the same as in the training data
        new_data = new_data[self.X.columns]

        # Make predictions using the trained ANN model
        predictions = self.model.predict(new_data)

        # Inverse transform to get predictions in the original scale
        predictions_original_scale = self.scaler_y.inverse_transform(predictions).ravel()

        return predictions_original_scale

    def evaluate(self):
        # Evaluate the model on the test set
        y_pred = self.model.predict(self.X_test).ravel()

        # Inverse transform to get predictions in the original scale
        y_pred_original_scale = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        y_test_original_scale = self.scaler_y.inverse_transform(self.y_test.reshape(-1, 1)).ravel()

        # Calculate Mean Squared Error
        mse = mean_squared_error(y_test_original_scale, y_pred_original_scale)

        # Plotting
        plt.scatter(y_test_original_scale, y_pred_original_scale, alpha=0.5)
        plt.title(f'True Values vs Predicted Values of {self.target_variable[0]}')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.show()
        path = f'Knowledge/{self.problem_type}/{self.source}/{self.file_name}/graphs/true_values_vs_predicted_values_{self.target_variable[0]}.png'
        self.dict_manager.update_value(f"Neural Regressor Inference plot ", path)
        plt.savefig(path)
        model_path = f'Knowledge/{self.problem_type}/{self.source}/{self.file_name}/models/ann_model_{self.target_variable[0]}.h5'
        scaler_x_path = f'Knowledge/{self.problem_type}/{self.source}/{self.file_name}/models/scaler_x_{self.target_variable[0]}.pkl'
        scaler_y_path = f'Knowledge/{self.problem_type}/{self.source}/{self.file_name}/models/scaler_y_{self.target_variable[0]}.pkl'
        encoder_path = f'Knowledge/{self.problem_type}/{self.source}/{self.file_name}/models/label_encoder_{self.target_variable[0]}.pkl'
        column_names_path = f'Knowledge/{self.problem_type}/{self.source}/{self.file_name}/models/column_names_{self.target_variable[0]}.txt'
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_x_path)
        joblib.dump(self.scaler_y, scaler_y_path)
        joblib.dump(self.label_encoder, encoder_path)
        np.savetxt(column_names_path, self.X.columns, fmt='%s', delimiter=', ')
#         self.dict_manager.save_dictionary()
        
#         self.dict_manager.update_value(f"Neural Regressor {self.target_variable[0]} trained model path ", model_path)
#         self.dict_manager.update_value(f"Neural Regressor {self.target_variable[0]} X_scaler path ", scaler_x_path)
#         self.dict_manager.update_value(f"Neural Regressor {self.target_variable[0]} y_scaler path ", scaler_y_path)
#         self.dict_manager.update_value(f"Neural Regressor {self.target_variable[0]} label encoder path ", encoder_path)
#         self.dict_manager.update_value(f"Neural Regressor {self.target_variable[0]} column names path ", column_names_path)
        
        self.dict_manager.update_value(f"Neural Regressor trained model path ", model_path)
        self.dict_manager.update_value(f"Neural Regressor X_scaler path ", scaler_x_path)
        self.dict_manager.update_value(f"Neural Regressor y_scaler path ", scaler_y_path)
        self.dict_manager.update_value(f"Neural Regressor label encoder path ", encoder_path)
        self.dict_manager.update_value(f"Neural Regressor column names path ", column_names_path)
        
        self.dict_manager.save_dictionary()
        MyDictionaryManager.close_instance()
        print("clearing the gpu space of the model.!!!!!")
        clear_session()
        clear_session()
        clear_session()
        clear_session()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        return mse


# Example Usage:
# Assuming you have a DataFrame 'df' with your data and 'target' as the target variable
# ann_model = NeuralNetworkRegression(data=df, target_variable='target')
# predictions = ann_model.predict(new_data=df_new)
# mse = ann_model.evaluate()
