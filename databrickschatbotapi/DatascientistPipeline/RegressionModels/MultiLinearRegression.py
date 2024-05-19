import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression  # Changed import
import numpy as np
from databrickschatbotapi.DatascientistPipeline.JSON_creator import MyDictionaryManager
import json


class MultiLinearRegression:
    def __init__(self, source, X, y, target_variable, scaler, scaler_y, label_encoder, file_name,process_id, problem_type):
        self.source = source
        self.dict_manager = MyDictionaryManager(source, file_name,process_id, problem_type)
        self.target_variable = target_variable
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = self._prepare_data()
        self.mse_history = []
        self.scaler = scaler
        self.scaler_y = scaler_y
        self.label_encoder = label_encoder
        self.file_name = file_name
        self.problem_type = problem_type
        self.model = self._train_regression()
        self.process_id=process_id
        
    def _prepare_data(self):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def _train_regression(self):
        # Train the multi-linear regression model
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        return model
    
    def _predict(self, new_data):
        # Scale numerical columns using the same scaler
        new_data = new_data[self.X.columns]
        
        numeric_columns = new_data.select_dtypes(include=['float64', 'int64']).columns
        new_data[numeric_columns] = self.scaler.transform(new_data[numeric_columns])

        # Encode categorical columns using the same label_encoder
        for column in new_data.select_dtypes(include=['object']).columns:
            new_data[column] = self.label_encoder.fit_transform(new_data[column])

        # Ensure the order of columns is the same as in the training data
        new_data = new_data[self.X.columns]

        # Make predictions using the trained regression model
        predictions = self.model.predict(new_data)

        # Inverse transform to get predictions in the original scale
        predictions_original_scale = self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).ravel()

        return predictions_original_scale

    def evaluate(self):
        # Evaluate the model on the test set
        y_pred = self.model.predict(self.X_test)

        # Inverse transform to get predictions in the original scale
        y_pred_original_scale = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        y_test_original_scale = self.scaler_y.inverse_transform(self.y_test.reshape(-1, 1)).ravel()

        # Calculate Mean Squared Error
        mse = mean_squared_error(y_test_original_scale, y_pred_original_scale)
        
        # Saving the plot
        path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/graphs/true_values_vs_predicted_values_{self.target_variable}.png'
        self.dict_manager.update_value(f"Multi-linear Regression Inference plot ", path)
        plt.savefig(path)

        # Plotting
        plt.scatter(y_test_original_scale, y_pred_original_scale, alpha=0.5)
        plt.title(f'True Values vs Predicted Values of {self.target_variable}')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.show()
        

        
        # Saving the model
        model_path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/models/multi_linear_regression_model_{self.target_variable}.pkl'
        joblib.dump(self.model, model_path)
        self.dict_manager.update_value(f"Multi-linear Regression trained model path ", model_path)
        
        # Saving other necessary objects
        scaler_x_path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/models/scaler_x_{self.target_variable}.pkl'
        scaler_y_path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/models/scaler_y_{self.target_variable}.pkl'
        encoder_path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/models/label_encoder_{self.target_variable}.pkl'
        column_names_path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/models/column_names_{self.target_variable}.txt'
        
        joblib.dump(self.scaler, scaler_x_path)
        joblib.dump(self.scaler_y, scaler_y_path)
        joblib.dump(self.label_encoder, encoder_path)
        np.savetxt(column_names_path, self.X.columns, fmt='%s', delimiter=', ')
        
        # Updating paths in dictionary
        self.dict_manager.update_value(f"Multi-linear Regression X_scaler path ", scaler_x_path)
        self.dict_manager.update_value(f"Multi-linear Regression y_scaler path ", scaler_y_path)
        self.dict_manager.update_value(f"Multi-linear Regression label encoder path ", encoder_path)
        self.dict_manager.update_value(f"Multi-linear Regression column names path ", column_names_path)
        
        self.dict_manager.save_dictionary()
        MyDictionaryManager.close_instance()
        return mse


# Example Usage:
# Assuming you have a DataFrame 'df' with your data and 'target' as the target variable
# regression_model = MultiLinearRegression(data=df, target_variable='target')
# predictions = regression_model.predict(new_data=df_new)
# mse = regression_model.evaluate()
