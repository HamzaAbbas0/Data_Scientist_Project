import joblib
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler


class SupportVectorRegression:
    def __init__(self, X, y, target_variable, scaler, scaler_y, label_encoder, file_name, problem_type):
        
        self.target_variable = target_variable
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = self._prepare_data()
        self.model = self._train_svr()
        self.scaler = scaler
        self.scaler_y = scaler_y
        self.label_encoder = label_encoder
        self.file_name = file_name
        self.problem_type = problem_type

    def _prepare_data(self):     

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
       
        return X_train, X_test, y_train, y_test

    def _train_svr(self):
        # Train the Support Vector Regression model
        svr = SVR(kernel='rbf')  # You can customize the SVR parameters here
        print("self.X_train: ", self.X_train)
        print("self.y_train: ", self.y_train)
        svr.fit(self.X_train, self.y_train)
        return svr

    def _predict(self, new_data):
        new_data = new_data[self.X.columns]
        print("self.X.columns: ", self.X.columns)
        print("new_data:", new_data)

        # Scale numerical columns using the same scaler
        # Separate numerical and categorical columns
        numeric_columns = new_data.select_dtypes(include=['float64', 'int64']).columns
        print("numeric_columns: ", numeric_columns)
        categorical_columns = new_data.select_dtypes(include=['object']).columns
        print("categorical_columns: ", categorical_columns)

        # Scale numerical columns
        new_data[numeric_columns] = self.scaler.transform(new_data[numeric_columns])

        # Encode categorical columns using the same label_encoder
        for column in categorical_columns:
            # Handle unseen labels using fit_transform
            new_data[column] = self.label_encoder.fit_transform(new_data[column])

        # Ensure the order of columns is the same as in the training data
        new_data = new_data[self.X.columns]
        print("new_data:", new_data)    

        # Make predictions using the trained SVR model
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

        # Plotting
        plt.scatter(y_test_original_scale, y_pred_original_scale, alpha=0.5)
        plt.title('True Values vs Predicted Values')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.show()
        plt.savefig(f'{self.problem_type}/graphs/{self.file_name}_true_values_vs_predicted_values.png')

        joblib.dump(self.model, f'{self.problem_type}/models/{self.file_name}_svr_model.h5')
        return mse

# Example Usage:
# Assuming you have a DataFrame 'df' with your data and 'target' as the target variable
# svr_model = SupportVectorRegression(data=df, target_variable='target')
# predictions = svr_model.predict(new_data=df_new)
# mse = svr_model.evaluate()
