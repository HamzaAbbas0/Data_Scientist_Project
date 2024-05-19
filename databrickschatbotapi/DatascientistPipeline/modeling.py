import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import json
from sklearn.ensemble import RandomForestClassifier
import joblib
# from ClassificationModels.RandomForestModel import RandomForestModel
from databrickschatbotapi.DatascientistPipeline.RegressionModels.NeuralNetworkRegression import NeuralNetworkRegression
from databrickschatbotapi.DatascientistPipeline.JSON_creator import MyDictionaryManager
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.backend import clear_session
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import BaggingClassifier
import torch

class Modeling:
    def __init__(self, source, df, file_name,process_id, problem_type, type_column, dependent_var):
        self.source = source
        self.dict_manager = MyDictionaryManager(source, file_name,process_id, problem_type)
        self.file_name = file_name
        self.data = df
        self.problem_type = problem_type
        self.type_column = type_column
        self.dependent_var = dependent_var
        # LSTM related variables
        self.scaled_data = None
        self.num_features = None  
        self.sequence_length = None  
        self.y_test = None
        self.y_pred_inv = None
        self.numeric_columns = None
        # Random forest Classifier
        self.baggingrfc_model = None
        self.n_estimators = None
        self.max_depth = None
        # training related variables
        self.transformed_data = None
        self.transformed_data_y = None
        self.X_data = None
        self.Y_data = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.test_size = 0.2
        self.lstm_units = 50
        self.epochs = 100
        self.batch_size = 32
        self.validation_data = None
        self.random_state = None
        # Other variables
        self.scaler = None
        self.label_encoder = None
        self.label_encoder_y = None
        self.process_id=process_id

    def update_attributes(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Attribute '{key}' does not exist in the class.")

    def run_modeling(self):
        try:
            if self.problem_type.lower() == "time series":
                self.time_series()

            if self.problem_type.lower() == "categorical":
                self.categorical()
        except Exception as e:
            print(f"An error occurred while running the model: {str(e)}")
            print("Skipping model process.")

    def time_series(self, encoder, scaler):
        
        print("______Data feeding in the LSTM_____ :\n " ,self.data)
        print("______scaler type in LSTM __________", type(scaler))
        
        scalerpath = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/models/scaler_{self.dependent_var}.pkl'
        self.dict_manager.update_value(f"Scaler path ", scalerpath)
        joblib.dump(scaler, scalerpath)
        
        
#         column_names_path = f'Knowledge/{self.problem_type}/{self.source}/{self.file_name}/models/column_names_{self.dependent_var}.txt'
#         np.savetxt(column_names_path, self.scaled_data.columns, fmt='%s', delimiter=', ')
        
        
#         self.X_data, self.Y_data = self.create_sequences(self.scaled_data)
#         self.X_train, self.X_test, self.Y_train, self.Y_test =self.train_test_splitter1(self.X_data, self.Y_data, test_size = self.test_size)
#         self.build_lstm_model(lstm_units = self.lstm_units)
#         self.train_lstm_model(epochs = self.epochs, batch_size= self.batch_size, validation_data= self.validation_data)
#         test_loss = self.lstm_evaluation()
#         self.my_lstm_evaluation(self.X_test, self.Y_test)

############################## New ADDING FUNCTION THINGS ######################
        self.prepare_data_for_lstm(self.data,1)
        self.create_lstm_model(sequence_length=self.sequence_length,lstm_units = self.lstm_units)
        self.train_and_save_model(self.data, dependent_var=self.dependent_var, sequence_length=self.sequence_length, epochs=self.epochs, batch_size=self.batch_size)

##################################### END ######################################
    #     lstm_model_instance.evaluate_model()
        #self.lstm_inference()
        #self.plot_lstm_inference()
        self.dict_manager.save_dictionary()
        MyDictionaryManager.close_instance()

    def categorical(self):
        
        print("______Data feeding in Random Forest Classifier_____ :\n " , (self.X_data, self.Y_data))
        self.BaggingRandomForestClassifier()
        self.dict_manager.save_dictionary()
        MyDictionaryManager.close_instance()
################### new add function things thing ##########################
    def prepare_data_for_lstm(self,df,sequence_length=1):
        
        X, Y = [], []
        for i in range(len(self.data) - self.sequence_length):
            X.append(self.data[i:(i + self.sequence_length), 0])
            Y.append(self.data[i + self.sequence_length, 0])
        return np.array(X), np.array(Y)
    
    def create_lstm_model(self,sequence_length,lstm_units = 50):
        model = Sequential()
        model.add(LSTM(units=lstm_units, input_shape=(self.sequence_length, 1)))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model 
        
    
    def train_and_save_model(self,data, dependent_var, sequence_length=1, epochs=10, batch_size=1):
        
        X, Y = self.prepare_data_for_lstm(self.data, self.sequence_length)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        model = self.create_lstm_model(self.sequence_length)
        history = model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=2)
        print("model training...........")
        path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/models/LSTM_model_{self.dependent_var}.h5'
        self.dict_manager.update_value(f"LSTM trained model path", path)
#         self.save_model(path)
        model.save(path)
        print("model saved !!!!!")
        
        
            # Plot training loss
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.title('Model Loss Over Epochs')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        loss_graph_path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/graphs/LSTM_training_loss_graph_{self.dependent_var}.png'
        self.dict_manager.update_value("LSTM Training loss graph", loss_graph_path)
        plt.savefig(loss_graph_path)
        plt.show()
    

    
        

#################### END ##########################################
        
        
    def X_Y_splitter(self):
        self.X_data = self.data.drop(self.dependent_var[0], axis=1)
        self.Y_data = self.data[self.dependent_var[0]]
        

    def create_sequences(self, dataset):
        X, Y = [], []
        for i in range(len(dataset) - self.sequence_length):
            # Extract the input sequence (features)
            X.append(dataset.iloc[i:(i + self.sequence_length), :-1].values)
            # Extract the target value
            Y.append(dataset.iloc[i + self.sequence_length, -1])
        return np.array(X), np.array(Y)
    
    #previous code
    def train_test_splitter1(self, X, Y, test_size=0.2, random_state = 42):
        
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size = test_size, random_state = random_state)
        
        # Convert to NumPy arrays
        self.X_train = np.array(self.X_train)
        self.X_test = np.array(self.X_test)
        # Assuming your original shape is (batch_size, sequence_length, num_samples, num_features)
        # Reshape it to (batch_size, sequence_length, num_samples * num_features)
        
        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], -1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], -1)
        self.num_features = self.X_train.shape[2]  # Assuming features are along axis 1
        self.sequence_length = self.X_train.shape[1]
        print("X_Train!!!!!!!!!!!!!1",self.X_train.shape) 
        print("X_Test!!!!!!!!!!!!!1",self.X_test.shape)
        print('num_features!!!!!!!!!!!!!!!',self.num_features)
        print('sequence_length!!!!!!!!!!!!!!!',self.sequence_length)
        return self.X_train, self.X_test, self.Y_train, self.Y_test


    def train_test_splitter(self, X, Y, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

        # Convert to NumPy arrays
        self.X_train = np.array(self.X_train)
        self.X_test = np.array(self.X_test)

        # No need to reshape for 2D arrays
        self.num_features = self.X_train.shape[1]  # Assuming features are along axis 1
        self.sequence_length = self.X_train.shape[0]

        

        return self.X_train, self.X_test, self.Y_train, self.Y_test

    
    def build_lstm_model(self, lstm_units = 50):
#         self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], 1, self.X_train.shape[1]))
#         self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], 1, self.X_test.shape[1]))
        self.model = Sequential()
        self.model.add(LSTM(units=lstm_units, input_shape=(self.sequence_length, self.num_features)))
        self.model.add(Dense(units=1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
    def train_lstm_model(self, epochs=50, batch_size=32, validation_data=None):
        if validation_data is None:
            validation_data = (self.X_test, self.Y_test)
        self.history = self.model.fit(self.X_train, self.Y_train, epochs=epochs, batch_size=batch_size,
                                      validation_data=validation_data, verbose=1)
        path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/models/LSTM_model_{self.dependent_var}.h5'
        column_names_path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/models/column_names_{self.dependent_var}.txt'
        
        scaler_x_path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/models/scaler_x_{self.dependent_var}.pkl'
        scaler_y_path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/models/scaler_y_{self.dependent_var}.pkl'
        encoder_path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/models/label_encoder_{self.dependent_var}.pkl'
        
        self.dict_manager.update_value(f"LSTM trained model path", path)
        self.save_model(path)
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss Over Epochs')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/graphs/LSTM_training_loss_graph_{self.dependent_var}.png'
        self.dict_manager.update_value(f"LSTM Training loss graph ", path)
        plt.savefig(path)
        plt.show()
        print("clearning the lstm gpu space!!!!!")
        clear_session()
        clear_session()
        clear_session()
        clear_session()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        self.model = None
        print("clearning the lstm gpu already cleaned!!!!!")
    
    def save_model(self, model_file_path):
        # Save the entire model (architecture, optimizer, and weights)
        self.model.save(model_file_path)
        print(f"Model saved to {model_file_path}")
        
    def my_lstm_evaluation(self, X_test=None, Y_test=None):
        if X_test is None:
            X_test = self.X_test
        if Y_test is None:
            Y_test = self.Y_test

        if self.model is None:
            return None
        
        test_predict = model.predict(X_test)
        test_predict = scaler.inverse_transform(test_predict)
        print(test_predict[:5])
        
        
        
                           
    
    def lstm_evaluation(self, X_test=None, Y_test=None):
        if X_test is None:
            X_test = self.X_test
        if Y_test is None:
            Y_test = self.Y_test

        if self.model is None:
            return None

        try:
            loss = self.model.evaluate(X_test, Y_test)
            print(f'Test Loss: {loss}')
            from sklearn.metrics import mean_squared_error
            import matplotlib.pyplot as plt

            # Assuming X_test, y_test are your test data and model is your trained LSTM model

            # Predict on test data
            y_pred = self.model.predict(X_test)
            print("hhhhhhhhhhhhhh",y_pred)
            # Assuming 'num_features' is the number of features in the dataset before reshaping for LSTM
            num_features = X_test.shape[2]

            # Assuming the original data had 8 features
            num_original_features = X_test

            # Reconstruct the y_pred array to match the original number of features
            y_pred_reconstructed = np.zeros((len(y_pred), num_original_features))
            y_pred_reconstructed[:,0] = y_pred.ravel()  # Assuming the target variable is at index 0

            # Inverse transform predictions
            y_pred_inv = scaler.inverse_transform(y_pred_reconstructed)[:,0]

            # Similarly, reconstruct y_test if it was also scaled
            y_test_reconstructed = np.zeros((len(y_test), num_original_features))
            y_test_reconstructed[:, 0] = y_test.ravel()
            y_test_inv = scaler.inverse_transform(y_test_reconstructed)[:,0]

            # Calculate the loss (e.g., RMSE)
            rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
            print(f"Test RMSE: {rmse}")


            # Plot actual vs predicted values
            title='LSTM Model: Actual vs Predicted Test Samples Values'
            plt.figure(figsize=(15, 6))
            plt.plot(y_test_inv, label='Actual')
            plt.plot(y_pred_inv, label='Predicted')
            plt.title(title)
            plt.ylabel('Value')
            plt.xlabel('Sample Index')
            plt.legend()
            plt.show()
#             plt.savefig('/content/drive/MyDrive/InssurancePrediction/Results/'+title+'graph.png')

            # Plot training and validation loss from the history
            title='LSTMModel Loss Over Epochs'
            plt.figure(figsize=(12, 6))
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title(title)
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend()
            plt.show()
#             plt.savefig('/content/drive/MyDrive/InssurancePrediction/Results/'+title+'graph.png')

            return loss
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return None

    
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

    def lstm_inference(self, X_test=None, Y_test=None):
        if X_test is None:
            X_test = self.X_test
        if Y_test is None:
            Y_test = self.Y_test
        # Finding index of dependent variable    
        dependent_index_value = int(self.numeric_columns.get_loc(self.dependent_var))
        print("Index of the dependent variable : ", dependent_index_value)
        # Predict on test data
        y_pred = self.model.predict(X_test)

        # Assuming the original data had 9 features
        num_original_features = int(self.scaled_data.shape[1])

        # Reconstruct the y_pred array to match the original number of features
        y_pred_reconstructed = np.zeros((len(y_pred), num_original_features))
        y_pred_reconstructed[:,dependent_index_value] = y_pred.ravel()  # Assuming the target variable is at index 0
#         y_pred_reconstructed[:,0] = y_pred.ravel()  # Assuming the target variable is at index 0

        # Inverse transform predictions
        self.y_pred_inv = self.scaler.inverse_transform(y_pred_reconstructed)[:,0]

        # Similarly, reconstruct y_test if it was also scaled
        y_test_reconstructed = np.zeros((len(Y_test), num_original_features))
        y_test_reconstructed[:,dependent_index_value] = Y_test.ravel()
#         y_test_reconstructed[:,0] = Y_test.ravel()
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
        path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/models/evaluation_metrics_{self.dependent_var}.json'
        self.dict_manager.update_value(f"Evaluation metrics of LSTM model ", path)

        # Write the dictionary to a JSON file
        with open(path, "w") as json_file:
            json.dump(metrics, json_file)

        print(f"Metrics saved to {path}.")
       
    def plot_lstm_inference(self):
        # Plot actual vs predicted values
        title='LSTM Model: Actual vs Predicted Test Samples Values'
        plt.figure(figsize=(15, 6))
        plt.plot(self.y_test_inv, label='Actual')
        plt.plot(self.y_pred_inv, label='Predicted')
        plt.title(title)
        plt.ylabel('Value')
        plt.xlabel('Sample Index')
        plt.legend()
        path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/graphs/LSTM_inference_plot_{self.dependent_var}.png'
        self.dict_manager.update_value(f"LSTM Inference plot ", path)
        plt.savefig(path)
        plt.show()
#         plt.savefig('/content/drive/MyDrive/MAL2324_CW_DataSet_Initial/Results/'+title+'graph.png')



## Classification


    # Main function of Random forest
    def BaggingRandomForestClassifier(self):
        # print("data\n", self.X_data, self.Y_data)
        # Split the data using train_test_splitter
        self.X_train, self.X_test, self.Y_train, self.Y_test = self.train_test_splitter(self.X_data, self.Y_data, test_size=self.test_size)

        # Now you can access self.X_train and self.Y_train
        max_samples_ratio = 0.9
        n_estimators = self.X_train.shape[1]
        self.train_rfc_model(self.X_train, self.X_test, self.Y_train, self.Y_test)

        # Use RandomForestClassifier as the base estimator
#         base_estimator = RandomForestClassifier(n_estimators=n_estimators)

#         # Create the BaggingClassifier without specifying base_estimator
#         self.baggingrfc_model = BaggingClassifier(base_estimator, n_estimators=n_estimators, max_samples=max_samples_ratio)
#         self.baggingrfc_model.fit(self.X_train, self.Y_train)

#         # You might want to calculate accuracy using self.X_test and self.Y_test
#         rfc_accuracy = self.baggingrfc_model.score(self.X_test, self.Y_test)
#         print("Random Forest Model Model Accuracy : ", rfc_accuracy)

        rfc_result = self.predict_bagggingrfc_model(self.X_data)
        print("Results of Random Forest Classifier on Test set : ", rfc_result)

        
        
    def train_rfc_model(self,X_train, X_test, y_train,  y_test):
        # Create and train the Random Forest model

        

        max_samples_ratio = 0.9
        n_estimators = self.X_train.shape[1]
        # Use RandomForestClassifier as the base estimator
        base_estimator = RandomForestClassifier(n_estimators=n_estimators)

        # Correct parameter name is 'base_estimator', not 'base_estimator'
        self.baggingrfc_model = BaggingClassifier(base_estimator, n_estimators=n_estimators, max_samples=max_samples_ratio)
        self.baggingrfc_model.fit(self.X_train,self.Y_train)

        #self.score(rfc_model)
        model_path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/models/RFC_model_{self.dependent_var}.joblib'
        scaler_x_path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/models/scaler_x_{self.dependent_var}.pkl'
        x_encoder_path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/models/x_encoder_{self.dependent_var}.pkl'
        y_encoder_path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/models/y_encoder_{self.dependent_var}.pkl'
        column_names_path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/models/column_names_{self.dependent_var}.txt'
        #column_score = f'Knowledge/{self.problem_type}/{self.source}/{self.file_name}/models/column_names_{self.dependent_var[0]}.png'
        
        print(self.scaler)
        print(self.scaler)
        
        
        joblib.dump(self.baggingrfc_model, model_path)
        joblib.dump(self.scaler, scaler_x_path)
        joblib.dump(self.label_encoder_y, y_encoder_path)
        joblib.dump(self.label_encoder, x_encoder_path)
            # Save column names if available
        if isinstance(self.X_data, pd.DataFrame):
            np.savetxt(column_names_path, self.X_data.columns, fmt='%s', delimiter=', ')
            
#         self.dict_manager.update_value(f"Random Forest Classifier {self.dependent_var[0]} Trained model path ", model_path)
#         self.dict_manager.update_value(f"Random Forest Classifier {self.dependent_var[0]} x scaler path ", scaler_x_path)
#         self.dict_manager.update_value(f"Random Forest Classifier {self.dependent_var[0]} label y encoder  path ", y_encoder_path)
#         self.dict_manager.update_value(f"Random Forest Classifier {self.dependent_var[0]} label x encoder path ", x_encoder_path)
#         self.dict_manager.update_value(f"Random Forest Classifier {self.dependent_var[0]} column names path ", column_names_path)
        
        self.dict_manager.update_value(f"Random Forest Classifier Trained model path ", model_path)
        self.dict_manager.update_value(f"Random Forest Classifier x scaler path ", scaler_x_path)
        self.dict_manager.update_value(f"Random Forest Classifier label y encoder  path ", y_encoder_path)
        self.dict_manager.update_value(f"Random Forest Classifier label x encoder path ", x_encoder_path)
        self.dict_manager.update_value(f"Random Forest Classifier column names path ", column_names_path)
        
        
        accuracy_path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/models/model_accuracy_{self.dependent_var}.txt'
        self.dict_manager.update_value(f"Random Forest Classifier Accuracy ", accuracy_path)
        
        
        print(f"Random Forest model saved as {model_path}")
        
        
        # Evaluate the model
        accuracy = self.baggingrfc_model.score(X_test, y_test)
#         np.savetxt(accuracy_path, accuracy, fmt='%s', delimiter=', ')
        with open(accuracy_path, 'w') as file: file.write(str(accuracy))
        
        predictions_decoded = self.predict_bagggingrfc_model(X_test)
        y_test_decoded = self.label_encoder_y.inverse_transform(y_test)
        

        # Calculate the confusion matrix
        
        conf_matrix = confusion_matrix(y_test_decoded, predictions_decoded, labels = self.label_encoder_y.classes_)
        conf_matrix_df = pd.DataFrame(conf_matrix, index=self.label_encoder_y.classes_, columns=self.label_encoder_y.classes_)
        cm_csv_path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/csv/confusion_matrix_csv_{self.dependent_var}.csv'
        self.dict_manager.update_value(f"Random Forest Classifier confusion matrix csv path ", cm_csv_path)
        conf_matrix_df.to_csv(cm_csv_path)
        
        # Plot the heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=self.label_encoder_y.classes_, yticklabels=self.label_encoder_y.classes_)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        cm_path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/graphs/confusion_matrix_{self.dependent_var}.png'
        self.dict_manager.update_value(f"Random Forest Classifier confusion matrix graph path ", cm_path)
        plt.savefig(cm_path)
        plt.show()
        
        
    

        # Open the file in write mode ('w')
        with open(accuracy_path, 'w') as file:
            # Write the float value to the file
            file.write(str(accuracy))
#         print(f"Model Accuracy: {accuracy}")

        print("clearning the RFC CPU/GPU resources!!!!!")
        clear_session()
        clear_session()
        clear_session()
        clear_session()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        return self.baggingrfc_model, accuracy

    def predict_bagggingrfc_model(self, X_test):
#         if input_data == None:
#             X_test = self.X_test
#         else:
#             X_test = input_data
        
        # Make predictions on the test set
        y_pred = self.baggingrfc_model.predict(X_test)

        # Decode the predicted results (inverse_transform for LabelEncoder)
        y_pred_decoded = self.label_encoder_y.inverse_transform(y_pred)

        return y_pred_decoded
    
            
        
        
# Classification related functions
        
    
    
    
    
