import numpy as np

class LinearRegression:
    def __init__(self, data, dependent_variable, test_size=0.2, random_seed=None):
        """
        Initialize the Linear Regression model with data, splitting it into input features (X) and target variable (y).

        Parameters:
        - data: Input features and target variable as a tuple (X, y)
        - test_size: Proportion of the data to include in the test split (default is 0.2)
        - random_seed: Seed for random number generation to ensure reproducibility (default is None)
        """
        X, y = self.extract_features_target(data, dependent_variable)
        self.X_train, self.X_test, self.y_train, self.y_test = self.train_test_split(X, y, test_size, random_seed)
        self.coefficients = self.train()

    def extract_features_target(self, data, dependent_variable):
        # Find the index of the dependent variable
        X = data.drop(columns=[dependent_variable])
        y = data[dependent_variable]

        return X, y



    def train(self):
        """
        Train the linear regression model using the normal equation method.

        Returns:
        - coefficients: Coefficients of the linear regression model
        """
        X_train = np.concatenate((np.ones((self.X_train.shape[0], 1)), self.X_train), axis=1)
        y_train = self.y_train.values.reshape(-1, 1)
        print(X_train)
        print(y_train)

        X_transpose = np.transpose(X_train)
        coefficients = np.linalg.inv(X_transpose @ X_train) @ X_transpose @ y_train
        return coefficients

    def transform_data(self, X):
        """
        Transform input features according to the linear regression model.

        Parameters:
        - X: Input features to be transformed (numpy array or matrix)

        Returns:
        - Transformed features using the model
        """
        return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    def apply_model(self, X):
        """
        Apply the linear regression model to input features.

        Parameters:
        - X: Input features to apply the model on (numpy array or matrix)

        Returns:
        - Predicted values using the model
        """
        X_transformed = self.transform_data(X)
        return X_transformed @ self.coefficients

    def evaluate_model(self):
        """
        Evaluate the linear regression model using the mean squared error.

        Returns:
        - Mean Squared Error (MSE) of the model on the training data
        """
        X_train_transformed = self.transform_data(self.X_train)
        predictions = X_train_transformed @ self.coefficients
        mse = np.mean((predictions - self.y_train) ** 2)
        return mse

    def make_predictions(self, X):
        """
        Make predictions using the trained linear regression model.

        Parameters:
        - X: Input features for making predictions (numpy array or matrix)

        Returns:
        - Predicted values using the model
        """
        X_transformed = self.transform_data(X)
        return X_transformed @ self.coefficients

    def train_test_split(self, X, y, test_size=0.2, random_seed=0):
        """
        Split the data into training and testing sets.

        Parameters:
        - X: Input features (numpy array or matrix)
        - y: Target variable (numpy array)
        - test_size: Proportion of the data to include in the test split (default is 0.2)
        - random_seed: Seed for random number generation to ensure reproducibility (default is None)

        Returns:
        - X_train, X_test, y_train, y_test: Split data
        """
        print("X len", len(X))
        print("y len", len(y))
        print("X sahpe", X.shape)
        print("y shape", y.shape)
        indices = range(len(X))
        print(indices)
        if random_seed:
            np.random.seed(random_seed)

        num_samples = X.shape[0]
        print("num_samples ", num_samples)
        indices = np.random.permutation(num_samples)
        print("indices ", indices)
        test_size = int(test_size * num_samples)
        print("test_size ", test_size)

        test_indices = indices[:test_size]
        print("test_indices ", test_indices)
        train_indices = indices[test_size:]
        print("train_indices", train_indices)

        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
        print("X_train ", X_train)
        print("y_train ", y_train)
        print("x_test ", X_test)
        print("y_test ", y_test)

        return X_train, X_test, y_train, y_test
