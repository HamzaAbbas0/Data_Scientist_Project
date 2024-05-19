import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

class FeatureSelector:
    def __init__(self, X, y):
        """
        Initialize the FeatureSelector with input features (X) and target variable (y).
        """
        self.X = X
        self.y = y

    def correlation_analysis(self, threshold=0.5):
        """
        Perform correlation analysis and select features based on correlation coefficient.

        Parameters:
        - threshold (float): Threshold for feature selection based on correlation coefficient.

        Returns:
        - selected_features (list): List of selected features.
        """
        # Calculate the correlation matrix
        corr_matrix = self.X.corr()

        # Find features with correlation above the threshold
        correlated_features = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    colname = corr_matrix.columns[i]
                    correlated_features.add(colname)

        # Remove correlated features
        selected_features = [col for col in self.X.columns if col not in correlated_features]

        return selected_features

    def random_forest_selection(self, n_estimators=100, importance_threshold=0.01):
        """
        Perform feature selection using Random Forest.

        Parameters:
        - n_estimators (int): Number of trees in the random forest.
        - importance_threshold (float): Threshold for feature selection based on feature importance.

        Returns:
        - selected_features (list): List of selected features.
        """
        # Create a random forest classifier
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

        # Fit the model
        rf.fit(self.X, self.y)

        # Get feature importances
        feature_importances = rf.feature_importances_

        # Select features based on importance threshold
        sfm = SelectFromModel(rf, threshold=importance_threshold)
        sfm.fit(self.X, self.y)

        # Get selected features
        selected_features = self.X.columns[sfm.get_support()]
        

        return selected_features

# Example usage:
# Assuming 'X' is your feature matrix and 'y' is your target variable
# selector = FeatureSelector(X, y)

# Correlation analysis
# selected_features_corr = selector.correlation_analysis(threshold=0.5)
# print("Selected Features (Correlation Analysis):", selected_features_corr)

# Random Forest feature selection
# selected_features_rf = selector.random_forest_selection(n_estimators=100, importance_threshold=0.01)
# print("Selected Features (Random Forest):", selected_features_rf)
