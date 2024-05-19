# !pip install colorama
# !pip install pip install scikit-plot
# ! pip install missingno

# import relevant modules
#%matplotlib inline
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import os
from sklearn.ensemble import RandomForestRegressor
import missingno as msno
from sklearn import preprocessing
from scipy.stats import kruskal
from databrickschatbotapi.DatascientistPipeline.JSON_creator import MyDictionaryManager
from IPython.display import display
from sklearn.inspection import permutation_importance
from scipy.stats import kruskal


# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

class CorrelationAnalysis:
    def __init__(self, source, df, file_name,process_id, problem_type, type_column, dependent_var, corr_thres):
        self.source = source
        self.data = df
#         self.selected_data = None
        self.file_name = file_name
        self.problem_type = problem_type
        self.type_column = type_column
        self.dependent_var = dependent_var
        self.df_describe = self.data.describe()
        self.most_corr_features = None
        self.corr_thres = corr_thres
        self.most_cor_data=None
        self.process_id=process_id
#         self.scaler = MinMaxScaler()

        #JSON Innitialization
        self.dict_manager = MyDictionaryManager(source, file_name, process_id, problem_type)
        
        if problem_type.lower() == "time series":
            self.time_series()
            
        if problem_type.lower() == "categorical":
            self.categorical()

        elif problem_type.lower() == "numerical":
            self.numerical()


    
    def time_series(self):
            try:
        #         self.dis_plot()
                self.corr_heatmap(self.data, self.dependent_var)
        #         self.pairplot()
                #corr_thres = self.corr_thres
                #self.most_corr_features = self.corr_feature_selector(self.data, self.dependent_var, corr_thres = corr_thres)
                self.my_corr_feature_selector(self.data, self.dependent_var, self.problem_type, self.corr_thres)
                self.dict_manager.save_dictionary()
            except Exception as e:
                print("An error occurred during time series analysis:", e)

    def categorical(self):
        try:
            self.corr_heatmap(self.data, self.dependent_var)
            numerical_columns = self.data.select_dtypes(include=np.number).columns
            categorical_columns = self.data.select_dtypes(include='object').columns

            KW_features = self.Kruskal_Wallis_test(self.data)  # Pass the entire DataFrame
            rfc_features = self.random_forest_selection(self.data)
            
    #         most_corr_features = KW_features.extend(rfc_features)
            self.most_corr_features = list(set(KW_features) | set(rfc_features))
            self.most_corr_features.append(self.dependent_var)
            print("Selected Features : ",self.most_corr_features)
        except Exception as e:
            print("An error occurred during categorical analysis:", e)

    def numerical(self):
        try:
            self.corr_heatmap(self.data, self.dependent_var)
            self.random_forest_selection_numerical()
    #         self.dict_manager.save_dictionary()
        except Exception as e:
            print("An error occurred during numerical analysis:", e)
        
    def random_forest_selection_numerical(self):
        try:
            # Separate features and target variable
            n_estimators = 100
            importance_threshold = 0.05

            X = self.data.drop(self.dependent_var, axis=1)
            y = self.data[self.dependent_var]

            # Encode categorical columns
            label_encoder = LabelEncoder()
            for column in X.select_dtypes(include='object').columns:
                X[column] = label_encoder.fit_transform(X[column])

            # Initialize the Random Forest Regressor
            rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)

            # Fit the model
            rf.fit(X, y)

            # Calculate permutation importance
            result = permutation_importance(rf, X, y, n_repeats=10, random_state=42)

            # Get feature importances
            feature_importances = result.importances_mean

            # Select features based on importance threshold
            selected_features = X.columns[feature_importances > importance_threshold]

            most_corr = pd.DataFrame({
                'Feature': selected_features,
                'Importance': feature_importances[feature_importances > importance_threshold]
            })

            # Save to CSV
            path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/csv/most_correlated_features_with_{self.dependent_var}.csv'
            self.dict_manager.update_value(f"Most correlated Features with dependent Variable ", path)
            most_corr.to_csv(path, index=False)
            plt.figure(figsize=(22,10))
            most_corr.plot.bar()
            plt.tight_layout()

            df_important = self.data[selected_features]

            df_important[self.dependent_var] = y

            self.most_corr_features = df_important.columns

            display(most_corr)
                                   

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            
    def get_most_corr_data(self):
        return self.most_cor_data
    
    def get_data(self):
        return self.data
    
    def msno_plot(self):
        msno.bar(self.data) # you can see pandas-profilin count part
        plt.title('Count of Values per Column in Dataset for Missing value Analysis', size=16)
        
    def pairplot(self):
        try:
            print("____________Pair Plot______________ \n")
            sns.pairplot(self.data)
            path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/graphs/pairplot.png'
            self.dict_manager.update_value(f"Pairwise plots ", path)
            plt.savefig(path)
            plt.show()

        except Exception as e:
            print(f"An error occurred in pair plot: {str(e)}")

    def count_plot(self, dependent_var):
        try:
            plt.figure(figsize=(10, 7))
            plt.title(f'Count of {dependent_var} Samples', size=16)
            sns.countplot(x=dependent_var, data=self.data)
            plt.ylabel('Count', size=14)
            plt.xlabel(dependent_var, size=14)
            sns.despine(top=True, right=True, left=False, bottom=False)
            path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/graphs/count_plot.png'
            self.dict_manager.update_value(f"count plot ", path)
            plt.savefig(path)
            plt.show()

        except Exception as e:
            print(f"An error occurred in count plot: {str(e)}")

    
    def dis_plot(self):
        columnnames=list(self.df_describe.columns) # This will contain the df without date column, that will be used for data transformation
        for columnname in columnnames:
            plt.figure(figsize=(22,22))
            sns.displot(self.data[columnname][1:], kind="kde")
            plt.title('Plot of '+columnname)
            plt.tight_layout()
            plt.show()
            
     
    def corr_heatmap(self, df, dependent_var):
        try:
            idx_s = 0
            idx_e = len(df) - 1
            plt.figure(figsize=(22, 10))
            y = df[dependent_var]
            temp = df.iloc[:, idx_s:idx_e]
            if 'id' in temp.columns:
                del temp['id']
            temp[dependent_var] = df[dependent_var]
            sns.heatmap(temp.corr(), annot=True, fmt='.2f')
            plt.title('Heat Map of Correlation Matrix', fontsize=22)
            path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/graphs/heatmap.png'
            self.dict_manager.update_value(f"Correlation Heatmap ", path)
            plt.savefig(path)
            plt.show()

        except Exception as e:
            print(f"An error occurred in correlation heatmap: {str(e)}")

        
    def my_corr_feature_selector(self, df, dependent_var, problem_type, corr_thres):
        try:
            # Assuming df is your DataFrame
            encoder = LabelEncoder()

            # Identify object type (categorical) columns
            categorical_cols = df.columns[df.dtypes == 'object']

            # Apply LabelEncoder to these columns only
            for col in categorical_cols:
                df[col] = encoder.fit_transform(df[col])
            corrmat = df.corr()
            self.corr_heatmap(df, dependent_var)
            # Correlation with output variable
            cor_target = abs(corrmat[dependent_var])

            # Selecting highly correlated features
            relevant_features = cor_target[cor_target > 0.01]
            most_corr1 = pd.DataFrame(columns=['Most Correlated Features', 'Score'])
            most_corr1['Most Correlated Features'] = relevant_features.index
            most_corr1['Score'] = relevant_features.values

            plt.figure(figsize=(22, 10))
            cor_target = cor_target.dropna()
            path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/csv/all_correlated_features_with_{self.dependent_var}.csv'
            self.dict_manager.update_value(f"All correlated Features with dependent Variable ", path)
            cor_target.to_csv(path)
            cor_target.plot.bar();
            plt.title('Top Most Correlated Features from Correlation Matrix', fontsize=22)
            plt.tight_layout()
            path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/graphs/most_correlated_features_with_{self.dependent_var}.png'
            self.dict_manager.update_value(f"Graph of most correlated Features with dependent Variable ", path)
            plt.savefig(path)
            plt.show()
            self.most_corr_features = most_corr1['Most Correlated Features']
            self.most_cor_data = self.data[self.most_corr_features].copy()
            return most_corr1['Most Correlated Features']

        except Exception as e:
            print(f"An error occurred: {str(e)}")

       
    
    def Kruskal_Wallis_test(self, df):
        try:
            # Load your dataset

            # Separate features (X) and target variable (y)
            X = df.copy()
            y = self.data[self.dependent_var]

            # Label encode categorical columns
            categorical_cols = X.select_dtypes(include='object').columns
            label_encoder = LabelEncoder()
            for col in categorical_cols:
                X[col] = label_encoder.fit_transform(X[col])

            # Perform Kruskal-Wallis test for each numerical variable
            kruskal_results = []
            for col in X.select_dtypes(include='number').columns:
                # Check for variability within each group
                if X[col].nunique() > 1:
                    groups = [X[col][y == c] for c in y.unique()]
                    _, p_value = kruskal(*groups)
                    kruskal_results.append((col, p_value))
                else:
                    print(f"Skipping Kruskal-Wallis test for {col} as there is no variability.")

            # Display Kruskal-Wallis test results
            kruskal_df = pd.DataFrame(kruskal_results, columns=['Variable', 'Kruskal-Wallis P-Value'])
            path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/csv/Kruskal-Wallis_result.csv'
            self.dict_manager.update_value(f"Result of Kruskal-Wallis test for feature selection", path)
            kruskal_df.to_csv(path)
            print(kruskal_df)
            print("Kruskal_Wallis features : ", kruskal_df.Variable)
            return list(kruskal_df.Variable)

        except Exception as e:
            print(f"An error occurred in Kruskal-Wallis test: {str(e)}")


    def random_forest_selection(self, df, n_estimators=100, importance_threshold=0.02):
        try:
            sccombined_df = self.data_scaller(df)
            combined_dfcat = self.categorical_encoder(df)

            combined_df_x = pd.concat([sccombined_df, combined_dfcat], axis=1)

            X = combined_df_x
            y = combined_df_x[self.dependent_var]

            if self.dependent_var in X.columns:
                X.drop(self.dependent_var, axis=1, inplace=True)
                print(f"Column '{self.dependent_var}' dropped.")
            else:
                print(f"Column '{self.dependent_var}' not found in the DataFrame.")

            # Create a random forest classifier
            rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

            # Fit the model
            rf.fit(X, y)

            # Get feature importances
            feature_importances = rf.feature_importances_

            # Select features based on importance threshold
            sfm = SelectFromModel(rf, threshold=importance_threshold)
            sfm.fit(X, y)

            # Get selected features
            selected_features = X.columns[sfm.get_support()]

            print("Random Forest classifier features : ", selected_features)

            # Save to CSV
            most_corr = pd.DataFrame({
                'Feature': selected_features
            })
            path = f'Knowledge/{self.problem_type}/{self.source}/{self.process_id}/csv/most_correlated_features_with_{self.dependent_var}.csv'
            self.dict_manager.update_value(f"Most correlated Features with dependent Variable ", path)
            most_corr.to_csv(path)

            return list(selected_features)

        except Exception as e:
            print(f"An error occurred in random forest selection: {str(e)}")
    
    
    def data_scaller(self, df):
        scaler = StandardScaler()

        # extract numerical attributes and scale it to have zero mean and unit variance
        cols = df.select_dtypes(include=['float64','int64']).columns
        sccombined_df = scaler.fit_transform(df.select_dtypes(include=['float64','int64']))

        # turn the result back to a dataframe
        sccombined_df = pd.DataFrame(sccombined_df, columns = cols)

        return sccombined_df
    
    def categorical_encoder(self, df):
        
        encoder = LabelEncoder()

        # extract categorical attributes from both combined_dfing and test sets
        catcombined_df = df.select_dtypes(include=['object']).copy()


        # encode the categorical attributes
        combined_dfcat = catcombined_df.apply(encoder.fit_transform)
        return combined_dfcat
    
    
    
#     def corr_feature_selector(self, data, dependent_var, corr_thres):
        
#         target = data
#         if target[dependent_var[0]].dtype == object:
#             le = preprocessing.LabelEncoder()
#             target[dependent_var]=le.fit_transform(target[dependent_var])
#         corrmat = target.corr()
#     #         print("_____ Correlation Matrix ______ \n", corrmat)
#     #         corrmat.to_csv(f'LSTM_result_csv/correlation_matrix.csv')
#         path = f'Knowledge/{self.problem_type}/{self.source}/{self.file_name}/csv/correlation_matrix.csv'
#         self.dict_manager.update_value(f"Correlation Matrix ", path)
#         corrmat.to_csv(path)

#     #         print("Correlation Matrix",corrmat)

#         # Correlation with output variable
#     #         cor_target_negpos = corrmat[dependent_var]
#     #         cor_target_negpos.to_csv(f'LSTM_result_csv/all_correlated_features.csv')
#     #         cor_target = abs(corrmat[dependent_var])
#     #         cor_target.to_csv(f'LSTM_result_csv/all_correlated_features.csv')

#         # Selecting highly correlated features
#     #         cor_target = abs(corrmat[dependent_var])
#         cor_target = corrmat[dependent_var]
#     #         cor_target.to_csv(f'LSTM_result_csv/all_correlated_features.csv')
#         path = f'Knowledge/{self.problem_type}/{self.source}/{self.file_name}/csv/all_correlated_features_with_{self.dependent_var[0]}.csv'
#         self.dict_manager.update_value(f"All correlated Features with dependent Variable ", path)
#         cor_target.to_csv(path)
#         # print(cor_target)
#         #cor_target.to_csv('/content/drive/MyDrive/MAL2324_CW_DataSet_Initial/Results/cor_targetfeatures.csv')
#         # Selecting highly correlated features
#         relevant_features_pos = cor_target[cor_target > corr_thres]
#         relevant_features_pos = relevant_features_pos.dropna()
#     #         print(relevant_features_pos)
#         relevant_features_neg = cor_target[cor_target < -corr_thres]
#         relevant_features_neg = relevant_features_neg.dropna()
#     #         print(relevant_features_neg)
#         relevant_features = relevant_features_pos.append(relevant_features_neg, ignore_index=False)
#     #         print(relevant_features)
#         # print("Most Correlated:",relevant_features)
#         most_corr1 = pd.DataFrame(columns=['Most Correlated Features','Score'])
#         most_corr1['Most Correlated Features']=relevant_features.index
#         most_corr1['Score']=relevant_features.values
#     #         most_corr_negpos = cor_target_negpos[cor_target_negpos.index.isin(relevant_features.index)]
#     #         most_corr_negpos.to_csv(f'LSTM_result_csv/most_correlated_features.csv')
#     #         most_corr1.to_csv(f'LSTM_result_csv/most_correlated_features.csv')
#         path = f'Knowledge/{self.problem_type}/{self.source}/{self.file_name}/csv/most_correlated_features_with_{self.dependent_var[0]}.csv'    
#         self.dict_manager.update_value(f"Most correlated Features with dependent Variable ", path)
#         most_corr1.to_csv(path)

#         plt.figure(figsize=(22,10))

#         most_corr1=most_corr1.sort_values(by='Score', ascending=False)
#         most_corr1=most_corr1.dropna()
#         print(f"Most correlated Features of '{dependent_var}' with Threshould limit : {corr_thres}: \n  \n",most_corr1)
#         most_corr1.plot.bar(x='Most Correlated Features', y='Score', rot=90, legend=False)
#         #cor_target.to_csv('/content/drive/MyDrive/MAL2324_CW_DataSet_Initial/Results/cor_targetfeatures.csv')

#         plt.title(f'Top Most Correlated Features with {dependent_var[0]}', fontsize = 22)
#         plt.tight_layout()
#         #plt.savefig('/content/drive/MyDrive/MAL2324_CW_DataSet_Initial/Results/Correlation Matrix Results/correlationgraph.png')
#     #         plt.savefig(f'LSTM_graphs/most_correlated_features.png')
#         path =f'Knowledge/{self.problem_type}/{self.source}/{self.file_name}/graphs/most_correlated_features_with_{self.dependent_var[0]}.png'    
#         self.dict_manager.update_value(f"Graph of most correlated Features with dependent Variable ", path)
#         plt.savefig(path)
#         plt.show()
#         return most_corr1['Most Correlated Features']
        
        