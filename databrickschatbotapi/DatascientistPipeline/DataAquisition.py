import pandas as pd
import os
import pandas as pd
from pandas.api.types import is_numeric_dtype
# from JSON_creator import MyDictionaryManager

class DataAquisition:

    def __init__(self, source, folder_path, file_name,process_id):
#         self.filepath = filepath
        self.source = source
        self.problem_type = None
        self.file_name = file_name
        self.filepath = os.path.join(folder_path, file_name)
        self.data = None
        self.process_id=process_id
#         self.dict_manager = MyDictionaryManager()
#         self.dict_manager.update_value('Folder path', folder_path)
#         self.dict_manager.update_value('File name', file_name)

    def delete_similar_columns(self, df):
        # Create a list to store the columns to be deleted
        columns_to_delete = []

        # Iterate through each pair of columns
        for i in range(len(df.columns)):
            for j in range(i+1, len(df.columns)):
                col1 = df.iloc[:, i]
                col2 = df.iloc[:, j]

                # Check if values in both columns are the same for all rows
                if col1.equals(col2):
                    # Add the column to the list to be deleted
                    columns_to_delete.append(df.columns[j])

        # Delete the columns in the list
        df = df.drop(columns=columns_to_delete)
        print(f"Deleted columns: {columns_to_delete}")

        return df
    
    
    def remove_unamed(self,df):
        for col in df.columns:
            if 'Unnamed: 0' in col or 'Unnamed' in col:
                df.drop([col], axis=1, inplace=True)
                
        return df

    def get_data(self):
        return self.data

    def read_data(self):
        if self.filepath.endswith('.csv'):
            df = self.read_csv()
        elif self.filepath.endswith('.xlsx') or self.filepath.endswith('.xls'):
            df = self.read_excel()
        else:
            raise ValueError("Unsupported file format. Only CSV and Excel files are supported.")

#         Convert all object-type columns to datetime if they contain datetime information
#         for column in df.select_dtypes(include='object').columns:
#             df[column] = pd.to_datetime(df[column], errors='coerce')

        df = self.delete_similar_columns(df)
        df = self.remove_unamed(df)
        self.data = df
        return df


    def read_csv(self):
        df = pd.read_csv(self.filepath)
        # Additional CSV-specific processing if needed
        return df

    def read_excel(self):
        df = pd.read_excel(self.filepath)
        # Additional Excel-specific processing if needed
        return df
    
    
    



    def analyze_problem_type(self, data, target_column):
        """
        Analyzes the problem type based on the characteristics of the data and the target column.

        Parameters:
            data (DataFrame): Input DataFrame containing the data.
            target_column (str): Name of the target column.

        Returns:
            str: Problem type inferred from the data ('numerical', 'time_series', 'categorical', 'unknown').
        """
        date_formats = ["%d-%m-%Y %H:%M:%S", "%d-%m-%Y", "%Y-%m-%d", "%Y-%m", "%m-%d-%Y %H:%M:%S", "%Y", "%m/%d/%Y %H:%M","%d/%m/%Y %H:%M","%m/%d/%Y","%d/%m/%Y"]




        for column in data.columns:
            if data[column].apply(lambda x: isinstance(x, str)).all():
                matched_format = None
                for date_format in date_formats:
                    try:
                        pd.to_datetime(data[column], format=date_format)
                        matched_format = date_format
                        break
                    except ValueError:
                        pass
                if matched_format:
                    try:
                        data[column] = pd.to_datetime(data[column], format=matched_format, errors='coerce')
                    except ValueError:
                        pass
                else:
                    # If no matching format found, keep the column as object
                    data[column] = data[column].astype('object')
                    
        print(data.info())

        try:
            if pd.api.types.is_numeric_dtype(data[target_column]):  
                has_datetime = any(pd.api.types.is_datetime64_any_dtype(data[col]) for col in data.columns)
                # If the target is numeric and of datetime type, it's likely a time series problem
                if has_datetime:     
                    date_index = input("Please Enter Right Date Column Name : ")
                    print("time series and numerical!!!!")
                    return 'time series','numerical', list(data.columns), data[date_index]
                else:
                    print("only numerical!!!!!!!")
                    return 'numerical', None, list(data.columns), None

            # Check if the target column is categorical (including object dtype)
            elif pd.api.types.is_categorical_dtype(data[target_column]) or pd.api.types.is_object_dtype(data[target_column]):
                print("categorical !!!!!!")
                return 'categorical', None, list(data.columns), None

            else:
                print("Its unknown, can't apply time series with categorical")
#                 return 'unknown', None, list(data.columns), None

        except Exception as e:
            print(e)






            
            
    def make_directories(self, problem_type, file_name):
        self.file_name = file_name
        self.problem_type = problem_type
        os.makedirs("Knowledge") if not os.path.exists("Knowledge") else None
        os.makedirs(f"Knowledge/{problem_type}") if not os.path.exists(f"Knowledge/{problem_type}") else None
        
        os.makedirs(f"Knowledge/{problem_type}/{self.source}") if not os.path.exists(f"Knowledge/{problem_type}/{self.source}") else None
       
        os.makedirs(f"Knowledge/{problem_type}/{self.source}/{self.process_id}") if not os.path.exists(f"Knowledge/{problem_type}/{self.source}/{self.process_id}") else None
        
        os.makedirs(f"Knowledge/{problem_type}/{self.source}/{self.process_id}/graphs") if not os.path.exists(f"Knowledge/{problem_type}/{self.source}/{self.process_id}/graphs") else None
        
        os.makedirs(f"Knowledge/{problem_type}/{self.source}/{self.process_id}/csv") if not os.path.exists(f"Knowledge/{problem_type}/{self.source}/{self.process_id}/csv") else None
        
        os.makedirs(f"Knowledge/{problem_type}/{self.source}/{self.process_id}/models") if not os.path.exists(f"Knowledge/{problem_type}/{self.source}/{self.process_id}/models") else None
        
        os.makedirs(f"Knowledge/{problem_type}/{self.source}/{self.process_id}/json") if not os.path.exists(f"Knowledge/{problem_type}/{self.source}/{self.process_id}/json") else None
        
        os.makedirs(f"Knowledge/{problem_type}/{self.source}/{self.process_id}/report") if not os.path.exists(f"Knowledge/{problem_type}/{self.source}/{self.process_id}/report") else None
        
        os.makedirs(f"Knowledge/{problem_type}/{self.source}/{self.process_id}/dataset") if not os.path.exists(f"Knowledge/{problem_type}/{self.source}/{self.process_id}/dataset") else None
        
    def move_dataset_to_knowledge(self, df):
        df.to_csv(f"Knowledge/{self.problem_type}/{self.source}/{self.process_id}/dataset/{self.file_name}")

        
    def connect_to_database(self, server, database, username, password):
        try:
            connection_string = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};"
                f"UID={username};PWD={password}"
            )
            self.database_connection = pyodbc.connect(connection_string)
            self.database_cursor = self.database_connection.cursor()
            print(f"Connected to {database} DB")
        except Exception as e:
            print(f"Error connecting to {database} DB: {e}")

    def close_database_connection(self):
        if hasattr(self, 'database_connection') and self.database_connection:
            self.database_connection.close()
            print("Closed database connection")
   
    