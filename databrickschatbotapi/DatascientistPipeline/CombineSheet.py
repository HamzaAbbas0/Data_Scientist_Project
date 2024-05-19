import os
import pandas as pd
import sqlite3
import csv

class CombineSheet:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.output_file = 'Combined.csv'
        self.output_folder = 'combined_datasets'

        
    def find_base_table(self, tables):
        table_scores = {}

        for table1_name, table1_data in tables.items():
            table_scores[table1_name] = 0
            for table2_name, table2_data in tables.items():
                if table1_name != table2_name:
                    common_columns = set(table1_data.columns) & set(table2_data.columns)
                    table_scores[table1_name] += len(common_columns)

        base_table = max(table_scores, key=table_scores.get)
        return base_table

    def combine_tables(self):
        # Get a list of CSV and Excel files in the folder
        files = [f for f in os.listdir(self.folder_path) if f.endswith('.csv') or f.endswith('.xlsx')]
        
        if files == None or files == []:
            return

        if not files:
            print("No CSV or Excel files found in the specified folder.")
            return
        
        self.output_folder = os.path.join(self.folder_path, self.output_folder)
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # Read all files into a dictionary of DataFrames
        tables = {}
        for file in files:
            table_name, extension = os.path.splitext(file)
            if extension == '.csv':
                tables[table_name] = pd.read_csv(os.path.join(self.folder_path, file), encoding='latin1')
            elif extension == '.xlsx':
                tables[table_name] = pd.read_excel(os.path.join(self.folder_path, file))

        base_table = self.find_base_table(tables)
        print("base_table:", base_table)

        # Initialize the denormalized dataset with the base table
        denormalized_dataset = tables[base_table]

        table_names_list = [base_table]
        
        # Merge other tables based on common ID columns
        for table_name, table_data in tables.items():
            if table_name != base_table:
                # Find common columns (ID columns) between the two tables
                common_columns = list(set(denormalized_dataset.columns) & set(table_data.columns))

                if common_columns:
                    # Merge the tables based on common columns
                    denormalized_dataset = pd.merge(denormalized_dataset, table_data, on=common_columns, how='left')
                    
                    table_names_list.append(table_name)
                    print("table_names_list: ", table_names_list)

                else:
                    # Save the table without common columns to the denormalized_datasets folder
                    table_data.to_csv(os.path.join(self.output_folder, f"{table_name}.csv"), index=False)

        # Save the denormalized dataset to the denormalized_datasets folder       
        self.output_file= f"{self.output_file.split('.')[0]}_{'_'.join(table_names_list)}.csv"
        denormalized_dataset.to_csv(os.path.join(self.output_folder, self.output_file), index=False)


        print(f"Combined dataset saved to {self.output_folder}/{self.output_file}")
        print(f"Tables without common columns saved in {self.output_folder}")
        return self.output_folder

    