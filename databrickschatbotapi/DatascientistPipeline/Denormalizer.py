import sqlite3
import csv
import os

class Denormalizer:
    def __init__(self, db_file):
        self.db_file = db_file
        self.output_dir = 'denormalized_datasets'

    def denormalize_tables(self):
        
        directory_path = os.path.dirname(self.db_file)
        self.output_dir = os.path.join(directory_path, self.output_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
         
        # Connect to the SQLite database
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()

        # Get all tables
        c.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in c.fetchall()]
        print("Tables in db: ", tables)

        # List to keep track of joined tables
        joined_tables = []

        all_joined_set = set()

        for table in tables:
            print(f"\n\n\nProcessing for Table {table}")
            if table not in joined_tables:
                # Reset variables for each table
                all_rows = []
                columns_to_exclude = []

                # Attempt to find related tables through foreign keys
                c.execute(f"PRAGMA foreign_key_list({table})")
                fks = c.fetchall()

                print(f"Fks in table {table} are {fks}")

                # Build a join query if foreign keys exist
                if fks:
                    join_query = f"SELECT *"
                    for fk in fks:
                        ref_table = fk[2]
                        # join_query += f", {ref_table}.*"

                    join_query += f" FROM {table}"
                    for fk in fks:
                        ref_table = fk[2]
                        join_query += f" JOIN {ref_table} ON {table}.{fk[3]}={ref_table}.{fk[4]}"

                        # Check for second-degree relationships
                        c.execute(f"PRAGMA foreign_key_list({ref_table})")
                        second_degree_fks = c.fetchall()

                        print(f"Second degree Fks in table {table}_{ref_table} are {second_degree_fks}")
                        if second_degree_fks != []:

                            for second_fk in second_degree_fks:
                                second_ref_table = second_fk[2]
                                join_query += f" JOIN {second_ref_table} ON {ref_table}.{second_fk[3]}={second_ref_table}.{second_fk[4]}"

                                # Add joined tables to the tracking list
                                joined_tables.extend([ref_table, second_ref_table])
                                all_joined_set.add(second_ref_table)
                        else:
                            joined_tables.extend([table, ref_table])
                            all_joined_set.add(table)
                            all_joined_set.add(ref_table)

                    print(f"Joined tables {joined_tables} for table {table}")
                    print(f"Join Query: {join_query}")
                    c.execute(join_query)
                    # Add the current table to the joined tables list
    #                 joined_tables.append(table)
    #                 print("joined_tables while appending", joined_tables)
                else:
                    # If no foreign keys, select all from the table
                    c.execute(f"SELECT * FROM {table}")

                # Fetch rows and columns
                rows = c.fetchall()
                print("Records: ", rows)
                columns = [i[0] for i in c.description]
                print(f"columns: {columns}")

                # Determine columns to exclude dynamically (those with 'ID' in their names)
                # columns_to_exclude = [column for column in columns if 'ID' in column or 'Id' in column or '_id' in column]


                # Export to CSV
                if joined_tables != []:

                    csv_file_path = os.path.join(self.output_dir, f'{table}_denormalized.csv')
                    print(f'\nSaving table {table}_denormalized.csv')
                    with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
                        writer = csv.writer(file)

                        # Write column headers excluding the columns with 'ID' in their names
                        writer.writerow([column for column in columns])

                        # Write all rows excluding the columns with 'ID' in their names
                        for row in rows:
                            filtered_row = [value for index, value in enumerate(row)]
                            writer.writerow(filtered_row)
                        joined_tables = []

        for table in tables:
            if table not in all_joined_set:
                c.execute(f"SELECT * FROM {table}")
                rows = c.fetchall()
                columns = [i[0] for i in c.description]

                # Determine columns to exclude dynamically (those with 'ID' in their names)
    #           columns_to_exclude = [column for column in columns if 'ID' in column or 'Id' in column or '_id' in column]

                # Export to CSV
                csv_file_path = os.path.join(self.output_dir, f'{table}_denormalized.csv')
                print(f'\nSaving non_joined_table {table}_denormalized.csv')
                with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)

                    # Write column headers excluding the columns with 'ID' in their names
                    writer.writerow([column for column in columns])

                    # Write all rows excluding the columns with 'ID' in their names
                    for row in rows:
                        filtered_row = [value for index, value in enumerate(row)]
                        writer.writerow(filtered_row)

        # Close the connection
        conn.close()
        return self.output_dir
        

# Usage
# db_file = 'input_folder/DB_TEST.db'
# output_dir = 'denormalization_result'
# denormalize_and_export(db_file, output_dir)
