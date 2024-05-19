from databrickschatbotapi.DatascientistPipeline.Llama_hamza import LlamaInference
import pandas as pd

class AnalysisGenerator:
    def __init__(self, auth_token, df,question):
        self.auth_token = auth_token
        self.llama_inference = LlamaInference(self.auth_token)
        self.data = df
        self.question = question
#         self.dependent_variable= dependent_variable
    
#         self.file_name = file_name

#         self.problem_type = problem_type
        
#         self.json_file = json_file
        
    def introduction(self):
        intro_res = self.llama_inference.intoduction(self.data)
        print('introduction response generated Successfully based on provided Data Execution complete!')
#         print(intro_res)
        return intro_res
    
    
    def analyze_imp_features(self,value):
        try:
            # Read the CSV file into a DataFrame
#             df = pd.read_csv(csv_file_path)

            # Filter out columns with names like 'Unnamed'
    #         valid_column_names = [col for col in df.columns if 'Unnamed' not in col]
    #         print('Columns',valid_column_names)

            # Read the corrmat CSV file
            corrmat = pd.read_csv(value)

            # Additional logic or analysis with corrmat and valid_column_names can be added here
            analysis_result = self.llama_inference.imp_features(corrmat, self.dependent_variable)
            print('Imp features response generated Successfully based on provided Data Execution complete!')

            return analysis_result
        except FileNotFoundError as e:
            return f"File not found: {e}"
        except Exception as e:
            return f"An error occurred: {e}"



        
#     def analyze_csv_and_corrmat(self,value):
        
# #         try:
#             # Read the CSV file into a DataFrame
# #             df = pd.read_csv(csv_file_path)

#             # Filter out columns with names like 'Unnamed'
# #             valid_column_names = [col for col in df.columns if 'Unnamed' not in col]
# #             print('Dependend Variable',valid_column_names)

# #             # Read the corrmat CSV file
#         corrmat = pd.read_csv(value)
#         print(self.dependent_variable)
# #             target_corr = corrmat[self.dependent_variable]

#         # Additional logic or analysis with corrmat and valid_column_names can be added here
#         analysis_result = self.llama_inference.corrmat_explain(corrmat,self.dependent_variable)
#         print('co relation response generated Successfully based on provided Data Execution complete!')

        return analysis_result
#         except FileNotFoundError as e:
#             return f"File not found: {e}"
#         except Exception as e:
#             return f"An error occurred: {e}"
        
    def analyze_user_question(self, question,user_specified_domain='any'):
        try:
#             df = pd.read_csv(corrmat_path)
            if not user_specified_domain:
                user_specified_domain = "any"
            # Additional processing or analysis logic can be added here
            analysis_result = self.llama_inference.user_domain_question(self.data, question,user_specified_domain)
            print('Domain specific question response generated Successfully based on provided Data Execution complete!')
            return analysis_result
        except FileNotFoundError as e:
            return f"File not found: {e}"
        except Exception as e:
            return f"An error occurred: {e}"
        
        
    def description(self,value):
        print(value)
        
#         try:
        csv_file_path = pd.read_csv(value)
        # Additional processing or analysis logic can be added here
        analysis_result = self.llama_inference.description(csv_file_path)
        print('Description response generated Successfully based on provided Data Execution complete!')
        return analysis_result


#         except FileNotFoundError as e:
#             return f"File not found: {e}"
#         except Exception as e:
#             return f"An error occurred: {e}"
        
    def analyze_csv_and_mostcorrmat(self,csv_file_path):
        
        try:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(csv_file_path)

            # Filter out columns with names like 'Unnamed'
            valid_column_names = [col for col in df.columns if 'Unnamed' not in col]
#             print('Columns',valid_column_names)

            # Read the corrmat CSV file
            corrmat = pd.read_csv(csv_file_path)
    #         corrmat = corrmat.to_csv(index=False)

            # Additional logic or analysis with corrmat and valid_column_names can be added here
            analysis_result = self.llama_inference.most_corrmat_explain(corrmat, valid_column_names)
            print('Most corelation response generated Successfully based on provided Data Execution complete!')
            return analysis_result
        except FileNotFoundError as e:
            return f"File not found: {e}"
        except Exception as e:
            return f"An error occurred: {e}"
        
    def cross_tab(self,csv_file_path,cross_tab1=None,cross_tab2=None):
#         try:
        df = pd.read_csv(csv_file_path)

        result = self.llama_inference.cross_tab(df,cross_tab1,cross_tab2)
        print('Cross-Tab response generated Successfully based on provided Data Execution complete!')
        return result

#         except FileNotFoundError as e:

#             return f"File not found: {e}"
#         except Exception as e:

#             return f"An error occurred: {e}"
        
    def chi_square(self,csv_file_path,chi_square1=None,chi_square2=None):
        try:
            chi_table = pd.read_csv(csv_file_path)

            result = self.llama_inference.chi_square(chi_table,chi_square1,chi_square2)
            print('Chi-Square response generated Successfully based on provided Data Execution complete!')
            return result

        except FileNotFoundError as e:

            return f"File not found: {e}"
        except Exception as e:

            return f"An error occurred: {e}"
        
    def confusion_matrix(self,csv_file_path):
        df = pd.read_csv(csv_file_path)
#         print(df)
        
        result = self.llama_inference.confusion_matrix(df)
        print('Confusion Matrix response generated Successfully based on provided Data Execution complete!')
        return result
        
        

