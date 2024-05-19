from huggingface_hub import login
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import textwrap
# from Llama import LlamaInference
import pandas as pd
import os
from databrickschatbotapi.DatascientistPipeline.Graph_description import GraphDescriptionPipeline
from ipywidgets import interact, widgets
import os
from databrickschatbotapi.DatascientistPipeline.deepseeklm import DeepSeekLM
import json
import pandas as pd
from databrickschatbotapi.DatascientistPipeline.Llama import LlamaInference
from databrickschatbotapi.DatascientistPipeline.DataAquisition import DataAquisition
# from realtime_analysis import RealtimeAnalysis
from IPython.display import clear_output
from IPython.display import JSON
from databrickschatbotapi.DatascientistPipeline.Graph_description import GraphDescriptionPipeline
import mercury as mr
from databrickschatbotapi.DatascientistPipeline.CodeGeneration import CodeGeneration
from databrickschatbotapi.DatascientistPipeline.Inference_Classification import Inference_Classification
from databrickschatbotapi.DatascientistPipeline.Inference_Regression import Inference_Regression
from databrickschatbotapi.DatascientistPipeline.Inference_TimeSeries import LSTMForecaster
from datetime import datetime
from databrickschatbotapi.DatascientistPipeline.chatbot_final_file import Chatbot

# langchain
import os
import shutil
import torch
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, pipeline
from datetime import datetime
from dateutil import parser
import re


class RealtimeAnalysis:
    def __init__(self, auth_token, df, file_name,process_id, problem_type, source, json_file,target_variable,datetime_column):
        self.auth_token = auth_token
#         self.chatbot = LlamaInference(self.auth_token)
        self.chatbot = Chatbot()
        # Graph Description
#         llm_manager = LLMManager()
#        self.img_to_text = img_to_text
        self.img_to_text = GraphDescriptionPipeline()
        self.file_name = file_name
        self.data = df
        self.problem_type = problem_type
        self.json_file = json_file
        self.fn_list = []
        self.dependent_var = None
        self.source = source
        self.process_id=process_id
        self.target_variable = target_variable
        self.datetime_column = datetime_column
        self.predictioncolumns= None
        self.predictiondate=None
        if self.problem_type == "time series":
            self.fn_list = ['general_chat','Summary statistics explainer','Domain Explainer', 'Probability distribution visualization', 'Missing Number plot visualization', 'Heatmap of dataset', 'correlation matrix explainer', 'Training loss visualization','top_most_important_correlated_features_csv','inference_forcasting_timeseries','NO function matches found form list','what will be '+self.target_variable+'on','forcast','give forcast']
        
        elif self.problem_type == "categorical":
            self.fn_list = ['general_chat','Summary statistics explainer','Domain Explainer','Crosstab of two variables','Chi Square Statistics for relationship','Bar chart visualization', 'Two variable histogram visualization', 'Probability distribution visualization', 'Missing Number plot visualization','Two variable cross tabulation chart', 'Heatmap of dataset', 'Confusion matrix explainer', 'inference_predict_classify_from_classification model','Most_Correlated_Features_explainer_csv', 'NO function matches found form list','what will be '+self.target_variable+' type','classify','categorize','give class','give type','give class']
        
        elif self.problem_type == "numerical":
            self.fn_list = ['general_chat','Summary statistics explainer','Domain Explainer','Bar chart visualization', 'Two variable histogram visualization', 'Probability distribution visualization', 'Missing Number plot visualization', 'Heatmap of dataset', 'Pairwise plot explainer','Most_Correlated_Features_explainer_csv','predict', 'inference_predict_from regression model', 'NO function matches found form list','what will be '+self.target_variable+' value','predict','give prediction']
        
    def run_analyzer(self, query):
#         print("i am query",query)
        fn_found = self.chatbot.function_caller(self.fn_list, query)
        print("The suggested function is : ", fn_found)
        if fn_found == None:
            response1 = self.chatbot.gen_exceptionchat('No function found relevant to your Query')
            response2=self.general_chat(query)
            response = response1 +' '+ response2
            
        elif fn_found == 'NO function matches found form list' or 'No' in fn_found:
            response1 = self.chatbot.gen_exceptionchat('No function found relevant to your Query')
            response2=self.general_chat(query)
            response = response1 +' '+ response2
            #print("No function found relevant to your Query\n")
            #print("Loading DeepSeek Coder LM...")
            #code_generation = CodeGeneration(query, self.data)
            #code_generation.run_process()
        elif fn_found == 'correlation matrix explainer':
            response=self.corrmat_explainer(query)
        elif fn_found == 'Summary statistics explainer':
            response=self.summary_stats_explainer(query)
        elif fn_found == 'Domain Explainer':
            response=self.analyze_user_question(query)
        elif fn_found == 'Most_Correlated_Features_explainer_csv':
            response= self.Most_Corr_Features_explainer(query)
        elif fn_found == 'Training loss Explainer':
            response=self.training_loss_explainer(query)            
        elif fn_found == 'Bar chart visualization':
            response=self.bar_chart_visualization(query)
        elif fn_found == 'Two variable histogram visualization':
            response=self.histogram_visualization(query)
#         elif fn_found == 'Two variable Stacked column chart':
#             self.stacked_column_chart(query)
        elif fn_found == 'Probability distribution visualization':
            response=self.prob_dist_visualization(query)           
        elif fn_found == 'Missing Number plot visualization':
            response= self.missing_num_plot_explainer(query)
        elif fn_found == 'Two variable cross tabulation chart':
            response=self.CrossTabulation_explainer(query)
        elif fn_found == 'Heatmap of dataset':
            response=self.heatmap_explainer(query)       
        elif fn_found == 'Chi Square Statistics for relationship':
            response=self.chi_stats_explainer(query)
        elif fn_found == 'Crosstab of two variables':
            response=self.crosstab_explainer(query)
        elif fn_found == 'Confusion matrix explainer':
            response=self.confusion_matrix_explainer(query)
        elif fn_found == 'Pairwise plot explainer':
            response= self.Pairplot_explainer(query)
        elif fn_found =='top_most_important_correlated_features_csv':
            response= self.top_most_important_features(query)
        elif fn_found == 'inference_predict_from regression model' or fn_found == 'predict' or fn_found =='what will be '+self.target_variable+' value'or fn_found =='predict'or fn_found =='give prediction':
            response= self.inference_regression(query)
        elif fn_found == 'inference_predict_classify_from_classification model' or fn_found == 'what will be '+self.target_variable+' type' or fn_found =='classify'or fn_found == 'categorize'or fn_found == 'give class'or fn_found == 'give type'or fn_found == 'give class':
            response= self.inference_classification(query)
        elif fn_found == 'inference_forcasting_timeseries'or fn_found == 'what will be '+self.target_variable+'on' or fn_found == 'forcast'or fn_found == 'give forcast':
            response= self.inference_timeseries(query)
        else:
            print("No function found relevant to your Query\n")
            response1 = self.chatbot.gen_exceptionchat('No function found relevant to your Query')
            response2=self.general_chat(query)
            response = response1 +' '+ response2
            #print("Loading DeepSeek Coder LM...")
            #code_generation = CodeGeneration(query, self.data)
            #code_generation.run_process()
        return response
            
    def analyzer_main(self):
        
        while True:
            try:
                clear_output(wait=True)
                query = input("Write your query & Enter 'exit' to end : ")

                if query.lower() == 'exit':
                    break  # exit the loop if the user enters 'exit'
                else:
                    self.run_analyzer(query)
            except Exception as e:
                # Code to handle the exception
                print(f"An exception occurred: {e}")

        print("Chat ended!")
            
            
# # Llama Functions by Hamza
            
#     def corrmat_explainer(self, query):
#         dependent_variable = self.dependent_var
#         value = 'Most correlated Features with dependent Variable '
#         found_path = self.json_file[value]

#     #         found_path = self.chatbot.file_path_extractor(self.json_file , query)
#         print("The path found : ", found_path)
#         if found_path is None:
#             found_path = input("Failed to determine the path please enter path of your relevant file : ")

#         dependent_variable = self.chatbot.inference_label_col(self.data, query)
#         print("Dependent_variable: ", dependent_variable)
#         if dependent_variable == []:
#             dependent_variable = [input("Model fails to recognize your dependent variable, please specify correct variable name")]
#         corrmat = pd.read_csv(found_path)
#         response = self.chatbot.corrmat_explain(corrmat, dependent_variable)
#         print(response)
# #         return response
        
# #     def summary_stats_explainer(self, query):
# #         value = 'Summary Statistics '
# #         found_path = self.json_file[value]
# # #         found_path = self.chatbot.file_path_extractor(self.json_file , query)
# #         print("The path found : ", found_path)
# #         if found_path is None:
# #             found_path = input("Failed to determine the path please enter path of your relevant file: ")
# #         stats = pd.read_csv(found_path)
# # #         print("Summary statistics", stats)
# #         response = self.chatbot.description(stats,query)
# #         print(response)
# #         return response
        
        
#     def analyze_user_question(self, question, user_specified_domain='any'):
#         try:
#             if not user_specified_domain:
#                 user_specified_domain = "any"
#             # Additional processing or analysis logic can be added here
#             analysis_result = self.chatbot.user_domain_question(self.data, question,user_specified_domain)
#             print(analysis_result)
#             return analysis_result
#         except FileNotFoundError as e:
#             return f"File not found: {e}"
#         except Exception as e:
#             return f"An error occurred: {e}"
        
#     def chi_stats_explainer(self, query):
#         var_found = self.chatbot.varaible_selector(self.data, query)
#         print(f"Explaining chi Square Stats .. ")
#         found_path = self.chatbot.file_path_extractor(self.json_file , query)
#         chi_table = pd.read_csv(found_path)
#         print("Chi-Square Table : \n ", chi_table)
#         if len(var_found) >1:
#             self.chatbot.chi_square(chi_table,query,var1 = var_found[0],var2 = var_found[1])
#         else:
#             self.chatbot.chi_square(chi_table,query)

#     def crosstab_explainer(self,query):
#         var_found = self.chatbot.varaible_selector(self.data, query)
#         print(f"Showing the Cross Tabulation for varaibles {var_found[0]} and {var_found[1]}")
#         found_path = self.chatbot.file_path_extractor(self.json_file , query)      
#         if os.path.isfile(f"{found_path}_{var_found[0]}_vs_{var_found[1]}.csv"):
#             full_path = f"{found_path}_{var_found[0]}_vs_{var_found[1]}.csv"
#         elif os.path.isfile(f"{found_path}_{var_found[1]}_vs_{var_found[0]}.csv"):
#             full_path = f"{found_path}_{var_found[1]}_vs_{var_found[0]}.csv"
#         else:
#             full_path = None
#             print("Sorry ! No relevant file found.")
                

#         print("File found for your query : ", full_path)
#         crosstab = pd.read_csv(found_path)
#         response = self.chatbot.cross_tab(crosstab,query)
#         print(response) 
#     def top_most_important_features(self, query):
#         value = 'All correlated Features with dependent Variable '
#         found_path = self.json_file[value]
# #         found_path = self.chatbot.file_path_extractor(self.json_file , query)
#         print("The path found : ", found_path)
#         if found_path is None:
#             found_path = input("Failed to determine the path please enter path of your relevant file : ")
#         path = pd.read_csv(found_path)
#         response = self.chatbot.imp_features(path, query)
#         print(response)
        
            
# Graph description codes

    def training_loss_explainer(self, query):
        prompt = "USER: <image>\n The given Image is a training loss curve of a machine learning model, You are required to   \nASSISTANT:"
        found_path = self.chatbot.file_path_extractor(self.json_file , query)
        self.img_to_text.display_image(found_path)
        print("File found for your query : ",found_path)
        response = self.img_to_text.process(found_path,prompt)
        start_index = response.find("ASSISTANT:") + len("ASSISTANT:")
        extracted_text = response[start_index:].strip()
        print(extracted_text)
        return extracted_text
    
    def bar_chart_visualization(self, query):
        var_found = self.chatbot.varaible_selector(self.data, query)
        print("Showing the Bar chart for varaible : ",var_found)
        value = 'Bar charts '
        found_path = self.json_file[value]
#         found_path = self.chatbot.file_path_extractor(self.json_file, query)      
        img_path = f"{found_path}_{var_found[0]}.png"
        print("File found for your query : ",img_path)
        self.img_to_text.display_image(img_path)
        response = self.img_to_text.BarCharts(img_path,query,var_found[0])
        print(response)
        return response
        
    def histogram_visualization(self, query):
        var_found = self.chatbot.varaible_selector(self.data, query)
        print(f"Showing the Bar chart for varaible {var_found[0]} and {var_found[1]}")
        value = 'Histogram of two columns '
        found_path = self.json_file[value]
#         found_path = self.chatbot.file_path_extractor(self.json_file , query)      
        if os.path.isfile(f"{found_path}_{var_found[0]}_vs_{var_found[1]}.png"):
            full_path = f"{found_path}_{var_found[0]}_vs_{var_found[1]}.png"
        elif os.path.isfile(f"{found_path}_{var_found[1]}_vs_{var_found[0]}.png"):
            full_path = f"{found_path}_{var_found[1]}_vs_{var_found[0]}.png"
        else:
            full_path = input("Failed to get your path please write path manually")
            
                
        print(f"{found_path}_{var_found[0]}_vs_{var_found[1]}.png")
        print(f"{found_path}_{var_found[1]}_vs_{var_found[0]}.png")
        print("File found for your query : ",full_path)
        self.img_to_text.display_image(full_path)
        response = self.img_to_text.Histogram(full_path,query,var_found[0], var_found[1])
        print(response)
        return response
        
    def stacked_column_chart(self, query):
        var_found = self.chatbot.varaible_selector(self.data, query)
        print(f"Showing the Stacked Bar chart for varaible {var_found[0]} and {var_found[1]}")
        found_path = self.chatbot.file_path_extractor(self.json_file , query)      
        if os.path.isfile(f"{found_path}_{var_found[0]}_vs_{var_found[1]}.png"):
              full_path = f"{found_path}_{var_found[0]}_vs_{var_found[1]}.png"
        elif os.path.isfile(f"{found_path}_{var_found[1]}_vs_{var_found[0]}.png"):
              full_path = f"{found_path}_{var_found[1]}_vs_{var_found[0]}.png"
        else:
              full_path = None

        print("File found for your query : ",full_path)
        self.img_to_text.display_image(full_path)
        response = self.img_to_text.StackedBarChart(full_path,query,var_found[0], var_found[1])
        print(response)
        return response
        
    def prob_dist_visualization(self,query):
        print(f"Showing the Probability distribution of {self.file_name}")
        value = 'Probability distributions '
        found_path = self.json_file[value]
#         found_path = self.chatbot.file_path_extractor(self.json_file , query)
        self.img_to_text.display_image(found_path)
        response = self.img_to_text.Distributions(found_path, query)
        print(response)
        return response
        
    def heatmap_explainer(self,query):
        print(f"Showing the Heatmap of {self.file_name}")
#         found_path = self.chatbot.file_path_extractor(self.json_file , query)
        value = self.json_file['Correlation Heatmap ']
        self.img_to_text.display_image(value)
        response = self.img_to_text.Heatmap_explainer(value, query)
        print(response)
        return response
        
    def missing_num_plot_explainer(self,query):
        print(f"Showing the Missing num plot of {self.file_name} before cleaning")
        value = self.json_file["Missing number plot before cleaning "]
#         found_path = self.chatbot.file_path_extractor(self.json_file , query)
        self.img_to_text.display_image(value)
        response = self.img_to_text.Missing_Number(value, query)
        print(response)
        return response
    
    def Most_Corr_Features_explainer(self,query):
        print(f"Showing the most correlated Features with dependent variable")
        value = self.json['Most correlated Features with dependent Variable ']
        found_path = self.chatbot.file_path_extractor(self.json_file , query)
        self.img_to_text.display_image(found_path)
        response = self.img_to_text.Most_Correlated_Features(found_path, query)
        print(response)
        return response
        
    def Trend_Graph_explainer(self,query):
        print(f" Trend Graph of {self.file_name}")
        found_path = self.chatbot.file_path_extractor(self.json_file , query)
        self.img_to_text.display_image(found_path)
        response = self.img_to_text.Trend_Graph(found_path, query)
        print(response)
        return response
        
    def Pairplot_explainer(self,query):
        print(f"Showing the Pairplot of {self.file_name}")
        found_path = self.chatbot.file_path_extractor(self.json_file , query)
        self.img_to_text.display_image(found_path)
        response = self.img_to_text.PairWise_Graph(found_path, query)
        print(response)
        return response
        
    def CrossTabulation_explainer(self, query):
        var_found = self.chatbot.varaible_selector(self.data, query)
        print(f"Showing the Cross Tabulation for varaibles {var_found[0]} and {var_found[1]}")
        found_path = self.chatbot.file_path_extractor(self.json_file , query)      
        if os.path.isfile(f"{found_path}_{var_found[0]}_vs_{var_found[1]}.png"):
              full_path = f"{found_path}_{var_found[0]}_vs_{var_found[1]}.png"
        elif os.path.isfile(f"{found_path}_{var_found[1]}_vs_{var_found[0]}.png"):
              full_path = f"{found_path}_{var_found[1]}_vs_{var_found[0]}.png"
        else:
              full_path = None

        print("File found for your query : ", full_path)
        self.img_to_text.display_image(full_path)
        response = self.img_to_text.CrossTabulation(full_path,query,var_found[0], var_found[1])
        print(response)
        return response
    def extract_json_from_text(self, query):
        try:
            # Find the JSON-like string, including the curly braces
            pattern = r'\{.*?\}'  # Non-greedy match to handle cases without nested structures
            match = re.search(pattern, query)
            if not match:
                return "No JSON-like data found."

            json_string = match.group()

            # Replace single quotes with double quotes to make it a valid JSON
            json_string = json_string.replace("'", '"')

            # Parse the JSON string
            json_data = json.loads(json_string)
            return json_data
        except (re.error, AttributeError, json.JSONDecodeError) as e:
            return f"Failed to parse JSON: {e}"       
      
    def inference_regression(self,query):
        try:
            file_name = self.file_name
            problem_type = self.problem_type
            source = self.source
            self.predictioncolumns = self.extract_json_from_text(query)
#             file_name = input("Enter the name of the file you want to inference from: ")
#             if not file_name:
#                 raise ValueError("File name cannot be empty.")

#             problem_type = input("Enter the problem type: ")
#             if not problem_type:
#                 raise ValueError("Problem type cannot be empty.")

#             source = input("Enter the source data: ")
#             if not source:
#                 raise ValueError("Source data cannot be empty.")

            target_variable = self.target_variable
            if not target_variable:
                raise ValueError("Target variable cannot be empty.")

            input_data = self.predictioncolumns
#             input_data = eval(input_data_str)
            if not isinstance(input_data, dict):
                raise ValueError("Input data should be dictionary.")

            file_path = f"Knowledge/{problem_type}/{source}/{self.process_id}/dataset/{file_name}"
            print("File path:", file_path)

            df = pd.read_csv(file_path)
            input_data_df = pd.DataFrame(input_data)

            inf = Inference_Regression(df, problem_type, source, self.process_id, target_variable)
            results = inf.regression_model(input_data_df)
            print(results)
            response = self.chatbot.inference_regression(df, input_data, target_variable, results)
            print(response)
            return response
#             print(f"The value of {target_variable} is: {results[0]}")

        except FileNotFoundError:
            print("File not found. Please ensure the file path is correct.")
            return None
        except ValueError as ve:
            print("Error:", ve)
            return None
        except Exception as e:
            print("An unexpected error occurred:", e)
            return None
  
    def inference_classification(self,query):
        try:
            file_name = self.file_name
            problem_type = self.problem_type
            source = self.source
#             file_name = input("Enter the name of the file you want to inference from: ")
#             if not file_name:
#                 raise ValueError("File name cannot be empty.")

#             problem_type = input("Enter the problem type: ")
#             if not problem_type:
#                 raise ValueError("Problem type cannot be empty.")

#             source = input("Enter the source data: ")
#             if not source:
#                 raise ValueError("Source data cannot be empty.")
            target_variable = self.target_variable
            if not target_variable:
                raise ValueError("Target variable cannot be empty.")

            input_data = self.extract_json_from_text(query)
            print("my_classification_test_str",input_data)
#             input_data = eval(input_data_str)
            if not isinstance(input_data, dict):
                raise ValueError("Input data should be a dictionary.")

            file_path = f"Knowledge/{problem_type}/{source}/{self.process_id}/dataset/{file_name}"
            print("File path:", file_path)

            df = pd.read_csv(file_path)
            input_data_df = pd.DataFrame(input_data)
            if 'Unnamed: 0' in input_data_df.columns:
                input_data_df.drop(['Unnamed: 0'], axis=1, inplace=True)

            inf = Inference_Classification(df, problem_type, source, self.process_id, target_variable)
            results = inf.classification_model(input_data_df)
            
            response = self.chatbot.inference_classification(df, input_data, target_variable, results)
            print(response)
            return response

        except FileNotFoundError:
            print("File not found. Please ensure the file path is correct.")
            return None
        except ValueError as ve:
            print("Error:", ve)
            return None
        except Exception as e:
            print("An unexpected error occurred:", e)
            return None

    def extract_datetime(self,query):
        # List of date formats to try
        date_formats = [
            "%d-%m-%Y %H:%M:%S", "%d-%m-%Y", "%Y-%m-%d", "%Y-%m", "%m-%d-%Y %H:%M:%S",
            "%Y", "%m/%d/%Y %H:%M", "%d/%m/%Y %H:%M", "%m/%d/%Y", "%d/%m/%Y",
            "%m-%d-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S",
            "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S",
            "%Y-%m-%d %H:%M", "%d-%m-%Y %H:%M", "%m-%d-%Y %H:%M", "%Y/%m/%d %H:%M", "%d/%m/%Y %H:%M",
            "%m/%d/%Y %H:%M", "%Y-%m-%d %H:%M", "%d-%m-%Y %H:%M", "%m-%d-%Y %H:%M",
            "%Y-%m-%d","%Y-%m-%d %H:%M:%S", "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y",
            "%Y%m%d", "%d%m%Y", "%m%d%Y", "%Y%m", "%m%Y",
            "%Y-%m-%d %H:%M:%S.%f", "%d-%m-%Y %H:%M:%S.%f", "%m-%d-%Y %H:%M:%S.%f", # Formats including milliseconds
            "%Y/%m/%d %H:%M:%S.%f", "%d/%m/%Y %H:%M:%S.%f", "%m/%d/%Y %H:%M:%S.%f",
            "%b %d %Y", "%b %d, %Y", "%d %b %Y", "%d %b, %Y", # Formats with abbreviated month names
            "%B %d %Y", "%B %d, %Y", "%d %B %Y", "%d %B, %Y" # Formats with full month names
        ]

        # Try each format explicitly
        for format in date_formats:
            try:
                dt = datetime.strptime(query, format)
                return dt
            except ValueError:
                continue

        # If no format matches, try fuzzy parsing
        try:
            dt = parser.parse(query, fuzzy=True)
            return dt
        except ValueError:
            return "No valid date found in the text."

    # Example usage
    #query = "I need the data by 10-05-2024 and make sure it's ready."
    #extracted_date = extract_datetime(query)
    #print(extracted_date)
    def deduce_date_format(self,date_str, formats):
        for fmt in formats:
            try:
                datetime.strptime(date_str, fmt)
                return fmt
            except ValueError:
                continue
        return None            
    def inference_timeseries(self,query):
        file_name = self.file_name
        problem_type = self.problem_type
        source = self.source
        self.predictiondate= self.extract_datetime(query)                      
        file_path = f"Knowledge/{problem_type}/{source}/{self.process_id}/dataset/{file_name}"
        print("File path:", file_path)

        df = pd.read_csv(file_path)
        input_data_df = pd.DataFrame(df)
        if 'Unnamed: 0' in input_data_df.columns:
            input_data_df.drop(['Unnamed: 0'], axis=1, inplace=True)
        target_variable = self.target_variable
        time_series_column= self.datetime_column
        input_data_df[time_series_column] = pd.to_datetime(input_data_df[time_series_column], errors='coerce')
        
                # Assuming you have a string date and time
        date_string = self.predictiondate
        # List of potential date formats
        potential_formats = [
            "%d-%m-%Y %H:%M:%S", "%d-%m-%Y", "%Y-%m-%d", "%Y-%m", "%m-%d-%Y %H:%M:%S",
            "%Y", "%m/%d/%Y %H:%M", "%d/%m/%Y %H:%M", "%m/%d/%Y", "%d/%m/%Y",
            "%m-%d-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S",
            "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S",
            "%Y-%m-%d %H:%M", "%d-%m-%Y %H:%M", "%m-%d-%Y %H:%M", "%Y/%m/%d %H:%M", "%d/%m/%Y %H:%M",
            "%m/%d/%Y %H:%M", "%Y-%m-%d %H:%M", "%d-%m-%Y %H:%M", "%m-%d-%Y %H:%M",
            "%Y-%m-%d","%Y-%m-%d %H:%M:%S", "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y",
            "%Y%m%d", "%d%m%Y", "%m%d%Y", "%Y%m", "%m%Y",
            "%Y-%m-%d %H:%M:%S.%f", "%d-%m-%Y %H:%M:%S.%f", "%m-%d-%Y %H:%M:%S.%f", # Formats including milliseconds
            "%Y/%m/%d %H:%M:%S.%f", "%d/%m/%Y %H:%M:%S.%f", "%m/%d/%Y %H:%M:%S.%f",
            "%b %d %Y", "%b %d, %Y", "%d %b %Y", "%d %b, %Y", # Formats with abbreviated month names
            "%B %d %Y", "%B %d, %Y", "%d %B %Y", "%d %B, %Y" # Formats with full month names
        ]
        # Assume the first non-null entry is representative
        sample_date = input_data_df[time_series_column].dropna().iloc[0]
        print("sample_date:",sample_date)
        # Deduce the format
        detected_format = self.deduce_date_format(str(sample_date), potential_formats)
        date_format = detected_format
        print("date_format:",date_format)
        #date_format = "%Y-%m-%d %H:%M:%S":
        print("date_string:",date_string)
        if date_string:
            print("formate of date:",date_string)
            formatted_date_string = date_string.strftime(date_format)
            print("date formate:",formatted_date_string )
            prediction_date = date_string
            if prediction_date is None:
                print(f"Error parsing date: no valid format found for {date_string}")
            else:
                prediction_date= date_string
                print(type(prediction_date))
            input_data_df = input_data_df.set_index(time_series_column)
            
            new_data = input_data_df[input_data_df.index < prediction_date]


            lstm_forecaster = LSTMForecaster(self.process_id, problem_type=problem_type, source=source,target_variable=target_variable,sequence_length=1)

            forecasted_value = lstm_forecaster.predict(new_data[[target_variable]].values, prediction_date)
            # Print the forecasted value
            print("Forecasted Value for", prediction_date, " :", forecasted_value)

            response = self.chatbot.inference_timeseries(input_data_df,target_variable,time_series_column, prediction_date,file_name,forecasted_value)
            print(response)
            return response
        else:
            return "Date is not available in your query"

#         input_data_df.set_index(time_series_column , inplace=True)

        # Example DataFrame
#         data = {
#             'date': [self.predictiondate]}
#         df = pd.DataFrame(data)
#         # Convert the date column to datetime format and set it as index
#         df['date'] = pd.to_datetime(df['date'])
                



        
        
        
# My langch work here 

    def general_chat(self,query):
        response = self.chatbot.gen_chat(query)
        return response
    def doc_chat(self,file_path,query):
        response = self.chatbot.generate_response_from_document(file_path, query)
        return response
        
    def summary_stats_explainer(self, query):
        value = 'Summary Statistics '
        found_path = self.json_file[value]
        print("real_time_file : ",found_path)
        print("query realtime_analysis : ",query)
        response = self.chatbot.generate_summary(found_path,query)
        return response

        
    def top_most_important_features(self, query):
        value = 'All correlated Features with dependent Variable '
        found_path = self.json_file[value]
        print("real_time_file : ",found_path)
        print("query realtime_analysis : ",query)
        result = self.chatbot.imp_features(found_path,query)
        return result
        
    def Most_Corr_Features_explainer(self, query):
        value = 'Most correlated Features with dependent Variable '
        found_path = self.json_file[value]
        print("real_time_file : ",found_path)
        print("query realtime_analysis : ",query)
        result = self.chatbot.imp_features(found_path,query)
        return result
        
        
        
        
        











         
        