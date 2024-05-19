from databrickschatbotapi.DatascientistPipeline.DataAquisition import DataAquisition
from databrickschatbotapi.DatascientistPipeline.FeatureSelection import FeatureSelector
from databrickschatbotapi.DatascientistPipeline.VARModel import VARModel
from databrickschatbotapi.DatascientistPipeline.CombineSheet import CombineSheet
import os
from databrickschatbotapi.DatascientistPipeline.deepseeklm import DeepSeekLM
import json
import pandas as pd
from databrickschatbotapi.DatascientistPipeline.DataAnalysis import DataAnalysis
from databrickschatbotapi.DatascientistPipeline.Denormalizer import Denormalizer
from databrickschatbotapi.DatascientistPipeline.Pix2StructAnalyzer import Pix2StructAnalyzer
from databrickschatbotapi.DatascientistPipeline.FeatureSelection import FeatureSelector
from databrickschatbotapi.DatascientistPipeline.Llama import LlamaInference
from databrickschatbotapi.DatascientistPipeline.LSTM_analysis import LSTMAnalysis
from databrickschatbotapi.DatascientistPipeline.LSTM_model import LSTMModel
import os
from databrickschatbotapi.DatascientistPipeline.EDA2 import EDA
from databrickschatbotapi.DatascientistPipeline.DataCleaning1 import DataCleaner
from databrickschatbotapi.DatascientistPipeline.DataTransformation1 import DataTransformer
from databrickschatbotapi.DatascientistPipeline.corr_analysis1 import CorrelationAnalysis
from databrickschatbotapi.DatascientistPipeline.modeling import Modeling
from databrickschatbotapi.DatascientistPipeline.RegressionModels.NeuralNetworkRegression import NeuralNetworkRegression
from databrickschatbotapi.DatascientistPipeline.RegressionModels.MultiLinearRegression import MultiLinearRegression
from IPython.display import display
from databrickschatbotapi.DatascientistPipeline.dynamic_format_final import DataScienceReport
#from databrickschatbotapi.DatascientistPipeline.Medflow_files.Medflow import Medflow
import torch
import sys
from databrickschatbotapi.ProcessLog.models import ProcessLog
from databrickschatbotapi.DataSciencePipelineResult.models import DataSciencePipelineResult
auth_token = "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"
import pandas as pd
import os
# from deepseeklm import DeepSeekLM
import json
import pandas as pd
# from Llama import LlamaInference
#from databrickschatbotapi.DatascientistPipeline.chatbot_final_file import Chatbot
from databrickschatbotapi.DatascientistPipeline.realtime_analysis import RealtimeAnalysis
from IPython.display import clear_output
from IPython.display import JSON
#from databrickschatbotapi.DatascientistPipeline.Graph_description import GraphDescriptionPipeline
import mercury as mr
from databrickschatbotapi.DatascientistPipeline.chatbot_final_file import Chatbot
from databrickschatbotapi.DatascientistPipeline.Graph_description import GraphDescriptionPipeline
# def run_data_pipeline(data):
#     print("data testing",data)
#     return "Testing function"




def run_data_pipeline(file_name, problem_type, target_variable, datetime_column, process_id):
    file_name = file_name
    problem_type = problem_type
    target_variable = target_variable
    date_index = datetime_column
    process_id = process_id
    
    print(file_name)
    print(problem_type)
    print(target_variable)
    print(date_index)
    print(process_id)
    
    df_read = file_name
    folder_path = file_name.split('/')[0]
    print(folder_path)
    file_name = file_name.split('/')[-1]
    print(file_name)
    source = file_name
    source = source.split('.')[-1]
    print(source)
    
    
    if file_name.endswith('.csv'):
        df = pd.read_csv(df_read)
    if file_name.endswith('.xlsx'):
        df = pd.read_csv(df_read)
    
    if problem_type.lower() == "time series":
        date_index = df[datetime_column] 
        type_column = df[target_variable]
        target_datatype = 'numerical'
        
    elif problem_type.lower() == "categorical" or datetime_column =='null':
        type_column = df[target_variable]
        target_datatype = None
        date_index = None
    elif problem_type.lower() == "numerical" or datetime_column =='null':
        type_column = df[target_variable]
        target_datatype = None
        date_index = None
        
    else:
        raise ValueError("Invalid problem type. Supported types are 'time series', 'categorical', and 'numerical'.")

          

 
    data_loader = DataAquisition(source, folder_path,file_name,process_id)
    # Craete Directories
    data_loader.make_directories(problem_type, process_id)
    #data_loader.move_dataset_to_knowledge(df)
    df.to_csv(f"Knowledge/{problem_type}/{source}/{process_id}/dataset/{file_name}")   
    print("till data loader its worked!!!")
    try:
        ProcessLog.add_log(process_id, "Data has been loaded successfully.")
    except Exception as e:
    # Log any failure
        ProcessLog.add_log(process_id, f"Pipeline process failed: {str(e)}")


    dependent_variable = target_variable
    ## Data Cleaning
    cleaning_instance = DataCleaner(source, df, file_name,process_id, problem_type, target_variable , dependent_variable)
    df1 = cleaning_instance.get_data()
    clean =df1.info()
    print("cleanining",clean)
    try:
        ProcessLog.add_log(process_id, "Data cleaning has been done.")
    except Exception as e:
# Log any failure
        ProcessLog.add_log(process_id, f"Pipeline process failed: {str(e)}")


    dependent_variable = target_variable
    ## EDA 
    EDA_instance = EDA(source, df1, file_name,process_id ,problem_type, type_column, dependent_variable,date_index)
    df =  EDA_instance.get_data()
    try:
        ProcessLog.add_log(process_id, "Exploratory data analysis has been done.")
    except Exception as e:
# Log any failure
        ProcessLog.add_log(process_id, f"Pipeline process failed: {str(e)}")




    dependent_variable = target_variable
    ## Feature Selection
    corr_instance = CorrelationAnalysis(source, df1, file_name,process_id, problem_type, type_column, dependent_variable, corr_thres = 0.3)
    df1 = corr_instance.get_data()
    print("Selected column: ", df1.columns)
    most_corr_df = corr_instance.get_most_corr_data()
    print("till correlation done")
    try:
        ProcessLog.add_log(process_id, "Variables Correlation Analysis has been done.")
    except Exception as e:
# Log any failure
        ProcessLog.add_log(process_id, f"Pipeline process failed: {str(e)}")
#                     print("most_corr_df: ", most_corr_df)
#                     print("df1: ", df1)


    ## Data Transformation
    if problem_type.lower() == "time series":
        tansformation_instance = DataTransformer(source, df1, file_name,process_id, problem_type, type_column, dependent_variable,date_index)
    else:
        tansformation_instance = DataTransformer(source, df1, file_name,process_id, problem_type, type_column, dependent_variable,date_index)
    # df = tansformation_instance.get_data()
    transformed_df = tansformation_instance.get_transformed_data()



    if problem_type.lower() == "time series":
        transformed_df = tansformation_instance.get_transformed_data()
        filled_scaler = tansformation_instance.get_filled_scaler()
        filled_encoder = tansformation_instance.get_filled_encoder()
        numeric_columns_names = tansformation_instance.numeric_columns
    elif problem_type.lower() == "categorical":
        transformed_df = tansformation_instance.transformed_data
        transformed_df_y = tansformation_instance.transformed_data_y
        filled_scaler = tansformation_instance.get_filled_scaler()
        filled_encoder = tansformation_instance.get_filled_encoder()
        filled_encoder_y = tansformation_instance.label_encoder_y
        numeric_columns_names = tansformation_instance.numeric_columns
        categorical_columns_names = tansformation_instance.categorical_columns
    elif problem_type.lower() == "numerical":
        transformed_df = tansformation_instance.transformed_data
        transformed_df_y = tansformation_instance.transformed_data_y
        filled_scaler = tansformation_instance.get_filled_scaler()
        filled_encoder = tansformation_instance.get_filled_encoder()
        filled_scaler_y = tansformation_instance.get_filled_scaler_y()
        numeric_columns_names = tansformation_instance.numeric_columns
        categorical_columns_names = tansformation_instance.categorical_columns

    print("transformed_df: ", transformed_df)
    print("Scaler_Type!!!!!!!",type(filled_scaler))
    
    try:
        ProcessLog.add_log(process_id, "Data transformation has been done.")
    except Exception as e:
# Log any failure
        ProcessLog.add_log(process_id, f"Pipeline process failed: {str(e)}")
#                     print("first: ", transformed_df.iloc[0,:])

    dependent_variable = target_variable
    ## Modeling
#     print("Selected transformed_df column: ", transformed_df.columns)
    modeling_instance = Modeling(source, transformed_df, file_name,process_id, problem_type, type_column, dependent_variable)
    if problem_type.lower() == "time series":
        modeling_instance.update_attributes(scaled_data = transformed_df,
                                            scaler = filled_scaler,
                                            sequence_length = 5,
                                            test_size=0.2, lstm_units=50,
                                            epochs = 2,
                                            numeric_columns = numeric_columns_names)
        modeling_instance.time_series(filled_encoder, filled_scaler)

    if problem_type.lower() == "categorical":
        modeling_instance.update_attributes(X_data = transformed_df,
                                            Y_data = transformed_df_y,
                                            scaler = filled_scaler,
                                            label_encoder = filled_encoder,
                                            label_encoder_y = filled_encoder_y,
                                            test_size=0.2,
                                            n_estimators = 150,
                                            max_depth = 10,
                                            epochs = 10,
                                           )
        modeling_instance.run_modeling()


    if problem_type.lower() == "numerical":
        multi_linear_reg = MultiLinearRegression(
            source = source,
            X = transformed_df, 
            y = transformed_df_y, 
            target_variable = dependent_variable, 
            scaler = filled_scaler, 
            scaler_y = filled_scaler_y, 
            label_encoder = filled_encoder , 
            file_name = file_name,
            process_id =process_id,
            problem_type = problem_type)
        mse = multi_linear_reg.evaluate()
        print("MSE: ",mse)
    try:
        ProcessLog.add_log(process_id, "Modeling has been done.")
    except Exception as e:
# Log any failure
        ProcessLog.add_log(process_id, f"Pipeline process failed: {str(e)}")
        
        
            # Hamza code for automated report generation
    try:
        df = pd.read_csv(f"Knowledge/{problem_type}/{source}/{process_id}/dataset/{file_name}")
        question = 'explain it'
        display(df.head())

        # Sample dictionary
        json_path = f'Knowledge/{problem_type}/{source}/{process_id}/json/file_paths.json'
        with open(json_path) as f:
            json_file = json.load(f)
        print(json_file)

        report = DataScienceReport(auth_token,df, "explain it",json_file, problem_type,process_id, source, file_name)
        report.text_analysis(auth_token, df,question)
        report.graph_analyze(question)

        report_file_path=report.generate_word_document()
        print("path of the docx file",report_file_path)
        
        print("Dynamic Report Created Successfully.")
        try:
            ProcessLog.add_log(process_id, "Analysis report has been generated.")
        except Exception as e:
            ProcessLog.add_log(process_id, f"Pipeline process failed: {str(e)}")
        result_json = json_file  # Example JSON data
#         report_file_path = '/path/to/your/file.docx'  # Example file path

        success, message = DataSciencePipelineResult.add_result(process_id, result_json, report_file_path)
    
        if success:
            print("Result successfully added.")
            try:
                ProcessLog.add_log(process_id, "Data analysis knowledge of this file has been transfered to chatbot.")
            except Exception as e:
        # Log any failure
                ProcessLog.add_log(process_id, f"Pipeline process failed: {str(e)}")
        else:
            print(f"Failed to add result: {message}")
        inputcolumnsdf = transformed_df
        inputcolumns = list(inputcolumnsdf.columns)
        return inputcolumns
    except FileNotFoundError as e:
        print("FileNotFoundError:", e)
    except Exception as e:
        print("An error occurred:", e)


    # Hamza code for automated report generation
    
    
    
    
    
from django.shortcuts import get_object_or_404
from databrickschatbotapi.DataSciencePipelineProcess.models import DataSciencePipelineProcess
from databrickschatbotapi.DatasetandInput.models import DatasetandInput


from django.shortcuts import get_object_or_404
from databrickschatbotapi.DataSciencePipelineProcess.models import DataSciencePipelineProcess
from databrickschatbotapi.DatasetandInput.models import DatasetandInput


def get_dataset_columns_by_process_id(process_id):
    # Get the DataSciencePipelineProcess object
    process = get_object_or_404(DataSciencePipelineProcess, id=process_id)

    # Retrieve the associated DatasetandInput object
    dataset = process.inputinfo

    # Extract individual column values
    dataset_id = dataset.id
    file_name = dataset.file.name
    target_variable = dataset.target_variable
    datetime_column = dataset.datetime_column
    problem_type = dataset.problem_type

    return file_name,target_variable,datetime_column,problem_type


    
    
    
    
    
    
    
    
    
#     try:
#         ProcessLog.add_log(process_id, "Pipeline process completed successfully.")
#     except Exception as e:
#         # Log any failure
#         ProcessLog.add_log(process_id, f"Pipeline process failed: {str(e)}")
        
#     result_json = {'some': 'data'}  # Example JSON data
#     report_file_path = '/path/to/your/file.docx'  # Example file path

#     success, message = DataSciencePipelineResult.add_result(process_id, result_json, report_file_path)
#     if success:
#         print("Result successfully added.")
#     else:
#         print(f"Failed to add result: {message}")

def general_chatbot_query(query):
    try:
    #clear_output(wait=True)
    #query = input("Write your query & Enter 'exit' to end : ")
        if query.lower() == 'exit':
            return "Exiting the system as requested."
        else:
            chatbot= Chatbot()
            response= chatbot.gen_chat(query)
            print(response)
            return response
    except Exception as e:
        # Code to handle the exception
        print(f"An exception occurred: {e}")
        return f"An exception occurred: {e}"
def image_chatbot_query(file_path,query):
    try:
    #clear_output(wait=True)
    #query = input("Write your query & Enter 'exit' to end : ")
        if query.lower() == 'exit':
            return "Exiting the system as requested."
        else:
            chatbot= GraphDescriptionPipeline()
            response=chatbot.generate_response_from_image(file_path, query)
            print(response)
            return response
    except Exception as e:
        # Code to handle the exception
        print(f"An exception occurred: {e}")
        return f"An exception occurred: {e}"
    
def doc_chatbot_query(file_path,query):
    try:
    #clear_output(wait=True)
    #query = input("Write your query & Enter 'exit' to end : ")
        if query.lower() == 'exit':
            return "Exiting the system as requested."
        else:
            chatbot= Chatbot()
            response=chatbot.generate_response_from_document(file_path, query)
            print(response)
            return response
    except Exception as e:
        # Code to handle the exception
        print(f"An exception occurred: {e}")
        return f"An exception occurred: {e}"
        
        

def data_chatbot_query(query,process_id,file_namepath,target_variable,datetime_column,problem_type):
    df_read = file_namepath
    folder_path = file_namepath.split('/')[0]
    print(folder_path)
    file_name = file_namepath.split('/')[-1]
    print(file_name)
    source = file_name
    source = source.split('.')[-1]
    print(source)
    auth_token = "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"
    df= f"Knowledge/{problem_type}/{source}/{process_id}/dataset/{file_name}"
    dfhead = pd.read_csv(df)
    # problem_type = "numerical"
    json_path = f"Knowledge/{problem_type}/{source}/{process_id}/json/file_paths.json"
    with open(json_path) as f:
        json_file = json.load(f)
    analyzer = RealtimeAnalysis(auth_token ,df , file_name,process_id, problem_type, source,json_file,target_variable,datetime_column)

    print("\n___ Visualization of first few rows of your data ___")
    display(dfhead.head())
    print(" \n___ The following data is available for the file you selected please, write your query Accordingly ___ \n")
    mr.JSON(json_file)
    #while True:
    try:
        #clear_output(wait=True)
        #query = input("Write your query & Enter 'exit' to end : ")
        if query.lower() == 'exit':
            return "Exiting the system as requested."
        else:
            response=analyzer.run_analyzer(query)
            print(response)
            return response
    except Exception as e:
        # Code to handle the exception
        print(f"An exception occurred: {e}")
        chatbot= Chatbot()
        response = chatbot.gen_exceptionchat(e)
        return response

        
def process_chatbot_query(query,process_id):
    if process_id:
        file_namepath,target_variable,datetime_column,problem_type =get_dataset_columns_by_process_id(process_id)
        if problem_type == "DocumentAnalysis":
            response = doc_chatbot_query(file_namepath,query)
            
            return response
        elif problem_type == "ImageAnalysis":
            response = image_chatbot_query(file_namepath,query)
            return response
        else:
            response=data_chatbot_query(query,process_id,file_namepath,target_variable,datetime_column,problem_type)
            return response
    else:
        response = general_chatbot_query(query)
        return response
    