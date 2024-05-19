from huggingface_hub import login
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import textwrap
import os

# langchain
import shutil
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from pdf2image import convert_from_path
from transformers import AutoTokenizer, TextStreamer, pipeline
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

class LlamaInference:

# Base Functions
    def __init__(self, auth_token,data=None):
        self.auth_token = auth_token
        login(self.auth_token)
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_auth_token=True)
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map='auto', torch_dtype=torch.float16, use_auth_token=True)
        ###################### Langchain work #####################
        #################################################

    def get_prompt(self, instruction, system_prompt):
        prompt_template = f"[INST]{system_prompt}{instruction}[/INST]"
        return prompt_template

    def cut_off_text(self, text, prompt):
        cutoff_phrase = prompt
        index = text.find(cutoff_phrase)
        if index != -1:
            return text[:index]
        else:
            return text

    def remove_substring(self, string, substring):
        return string.replace(substring, "")

    def generate(self, text, system_prompt, max_new_tokens):
        prompt = self.get_prompt(text, system_prompt)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda')
            outputs = self.model.generate(**inputs, 
                                          max_length=500,  # Adjust max_length as needed
                                          max_new_tokens=1000,  # Increase max_new_tokens
                                          eos_token_id=self.tokenizer.eos_token_id,
                                          pad_token_id=self.tokenizer.eos_token_id)
            final_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
#             print(final_outputs)
            final_outputs = self.cut_off_text(final_outputs, '</s>')
            final_outputs = self.remove_substring(final_outputs, prompt)


        return final_outputs

    def parse_text(self, text):
        wrapped_text = textwrap.fill(text, width=100)
        return wrapped_text + '\n\n'

# Pipeline Functions
    def inference_label_col(self, cleaned_data, question):
        columns_array = cleaned_data.columns.to_numpy()
        columns_str = '", "'.join(columns_array)

        DEFAULT_SYSTEM_PROMPT = """\
        You are an expert Data Scientist, perfectly extract dependent variables out of given variables list in the prompt.
        Don't give explanations and justifications."""

        SYSTEM_PROMPT = f"<<SYS>>\n{DEFAULT_SYSTEM_PROMPT}\n<</SYS>>\n\n"

        prompt = f'Given a dataset with list of columns: ["{columns_str}"]. {question}, please specify the dependent variable only from the given list. Write the column name inside "" only.'
        generated_text = self.generate(prompt, SYSTEM_PROMPT, 50)
        response = self.parse_text(generated_text)
        print(response)
        dependant_var = []
        for col in columns_array:
            if col.lower() in response.lower().replace("\n"," "):
                dependant_var.append(col)
                break
        if dependent_var == []:
            dependent_var = [input("The system failed to detect Dependent Variable, Write the name of variable : ")]
        return dependant_var
    
    
    def inference_independent_col(self, cleaned_data, question):
            columns_array = cleaned_data.columns.to_numpy()
            columns_str = '", "'.join(columns_array)

            DEFAULT_SYSTEM_PROMPT = """\
            You are an expert Data Scientist, perfectly extract dependent variables out of given variables list in the prompt.
            Don't give explanations and justifications."""

            SYSTEM_PROMPT = f"<<SYS>>\n{DEFAULT_SYSTEM_PROMPT}\n<</SYS>>\n\n"

            prompt = f'Given a dataset with list of columns: ["{columns_str}"]. {question}, please specify the independent variable(s) only from the given list. Write the column name(s) inside "" only.'
            generated_text = self.generate(prompt, SYSTEM_PROMPT, 50)
            response = self.parse_text(generated_text)
            print(response)
            independant_var = []
            for col in columns_array:
                if col.lower() in response.lower() or '"'+col.lower()+'"' in response.lower():
                     independant_var.append(col)

            return independant_var

    
    def question_validity_inference(self,cleaned_data, question):

    #     columns_array = cleaned_data.columns.to_numpy()

    #     # Join column names into a comma-separated string for the prompt
    #     columns_str = '", "'.join(columns_array)
    
        DEFAULT_SYSTEM_PROMPT = """\
        You are an expert Data Scientist, You reply to questions by analyzing the given dataset in the prompt.
        Don't provide explanations or justifications, write the answer only.
        If you don't know the answer to a question, please don't share false information."""

        SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS


        prompt = f"""Given a dataset: [{cleaned_data}]. \n {question}. Write Shorter responses only"""

        generated_text = self.generate(prompt,SYSTEM_PROMPT,200)
        response = self.parse_text(generated_text)

        return response

    def file_path_extractor(self, json, question):

        DEFAULT_SYSTEM_PROMPT = f"""\
        You are an expert json file reader, you perfectly provide path form this json file \n {json}. \n by analyzing the user query provided in the prompt. Remember, your job is not to generate answers just provide paths from the given json only."""

        SYSTEM_PROMPT = f"<<SYS>>\n{DEFAULT_SYSTEM_PROMPT}\n<</SYS>>\n\n"

        prompt = question
        generated_text = self.generate(prompt, SYSTEM_PROMPT, 100)
        response = self.parse_text(generated_text)
        print(response)
        path_found = None
        for value in json.values():
            if value.lower() in response.lower().replace("\n","") or '"'+value.lower()+'"' in response.lower().replace("\n",""):
                path_found = value
                break
        if path_found is None:
            path_found = input("Failed to determine the path please enter path of your relevant file : ")
        return path_found

    def function_caller(self, fn_list, question):
        print("List of all functions", fn_list)

        DEFAULT_SYSTEM_PROMPT = f"""\
        You are an expert Python Function caller, you perfectly select only ONE function from the given list of functions \n {fn_list}. \n by analyzing the user query provided in the prompt. Remember, "You can only select from the given list" your job is not to generate answers just select the right function to execute from the given list only."""

        SYSTEM_PROMPT = f"<<SYS>>\n{DEFAULT_SYSTEM_PROMPT}\n<</SYS>>\n\n"

        prompt = question
        generated_text = self.generate(prompt, SYSTEM_PROMPT, 100)
        response = self.parse_text(generated_text)
        print(response)
        
        for fn in fn_list:
            if fn.lower() in response.lower().replace("\n"," ") or '"'+fn.lower()+'"' in response.lower().replace("\n"," "):
                return fn
        else:
            return None
        
    def varaible_selector(self, cleaned_data, question):
        columns_array = cleaned_data.columns.to_numpy()
        columns_str = '", "'.join(columns_array)

        DEFAULT_SYSTEM_PROMPT = """\
        You are an expert Data Scientist, perfectly extract dependent variables out of given variables list in the prompt.
        Don't give explanations and justifications."""

        SYSTEM_PROMPT = f"<<SYS>>\n{DEFAULT_SYSTEM_PROMPT}\n<</SYS>>\n\n"

        prompt = f'Given the list of columns: ["{columns_str}"]. select column names only from the given list that are mentioned in this user query {question}. please specify the variable only from the given list. Write the column name(s) inside "" only, dont write with * please.'
        generated_text = self.generate(prompt, SYSTEM_PROMPT, 50)
        response = self.parse_text(generated_text)
        print(response)
        independant_var = []
        for col in columns_array:
            if col.lower() in response.lower() or '"'+col.lower()+'"' in response.lower():
                independant_var.append(col)

        return independant_var

    
# Hamza Functions


    def corrmat_explain(self, corrmat, dependent_var):
        dependent_var = None
        cor_target = corrmat[dependent_var]

        DEFAULT_SYSTEM_PROMPT = """\
        You are an expert Data Scientist. Provide concrete insights into the correlation matrix provided in the prompt. Offer explanations or justifications only where required. If you don't know the answer to a question, please don't share false information."""

        SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS


        prompt = f"""Analyze the relationship between the dependent variable '{dependent_var}' and other variables in the provided correlation matrix of target variable '{cor_target}'. Highlight the variable with the strongest correlation, discuss one with the weakest correlation, note any trends or patterns, and provide a summary. Be concise and avoid false details. 
Your query may look like: 'What are the most correlated features?', 'What is the domain of the data?', or 'What is the average refund?'.
"""
        
        generated_text = self.generate(prompt,SYSTEM_PROMPT,500)
        response = self.parse_text(generated_text)

        formatted_response = response.replace('. ', '.\n').replace('  ', ' ')

        return formatted_response
    
    def imp_features(self,path,query):

        DEFAULT_SYSTEM_PROMPT = """\
        You are an expert Data Scientist, you efficiently provide the concrete insights of important features provided in the prompt.
        You also give explanations or justifications only where required.
        If you don't know the answer to a question, please don't share false information."""

        SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS


        prompt = f""" analyze the provided {path} dataset to identify the most influential features contributing to the columns '{query}'. Highlight the feature with the highest importance (strongest impact), discuss insights on the one with the least importance (weakest impact), and recognize any notable trends or patterns. Ensure brevity and accuracy in your summary.\n"""

        generated_text = self.generate(prompt,SYSTEM_PROMPT,300)
        response = self.parse_text(generated_text)

        formatted_response = response.replace('. ', '.\n').replace('  ', ' ')

        return formatted_response
    
    
    def description(self, description, question):
        
        DEFAULT_SYSTEM_PROMPT = f"""\
            As a seasoned Data Scientist, your role is to provide a clear and concise summery statistics of the dataset based on user prompts, leveraging the information available in the provided dataset at {description}. Avoid unnecessary details and explanations, ensuring a focus on relevant insights. If there's any uncertainty about the answer, communicate it transparently, and refrain from offering inaccurate information.
        """
        SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS

        if description is None or description.empty:
            print("CSV file is missing or empty. Unable to generate a response.")
            return

        if not question:
            print("Question is missing. Unable to generate a response.")
            return

        prompt = f"""Given a summary statistics: [{description}]. \n {question}. Write response according to the user prompt"""
        generated_text = self.generate(prompt, SYSTEM_PROMPT, 500)
            # Process the generated text to obtain the final response
        
        response = self.parse_text(generated_text)
        
        formatted_response = response.replace('. ', '.\n').replace('  ', ' ')

        return formatted_response

    
    def user_domain_question(self, df, question,user_specified_domain='any'):
        DEFAULT_SYSTEM_PROMPT = f"""\
                    You are an expert {user_specified_domain} domain. Answer the user's questions about the dataset with concise and relevant information. If the question is specific to a column, provide insights accordingly. Avoid unnecessary details and explanations. If you're uncertain about the answer, be upfront and avoid providing inaccurate information."""
        SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS
        
        if df is None or df.empty:
            print("CSV file is missing or empty. Unable to generate a response.")
            return

        if not question:
            print("Question is missing. Unable to generate a response.")
            return
        
        prompt = f"""Given a dataset: [{df}]. \n {question}. Write response according to the user question"""
        generated_text = self.generate(prompt, SYSTEM_PROMPT, 500)
        response = self.parse_text(generated_text)
        
        formatted_response = response.replace('. ', '.\n').replace('  ', ' ')

        return formatted_response
    
    
    def dataset_intoduction(self, df):
        DEFAULT_SYSTEM_PROMPT = """\
        You are an expert Data Scientist, you efficiently provide the introduction based on the given dataset. If you don't know the answer to a question, please don't share false information."""

        SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS
        if len(df) <= 10:
            prompt = f"""Generate a general introduction based on the provided dataset {df} columns.\n"""
        else:
            prompt = f"""Generate a general introduction based on the provided dataset {df[:20]} columns.\n"""


        generated_text = self.generate(prompt, SYSTEM_PROMPT, 500)

        # Assuming that self.generate returns the generated text
        print(generated_text)
        return generated_text
    
    def cross_tab(self,df,question):
        DEFAULT_SYSTEM_PROMPT = """As an expert Data Scientist, your tasked with efficiently exploring the dataset. Leverage your expertise to generate a detailed cross-tabulation analysis based on the user's request.If you encounter uncertainty or lack information, avoid providing speculative details. Your insights are crucial for guiding the user without unnecessary model-generated content."""

        SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS
        if len(df) <= 10:
            prompt = f"""I have a dataset {df}, Could you please generate a cross-tabulation analysis on the user {question}. Additionally, If you don't know the answer to a question, please don't share false information..
\n"""
        else:
            prompt = f"""I have a dataset {df}, Could you please generate a cross-tabulation analysis on the user {question}. Additionally, if there are any notable trends or insights within the cross-tabulation, provide a brief interpretation. Be concise and avoid false details.\n"""


        generated_text = self.generate(prompt, SYSTEM_PROMPT, 500)

        # Assuming that self.generate returns the generated text
        print(generated_text)
        return generated_text
    
    
    
    def chi_square(self,chi_table,question,var1 = None,var2 = None):
        DEFAULT_SYSTEM_PROMPT = """As an expert Data Scientist, Leverage your expertise to explain chi-square statistic based on the user's request.If you encounter uncertainty or lack information, avoid providing speculative details. Your insights are crucial for guiding the user without unnecessary model-generated content."""

        SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS
        
        if var1 is None and var2 is None:
            prompt = f"""I have chi-square results {chi_table}, Please analyze the relationship between all the variables provide insights, including relevant chi-value and p-value information. Offer a concise interpretation of any significant associations found. Prioritize accuracy and avoid unnecessary details in your response.\n"""
        else:
            
            prompt = f"""I have chi-square results {chi_table}, Please analyze the relationship between {var1} and {var2} provide insights, including relevant chi-value and p-value information. Offer a concise interpretation of any significant associations found. Prioritize accuracy and avoid unnecessary details in your response.\n"""


        generated_text = self.generate(prompt, SYSTEM_PROMPT, 500)

        # Assuming that self.generate returns the generated text
        return generated_text

        
    def generate_research_questions(self, kb):

            DEFAULT_SYSTEM_PROMPT = """\
            User will provide you a JSON file containing table names and their corresponding schema. You just have to create research questions that can be answered using data in these table."""

            SYSTEM_PROMPT = f"<<SYS>>\n{DEFAULT_SYSTEM_PROMPT}\n<</SYS>>\n\n"

            prompt = f"""
            Here is the the tables and their schema, create few research questions.

            Start Table's Schema

            {kb}

            End Table's Schema

            """
          
            print(prompt)
            generated_text = self.generate(prompt, SYSTEM_PROMPT, 4000)
            response = self.parse_text(generated_text)
            print(response)
            
    def inference_regression(self, df, input_data, target_variable, results):
        
        DEFAULT_SYSTEM_PROMPT = f"""\
        As a skilled Data Scientist, you've obtained prediction results from a classification model. Please offer consise explanation of the predicted outcomes for the target variable '{target_variable}', ensuring clarity and accuracy in your response.
        """

        SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS

        prompt = f"""Given the prediction results from the regression model for the target variable '{target_variable}': {results} based on the provided input data: '{input_data}', kindly provide a concise explanation of the outcomes while maintaining accuracy and relevance.
        """

        generated_text = self.generate(prompt, SYSTEM_PROMPT, 150)
        response = self.parse_text(generated_text)

        return response
    
    

    def inference_classification(self, df, input_data, target_variable, results):
        
        DEFAULT_SYSTEM_PROMPT = f"""\
As a skilled Data Scientist, you've obtained prediction results from a classification model. Please offer an explanation of the predicted outcomes for the target variable '{target_variable}', ensuring clarity and accuracy in your response.
"""

        SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS

        prompt = f"""Given the prediction results from the classification model for the target variable '{target_variable}': {results} based on the provided input data: '{input_data}', kindly provide a concise explanation of the outcomes while maintaining accuracy and relevance.
        """


        generated_text = self.generate(prompt, SYSTEM_PROMPT, 150)
        response = self.parse_text(generated_text)

        return response

    def inference_timeseries(self, input_data_df, target_variable, time_series_column, prediction_date_input,file_name, forecasted_value):
        
        DEFAULT_SYSTEM_PROMPT = f"""\
    As a skilled Data Scientist, you've obtained prediction results from a time series model. Please provide an explanation of the forecasted value for the target variable '{target_variable}' at the prediction date '{prediction_date_input}', ensuring clarity and accuracy in your response.
    """

        SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS

        prompt = f"""Given the forecasted value '{forecasted_value}' for the target variable '{target_variable}' at the prediction date '{prediction_date_input}' based on the provided time series data of '{file_name}', please offer a concise explanation while maintaining accuracy and relevance.
    """

        generated_text = self.generate(prompt, SYSTEM_PROMPT, 150)

        # Remove the tags from the generated_text
        generated_text = generated_text.replace("[INST]<<SYS>>", "").replace("</SYS>>", "")

#         print(generated_text)
        response = self.parse_text(generated_text)

        return response


        
    