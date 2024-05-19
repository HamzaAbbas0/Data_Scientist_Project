import pandas as pd
#from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from accelerate.utils import release_memory
import time
class DeepSeekLM:
    def __init__(self, model_name = "deepseek-ai/deepseek-llm-7b-chat"):
        # deepseek-ai/deepseek-llm-7b-chat
        self.tokenizer_deepseek = AutoTokenizer.from_pretrained(model_name)
        self.model_deepseek = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
        self.model_deepseek.generation_config = GenerationConfig.from_pretrained(model_name)
        self.model_deepseek.generation_config.pad_token_id = self.model_deepseek.generation_config.eos_token_id
        self.df = None

    def set_data(self, df):
        self.df = df
        
    def load_data(self, df):
        self.df = pd.read_csv(csv_file_path)
    
    def Deepseek_inference(self,question, data = None ,max_tokens = 2000):
        
        prompt=""
        
        if data is None:
            prompt = question
        else:
            prompt = f'Given a dataset below: {data}. {question}.'

        messages = [
            {"role": "user", "content": prompt}
        ]
        
        input_tensor = self.tokenizer_deepseek.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        outputs = self.model_deepseek.generate(input_tensor.to(self.model_deepseek.device), max_new_tokens=max_tokens)
        result = self.tokenizer_deepseek.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
        release_memory(self.model_deepseek.generate)
        time.sleep(3)
        #print(result)
        return result
    
    def Deepseek_json_inference(self,question, json ,max_tokens = 2000):
        
       
        prompt = f'Given the JSON file below contains description of files in keys and their path in values :  {json}. {question}.'

        messages = [
            {"role": "user", "content": prompt}
        ]
        
        input_tensor = self.tokenizer_deepseek.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        outputs = self.model_deepseek.generate(input_tensor.to(self.model_deepseek.device), max_new_tokens=max_tokens)
        result = self.tokenizer_deepseek.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
        release_memory(self.model_deepseek.generate)
        time.sleep(3)
        #print(result)
        return result
    
    def question_validity_inference(self, question): #data = self.df
        prompt = f"""Give your Advise, considering data and this question: {question}.Do not give explanations and justifications."""
        response = self.Deepseek_inference(prompt, self.df[:10], 40)
        print(response)
        return response
        
#     def question_type_inference(self, question): # data = self.df
        
#         prompt = f"""Consider the given question: '{question}'can you determine the most relevant data analysis domain or technique for this
#         task? Specify if it involves text analysis, time series analysis, predictive modeling, clustering, or any other appropriate
#         domain."""
#         response = self.Deepseek_inference(prompt,self.df)
#         print(response)
#         return response
    
    def question_type_inference(self, question): # data = self.df
        
        prompt = f"""and this given question: '{question}'. Which technique can be applied here? You can answer only one item from this list: ['time series', 'text analysis', 'predictive analytics','social media analytics']"""
        response = self.Deepseek_inference(prompt, self.df[:10], 50)
        print(response)
        return response
    
    def data_explorer(self):
        prompt1 = "I want to know how many rows and columns are there in the given dataset. Write only answers not code"
        response1 = self.Deepseek_inference(prompt1,self.df)
        prompt2 = "Enlist only all the columns in the given dataset. dont write any code"
        response2 = self.Deepseek_inference(prompt2,self.df[:5])
        return response1, response2
        
    def distribution_col_selection(self, query): # data = self.df
        
        prompt = f"""write the names of columns from the given dataset only that are present in this user query : {query}"""
        response = self.Deepseek_inference(prompt,self.df[:3])
        var_list = []
        for col in self.df.columns:
            if col.lower() in response.lower():
                var_list.append(col)
                
        return var_list        
        
    def correlation_analysis(self, corr_table): #data = self.df
        prompt = f"""Considering the given Correlation table: {corr_table}. Give analytics and draw your colnclusions """
        response = self.Deepseek_inference(prompt)
        print(response)
        return response
    
    def correlation_analysis(self, corr_table): #data = self.df
        prompt = f"""Considering the given correlation matrix: {corr_table}. Give analytics and draw your colnclusions """
        response = self.Deepseek_inference(prompt)
        print(response)
        return response
    
    def time_var_select(self, question, list_time_var): #data = self.df
        prompt = f"""and this user query: {question}. You can select only one item only from this list: {list_time_var} which you feel is relavant to the user query, don't include any other variable please, enlist only one."""
        response = self.Deepseek_inference(prompt, self.df[:2], 50)
        time_var = ""
        for col in list_time_var:
            if col.lower() in response.lower():
                time_var = col
                break
        print(response)
        
        if time_var == "":
            print("Taking First time variable as Time feature")
            time_var = list_time_var[0]
        return time_var
    
    def inference_label_col_colwise(self, df, question): #data = self.df
        if df is None:
            df = self.df
        else:
            df = df
         
        prompt = f""" and this question {question}, please specify only ONE dependent variable only from the colmuns of given dataset. Write the column name inside "" only. Do not write more than one dependent variable."""
        
        response = self.Deepseek_inference(prompt, df[:3], 30)
        print(response)
#         print(df.columns)
        dependant_var = ""
        for col in df.columns:
            if col.lower() in response.lower():
                dependant_var = [col]
                break

        return dependant_var

    def inference_label_col(self, df, question): #data = self.df
        if df is None:
            df = self.df
        else:
            df = df
         
        prompt = f""" and this question {question}, please specify only ONE dependent variable only from the colmuns of given dataset. Write the column name inside "" only. Do not write more than one dependent variable."""
        
        response = self.Deepseek_inference(prompt, df[:3], 30)
        print(response)
        response = response.split("\"")
#         print(df.columns)
        dependent_var = []
        
        for item in response:
            if item in df.columns:
                dependent_var = [item]
#                 dependant_var = [col]
                break
        if dependent_var == []:
            dependent_var = self.inference_label_col_colwise(df,question)
        if dependent_var == []:
            dependent_var = [input("System failed to detect dependent variable, please write the name of dependent variable : ")]
        return dependent_var
    
    def corr_cols_detector(self, df, question): #data = self.df
        """
        This function analyze user querry and return list of columns that user is looking to analyze for correlation analysis
        """
         
        prompt = f""" and this question {question}, please write the correct name of variables that user is asking for. Remember, you can write the variables only from the colmuns of given dataset. Write the column name inside "" only."""
        
        response = self.Deepseek_inference(prompt, self.df[:2], 30)
        corr_cols = []
        for col in self.df.columns:
            if col.lower() in response.lower():
                dependant_var = [col]

        return dependant_var
    
    def decide_problem_type(self, df, query):
        # Hard Codded value
#         decision_query = "I want to predict the capital loss variable using given dataset"
        dependent_var = self.inference_label_col(df, query)
        if dependent_var == [] or dependent_var == "":
            dependent_var = [input("Model fails to recognize your dependent variable, please specify correct variable name")]
        print("Dependent_variable: ", dependent_var)
        if df[dependent_var[0]].dtype == object:
            problem_type = "Categorical"
        if df[dependent_var[0]].dtype == int or df[dependent_var[0]].dtype == float:
            problem_type = "Numerical"
        return problem_type, dependent_var     
    
    def file_path_extractor(self, json, question): #data = self.df
         
        prompt = f""" and this user query {question}, You are required to select only 'ONE' path from the given json file which you feel suits the given user query. Note You can select only one path and dont write anything outside of json values."""
        
        response = self.Deepseek_json_inference(prompt, json, 100)
        print(response)
#         print(df.columns)
#         dependant_var = ""
#         for col in df.columns:
#             if col.lower() in response.lower():
#                 dependant_var = [col]
#                 break
    
              
       
       