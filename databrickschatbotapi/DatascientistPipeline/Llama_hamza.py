from huggingface_hub import login
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import textwrap
import pandas as pd
from accelerate.utils import release_memory
from databrickschatbotapi.model_loader import llm_model
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"



class LlamaInference:

    def __init__(self, auth_token):
        self.auth_token = auth_token
        login(self.auth_token)
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_auth_token=True)
        #self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map='auto', torch_dtype=torch.float16, use_auth_token=True)
        self.model = llm_model

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
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                          eos_token_id=self.tokenizer.eos_token_id,
                                          pad_token_id=self.tokenizer.eos_token_id)
            final_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            final_outputs = self.cut_off_text(final_outputs, '</s>')
            final_outputs = self.remove_substring(final_outputs, prompt)
            release_memory(self.model.generate)

        return final_outputs

    def parse_text(self, text):
        wrapped_text = textwrap.fill(text, width=100)
        return wrapped_text + '\n\n'

    
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
            if col.lower() in response.lower():
                dependant_var.append(col)

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
        
        formatted_response3 = response.replace('. ', '.\n').replace('  ', ' ')

        return formatted_response3
    

#     def corrmat_explain(self,corrmat,valid_column_names):

# #     columns_array = cleaned_data.columns.to_numpy()

# #     # Join column names into a comma-separated string for the prompt
# #     columns_str = '", "'.join(columns_array)
    
#         DEFAULT_SYSTEM_PROMPT = """\
#         You are an expert Data Scientist, you efficiently provide the concrete insights of Correlation matrix provided in the prompt.
#         You also give explanations or justifications only where required.
#         If you don't know the answer to a question, please don't share false information."""

#         SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS


#         prompt = f"""Analyze the relationship between dependend '{valid_column_names}'variable with other variables in the provided {corrmat} dataset . Highlight the variable with the strongest correlation, discuss one with the weakest correlation, note any trends or patterns and summerization. Be concise and avoid false details.\n"""



#         generated_text = self.generate(prompt,SYSTEM_PROMPT,500)
#         response = self.parse_text(generated_text)
        
#         formatted_response2 = response.replace('. ', '.\n').replace('  ', ' ')

#         return formatted_response2

    def corrmat_explain(self,corrmat,dependent_var):
        cor_target = corrmat[dependent_var]

        DEFAULT_SYSTEM_PROMPT = """\
        You are an expert Data Scientist, you efficiently provide the concrete insights of Correlation matrix provided in the prompt.
        You also give explanations or justifications only where required.
        If you don't know the answer to a question, please don't share false information."""

        SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS


        prompt = f"""Analyze the relationship between dependend '{dependent_var}'variable with other variables in the provided {cor_target} dataset . Highlight the variable with the strongest correlation, discuss one with the weakest correlation, note any trends or patterns and summerization. Be concise and avoid false details.\n"""


        
        generated_text = self.generate(prompt,SYSTEM_PROMPT,300)
        response = self.parse_text(generated_text)

        formatted_response2 = response.replace('. ', '.\n').replace('  ', ' ')

        return formatted_response2
    
    
    def imp_features(self,corrmat,dependent_var):

        DEFAULT_SYSTEM_PROMPT = """\
        You are an expert Data Scientist, you efficiently provide the concrete insights of important features provided in the prompt.
        You also give explanations or justifications only where required.
        If you don't know the answer to a question, please don't share false information."""

        SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS


        prompt = f""" analyze the provided {corrmat} dataset to identify the most influential features contributing to the columns '{dependent_var}'. Highlight the feature with the highest importance (strongest impact), discuss insights on the one with the least importance (weakest impact), and recognize any notable trends or patterns. Ensure brevity and accuracy in your summary.\n"""

        generated_text = self.generate(prompt,SYSTEM_PROMPT,300)
        response = self.parse_text(generated_text)

        formatted_response1 = response.replace('. ', '.\n').replace('  ', ' ')

        return formatted_response1

    def description(self, csv_file_path):
        
        DEFAULT_SYSTEM_PROMPT = f"""\
            As a seasoned Data Scientist, your role is to provide a clear and concise summery statistics of the dataset, leveraging the information available in the provided dataset at {csv_file_path}. Avoid unnecessary details and explanations, ensuring a focus on relevant insights. If there's any uncertainty about the answer, communicate it transparently, and refrain from offering inaccurate information.
        """
        SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS

        if csv_file_path is None or csv_file_path.empty:
            print("CSV file is missing or empty. Unable to generate a response.")
            return

        prompt = f"""Given a dataset of summery staistics file: [{csv_file_path}]. \n """
        generated_text = self.generate(prompt, SYSTEM_PROMPT, 300)
            # Process the generated text to obtain the final response
        
        response = self.parse_text(generated_text)
        
        formatted_response = response.replace('. ', '.\n').replace('  ', ' ')

        return formatted_response
    
    def imp_features(self,corrmat,valid_column_names):
        
        DEFAULT_SYSTEM_PROMPT = """\
        You are an expert Data Scientist, you efficiently provide the concrete insights of important features provided in the prompt.
        You also give explanations or justifications only where required.
        If you don't know the answer to a question, please don't share false information."""

        SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS


        prompt = f""" analyze the provided {corrmat} dataset to identify the most influential features contributing to the columns '{valid_column_names}'. Highlight the feature with the highest importance (strongest impact), discuss insights on the one with the least importance (weakest impact), and recognize any notable trends or patterns. Ensure brevity and accuracy in your summary.\n"""

        generated_text = self.generate(prompt,SYSTEM_PROMPT,300)
        response = self.parse_text(generated_text)
        
        formatted_response1 = response.replace('. ', '.\n').replace('  ', ' ')

        return formatted_response1
    
    
#     def corrmat_explain(self,corrmat,dependent_var):
#         cor_target = corrmat[dependent_var]

#         DEFAULT_SYSTEM_PROMPT = """\
#         You are an expert Data Scientist, you efficiently provide the concrete insights of Correlation matrix provided in the prompt.
#         You also give explanations or justifications only where required.
#         If you don't know the answer to a question, please don't share false information."""

#         SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS


#         prompt = f"""Analyze the relationship between dependend '{dependent_var}'variable with other variables in the provided {cor_target} dataset . Highlight the variable with the strongest correlation, discuss one with the weakest correlation, note any trends or patterns and summerization. Be concise and avoid false details.\n"""


        
#         generated_text = self.generate(prompt,SYSTEM_PROMPT,500)
#         response = self.parse_text(generated_text)

#         formatted_response2 = response.replace('. ', '.\n').replace('  ', ' ')

#         return formatted_response2
    
    
    def most_corrmat_explain(self,corrmat,valid_column_names):
    
        DEFAULT_SYSTEM_PROMPT = """\
        You are an expert Data Scientist, you efficiently provide the concrete insights of Most Correlation matrix provided in the prompt.
        You also give explanations or justifications only where required.
        If you don't know the answer to a question, please don't share false information."""

        SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS


        prompt = f"""Analyze the Most Correlated Features in the provided {corrmat} dataset . Highlight the variable with the strongest correlation features, discuss one with the weakest correlation features, note any trends or patterns and summerization. Be concise and avoid false details.\n"""



        generated_text = self.generate(prompt,SYSTEM_PROMPT,300)
        response = self.parse_text(generated_text)
        
#         formatted_response2 = response.replace('. ', '.\n').replace('  ', ' ')

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
            if value.lower() in response.lower().replace("\n"," ") or '"'+value.lower()+'"' in response.lower().replace("\n"," "):
                path_found = value
                break
        return path_found
    
    def intoduction(self, df):
        DEFAULT_SYSTEM_PROMPT = """\
        You are an expert Data Scientist, you efficiently provide the introduction based on the given dataset. If you don't know the answer to a question, please don't share false information."""

        SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS

        prompt = f"""Generate a Overall general introduction based on the provided dataset columns {df.columns}.If you encounter uncertainty or lack information, avoid providing speculative details.\n"""


        generated_text = self.generate(prompt, SYSTEM_PROMPT, 300)

        # Assuming that self.generate returns the generated text
        return generated_text
    
    def cross_tab(self,df,cross_tab1=None,cross_tab2=None):
        DEFAULT_SYSTEM_PROMPT = """As an expert Data Scientist, your tasked with efficiently exploring the dataset. Leverage your expertise to generate a detailed cross-tabulation analysis based on the user's request.If you encounter uncertainty or lack information, avoid providing speculative details. Your insights are crucial for guiding the user without unnecessary model-generated content."""

        SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS

        prompt = f"""I have a dataset with various variables, and I'm interested in understanding the relationships between specific factors. Could you please perform a cross-tabulation analysis for {cross_tab1} and {cross_tab2} Provide insights into any notable trends or associations. Be concise and avoid unnecessary details.\n"""


        generated_text = self.generate(prompt, SYSTEM_PROMPT, 300)

        # Assuming that self.generate returns the generated text
        return generated_text
    
    
    
    def chi_square(self,chi_table,chi_square1 = None,chi_square2 = None):
        DEFAULT_SYSTEM_PROMPT = """As an expert Data Scientist, Leverage your expertise to explain chi-square statistic based on the user's request.If you encounter uncertainty or lack information, avoid providing speculative details. Your insights are crucial for guiding the user without unnecessary model-generated content."""

        SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS
        
        if chi_square1 is None and chi_square2 is None:
            prompt = f"""I have chi-square results {chi_table}, Please analyze the relationship between all the variables provide insights, including relevant chi-value and p-value information. Offer a concise interpretation of any significant associations found. Prioritize accuracy and avoid unnecessary details in your response.\n"""
        else:
            
            prompt = f"""I have chi-square results {chi_table}, Please analyze the relationship between {chi_square1} and {chi_square2} provide insights, including relevant chi-value and p-value information. Offer a concise interpretation of any significant associations found. Prioritize accuracy and avoid unnecessary details in your response.\n"""


        generated_text = self.generate(prompt, SYSTEM_PROMPT, 500)

        # Assuming that self.generate returns the generated text
        return generated_text
    
    
    
    
    def confusion_matrix(self,df):
        DEFAULT_SYSTEM_PROMPT ="""You are an expert Data Scientist. Efficiently analyze the given confusion matrix and extract key performance metrics, including accuracy, precision, recall, and F1-score. Provide concise insights into the model's performance without offering explanations or justifications.\n"""
        
        SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS
        
        prompt = f"""Please analyze the provided confusion matrix{df} and generate insights based on the classification results. Include key metrics such as accuracy, precision, recall, and F1-score. Additionally, offer a concise interpretation of the model's performance in predicting the classes.\n"""
        
        generated_text = self.generate(prompt, SYSTEM_PROMPT, 500)
        
#         print(generated_text)
        # Assuming that self.generate returns the generated text
        return generated_text

        

        

        
        

        
            

    
    
    


