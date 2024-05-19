from huggingface_hub import login
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import textwrap
import os
#####################
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

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, pipeline
import torch
# views.py
from databrickschatbotapi.model_loader import llm_model
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
class Chatbot:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         llm_manager = LLMManager()
        self.model = llm_model
#        self.tokenizer = llm_model
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=True, device=self.device)
#        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map='auto', torch_dtype=torch.float16)
        ###########################llama file work################################
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
        print(" query a value", question)
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
################################################################################################################

    def load_data_from_csv(self,csv_file):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found at {csv_file}")
        loader = CSVLoader(file_path=csv_file)
        return loader.load()

#     def remove_db_directory(self):
#         db_path = "db"
#         if os.path.exists(db_path):
#             shutil.rmtree(db_path)

    def setup_pipeline(self, data_csv):
        embeddings = HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-xl", model_kwargs={"device": self.device}
        )
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
        texts = text_splitter.split_documents(data_csv)
        return Chroma.from_documents(texts, embeddings)
    
    def generate_prompt(self, prompt: str, system_prompt: str) -> str:
        return f"""
        [INST] <<SYS>>
        {system_prompt}
        <</SYS>>

        {prompt} [/INST]
        """.strip()
    
    def gen_exceptionchat(self,query): 
        DEFAULT_SYSTEM_PROMPT = f"""\
        You are an AI assistant. A user has encountered an error while have query to chatbot this is the error {query}. analyze error. Your task is to generate a friendly, non-technical message explaining the issue and providing simple steps the user can take to resolve it. Use clear and supportive language."."""

        SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS


        prompt = f"{DEFAULT_SYSTEM_PROMPT}\nQuestion: '{query}'"

        generated_text = self.generate(prompt,SYSTEM_PROMPT,200)
        response = self.parse_text(generated_text)

        return response    
    def gen_chat(self,query): 
        DEFAULT_SYSTEM_PROMPT = """\
        You are an AI trained to respond directly and concisely to any questions. Provide the information in a structured and easily readable format. The format should include a clear header, followed by the content in bullet points or sections. If you do not know the answer, respond with: "I do not have enough information to answer that question."""

        SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS


        prompt = f"{DEFAULT_SYSTEM_PROMPT}\nQuestion: '{query}'"

        generated_text = self.generate(prompt,SYSTEM_PROMPT,200)
        response = self.parse_text(generated_text)

        return response
    
    ##############################################################################################################
    
        

    def generate_summary(self,csv_file,query):
        system_prompt = f"""
        As a seasoned Data Scientist, your role is to provide a clear and concise summary statistics of the dataset 
        based on user prompts in file, ensuring a focus on relevant insights. 
        Please don't provide the false information.
        """
        data_csv = self.load_data_from_csv(csv_file)
        db = self.setup_pipeline(data_csv)

        template = self.generate_prompt(""" You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.
    
    Context: {context}
    User: {question}
    Chatbot:""",system_prompt)
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        text_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=1024,
            temperature=0.5,
            top_p=0.95,
            repetition_penalty=1.15,
            streamer=streamer,
        )

        llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )

        # Execute a query (you can customize this part as needed)
        query_result = qa_chain(query)

                # Execute the query
#         query_result, source_documents = qa_chain.run(query)

        #print("llm!!!!!!!!!!!!!!!!!!!!! Result",query_result)

        # Extract the answer from the 'response' field using string manipulation
        # Assuming the answer always follows "Answer:" and ends at the end of the string
        response_content = query_result['result']
        answer_prefix = "Chatbot: "
        answer_start_index = response_content.find(answer_prefix)
        if answer_start_index != -1:
            answer = response_content[answer_start_index + len(answer_prefix):].strip()
            print(answer)
            return answer
        else:
            print("No answer found in the response.")
            return response_content

    
    
    
    def imp_features(self,csv_file,query):
        system_prompt = """\
        You are an expert Data Scientist, you efficiently provide the top most important features of the provided data.
        If you don't know the answer to a question, please don't share false information."""
        
        data_csv = self.load_data_from_csv(csv_file)
        db = self.setup_pipeline(data_csv)

        template = self.generate_prompt(""" You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.
    
    Context: {context}
    User: {question}
    Chatbot:""",system_prompt)
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        text_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=1024,
            temperature=0.5,
            top_p=0.95,
            repetition_penalty=1.15,
            streamer=streamer,
        )

        llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )

        # Execute a query (you can customize this part as needed)
        query_result = qa_chain(query)

                # Execute the query
#         query_result, source_documents = qa_chain.run(query)

        #print("llm!!!!!!!!!!!!!!!!!!!!! Result",query_result)

        # Extract the answer from the 'response' field using string manipulation
        # Assuming the answer always follows "Answer:" and ends at the end of the string
        response_content = query_result['result']
        answer_prefix = "Chatbot: "
        answer_start_index = response_content.find(answer_prefix)
        if answer_start_index != -1:
            answer = response_content[answer_start_index + len(answer_prefix):].strip()
            print(answer)
            return answer
        else:
            print("No answer found in the response.")
            return response_content

    ######################################
    

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
    

    def read_pdf(self, file_path):
        loader = PyPDFLoader(file_path)
        data = loader.load()
            
        return data

    def read_docx(self, file_path):
        loader = Docx2txtLoader(file_path)
        data = loader.load()
            
        return data

    def read_txt(self, file_path):
        loader = TextLoader(file_path)
        data = loader.load()
            
        return data

    def generate_response_from_document(self, file_path, query):
        """Generate a response based on a user query from a document."""
        # Determine the file type and extract text
        if file_path.lower().endswith('.pdf'):
            document_text = self.read_pdf(file_path)
        elif file_path.lower().endswith('.docx'):
            document_text = self.read_docx(file_path)
        elif file_path.lower().endswith('.txt'):
            document_text = self.read_txt(file_path)
        else:
            raise ValueError("Unsupported file type. Supported types are: 'pdf', 'docx', 'txt'.")
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        text_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=1024,
            temperature=0.5,
            top_p=0.95,
            repetition_penalty=1.15,
            streamer=streamer,
        )

        llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0.5})

        # Step 2: Split the text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
        texts = text_splitter.split_documents(document_text)

        # Step 3: Generate embeddings for the texts
        embeddings = HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-xl", model_kwargs={"device": "cuda"}
        )
        db = Chroma.from_documents(texts, embeddings)


#         SYSTEM_PROMPT = """
#         You are a knowledgeable and precise assistant designed to handle document-related queries efficiently and safely. Always respond as accurately as possible, ensuring your answers comply with ethical guidelines and legal standards. Avoid sharing any information that could be harmful, unethical, biased, or illegal.

#         If a query is ambiguous or unclear, ask clarifying questions to better understand the user's needs before providing a response. If the system cannot retrieve the requested information due to format limitations or content availability, inform the user clearly and suggest possible alternatives.

#         Ensure that your responses enhance user interaction with the system, guiding them on how to upload, retrieve, and manage their documents effectively. If you encounter a question outside your expertise or capability, advise the user accordingly without resorting to misleading or incorrect information.
#         """
        SYSTEM_PROMPT = """
        You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. give Answer with Answer:
        """

        prompt = PromptTemplate(template=""" You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.
    
    Context: {context}
    User: {question}
    Chatbot:""", input_variables=["context", "question"])

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )

        # Execute a query (you can customize this part as needed)
        query_result = qa_chain(query)

                # Execute the query
#         query_result, source_documents = qa_chain.run(query)

        #print("llm!!!!!!!!!!!!!!!!!!!!! Result",query_result)

        # Extract the answer from the 'response' field using string manipulation
        # Assuming the answer always follows "Answer:" and ends at the end of the string
        response_content = query_result['result']
        answer_prefix = "Chatbot: "
        answer_start_index = response_content.find(answer_prefix)
        if answer_start_index != -1:
            answer = response_content[answer_start_index + len(answer_prefix):].strip()
            print(answer)
            return answer
        else:
            print("No answer found in the response.")
            return response_content

    
    

    



