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

class Chatbot:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=True, device=self.device)
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map='auto', torch_dtype=torch.float16)

    def load_data_from_csv(self):
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"CSV file not found at {self.csv_file}")
        loader = CSVLoader(file_path=self.csv_file)
        return loader.load()

    def remove_db_directory(self):
        db_path = "db"
        if os.path.exists(db_path):
            shutil.rmtree(db_path)

    def setup_pipeline(self, data_csv):
        self.remove_db_directory()
        embeddings = HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-xl", model_kwargs={"device": self.device}
        )
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
        texts = text_splitter.split_documents(data_csv)
        return Chroma.from_documents(texts, embeddings, persist_directory="db")

#     def load_model(self):
#         tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True, device=self.device)
#         model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, device_map='auto', torch_dtype=torch.float16)
#         return tokenizer, model
    
    def generate_prompt(self, prompt: str, system_prompt: str) -> str:
        system_prompt = """
        As a seasoned Data Scientist, your role is to provide a clear and concise summary statistics of the dataset 
        based on user prompts, ensuring a focus on relevant insights. 
        Please don't provide the false information.
        """
        return f"""
        [INST] <<SYS>>
        {system_prompt}
        <</SYS>>

        {prompt} [/INST]
        """.strip()

    def generate_summary_statistics_prompt(self, prompt: str, system_prompt: str) -> str:
        system_prompt = """
        write me the intoduction of that file don't tell the false information
        """
        return f"""
        [INST] <<SYS>>
        {system_prompt}
        <</SYS>>

        {prompt} [/INST]
        """.strip()


    def generate_questions(self, query):
        system_prompt = """
        As a seasoned Data Scientist, your role is to provide a clear and concise summary statistics of the dataset 
        based on user prompts, ensuring a focus on relevant insights. 
        Please don't provide the false information.
        """
        correlation_prompt = self.generate_correlation_prompt()
        data_csv = self.load_data_from_csv()
        db = self.setup_pipeline(data_csv)

        template = self.generate_prompt(""" {context} Question: {question}""",system_prompt)
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

        result = qa_chain(query)
        return result
    
    
    def generate_summary(self, query):
        system_prompt = """
        write me the intoduction of that file don't tell the false information
        """
        data_csv = self.load_data_from_csv()
        db = self.setup_pipeline(data_csv)

        template = self.generate_summary_statistics_prompt(""" {context} Question: {question}""",system_prompt)
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

        result = qa_chain(query)
        return result



