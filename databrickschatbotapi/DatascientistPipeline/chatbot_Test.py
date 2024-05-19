import os
import shutil
import torch
import pandas as pd
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
        self.model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
        
#         self.DEFAULT_SYSTEM_PROMPT = """
#         As a knowledgeable assistant, I aim to generate questions based on provided contexts. My goal is to help you explore various aspects of the data and facilitate meaningful discussions. 
#         If any question seems irrelevant or unclear, I'll provide an explanation instead of generating a question. 
#         """.strip()
        
#         self.SYSTEM_PROMPT = """ Use the following pieces of context generate Questions from Table Relationship in CSV File.

#         Relationship related to comprehensive business strategy to optimize production levels, ensuring efficiency, cost-effectiveness, and quality. Consider market demand, competition, cost analysis, supply chain management, technology adoption, and sustainability.
#         Identify areas for improvement and outline actionable steps to enhance production performance

#         If you don't know the answer, just say that you don't know, don't try to make useless Question.
#         """

        self.template = self.generate_prompt("{context}\nQuestion: {question}")
        self.prompt = PromptTemplate(template=self.template, input_variables=["context", "question"])

    def load_data_from_csv(self):
        if os.path.exists(self.csv_file):
            loader = CSVLoader(file_path=self.csv_file)
            data = loader.load()
        else:
            raise FileNotFoundError(f"CSV file not found at {self.csv_file}")
        return data
    
    def remove_db_directory(self):
        if os.path.exists("db"):
            shutil.rmtree("db")

    def setup_pipeline(self, data_csv):
#         self.remove_db_directory()
        embeddings = HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-xl", model_kwargs={"device": self.device}
        )
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
        texts = text_splitter.split_documents(data_csv)
        db = Chroma.from_documents(texts, embeddings,)
        return db

    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True, device=self.device)
        model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, device_map='auto', torch_dtype=torch.float16)
        return tokenizer, model
    
    def generate_prompt(self, prompt: str, system_prompt: str = None) -> str:
#         default_system_prompt = self.DEFAULT_SYSTEM_PROMPT
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

    def generate_questions(self,query):
        data_csv = self.load_data_from_csv()
        db = self.setup_pipeline(data_csv)
        tokenizer, model = self.load_model()

        template = self.generate_prompt(""" {context} Question: {question}""")
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        text_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
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

        prompt = query

        result = qa_chain(prompt)
        return result
