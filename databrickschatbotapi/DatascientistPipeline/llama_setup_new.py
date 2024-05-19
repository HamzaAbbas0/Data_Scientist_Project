from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, pipeline
from langchain import HuggingFacePipeline
import torch

class LlamaLanguageModel:
    def __init__(self):
        self.qa_chain = None
        self.db = None
        self.template = self.generate_prompt("{context}\nQuestion: {question}")
        self.prompt = PromptTemplate(template=self.template, input_variables=["context", "question"])
        
        model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, auth_token="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")
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='auto', torch_dtype=torch.float16)
        self.llm = self.setup_language_model()
        

    def setup_language_model(self):
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        text_pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, max_new_tokens=1024, temperature=0.5, top_p=0.95, repetition_penalty=1.15, streamer=streamer)
        llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0.6})
        return llm
    
    
    
    def load_data_from_csv(self,file_path):
        data = []
        if os.path.exists(file_path):
            loader = CSVLoader(file_path=file_path)
            data = loader.load()
        else:
            print(f"Warning: File not found at {file_path}")
        return data

    def load_summary_statistics(self, file_path):
        self.data.extend(self.load_data_from_csv(file_path))


    def setup_qa_chain(self):
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.db.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt},
        )

    def apply_rag(self, question):
        result = self.qa_chain(question)
#         return result

    def generate_prompt(self, prompt: str, system_prompt: str = None) -> str:
        system_prompt="""
        As a seasoned Data Scientist, your role is to provide a clear and concise summery statistics of the dataset 
        based on user prompts,  ensuring a focus on relevant insights. 
        Please don't provide the false information.
        """
        return f"""
        [INST] <<SYS>>
        {system_prompt}
        <</SYS>>
        
        {prompt} [/INST]
        """.strip()

    def setup_chroma_db(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
        texts = text_splitter.split_documents(self.data)
        self.db = Chroma.from_documents(texts, HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={"device": "cuda"}))

