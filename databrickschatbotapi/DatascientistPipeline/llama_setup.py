from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, pipeline
from langchain import HuggingFacePipeline
from langchain_community.document_loaders.csv_loader import CSVLoader
import torch
import os
class LlamaLanguageModel:
    def __init__(self):
        self.llm = self.setup_language_model()

    def setup_language_model(self):
        model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, auth_token="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")
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='auto', torch_dtype=torch.float16)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        text_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024, temperature=0.5, top_p=0.95, repetition_penalty=1.15, streamer=streamer)
        llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0.6})
        return llm
