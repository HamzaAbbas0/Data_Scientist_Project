import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch

class CodeGeneration:
    def __init__(self, query, data):
        self.question = query
        self.data = data
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    
    def load_model(self):
        # Specify the GPU device you want to use (assuming GPU index 2 in this example)
#         gpu_device = 2

#         # Set the CUDA device
#         torch.cuda.set_device(gpu_device)

        # Move model to CUDA device
        self.model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct", trust_remote_code=True)

        # Move tokenizer to CUDA device (if necessary)
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct", trust_remote_code=True)

    
    def save_function(self, response):
        if "```python" in response:
            start = response.split("```python")
            code_snippet = start[1].split('```')
            code_snippet = code_snippet[0]

            # Save to a text file with the function name
            with open("generated_code.py", "w") as file:
                file.write(code_snippet)

            print("Code snippet saved to 'generated_code.py'")
            return code_snippet 
            
            
    def code_inference(self):

        content = f"""

        Given the question: '{self.question}' and data: '{self.data[6:10]}'. Define the following to answer this question.
        def analysis(data)

        """

        messages = [
            {'role': 'user', 'content': content}
        ]

        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        # 32021 is the id of the token
        outputs = self.model.generate(inputs, max_new_tokens=512, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=32021)
        print(self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))
        response = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        return response
               
            
    def code_verification(self, code_content):

        content = f"""
        Consider the question: "{self.question}".

        Your task is to verify if the following Python code is logically correct. If there is any problem in the code. Respond in Yes or No. If yes then provide the corrected version of same function.

        Here are the first few samples of the 'data' dataframe:

        {self.data[6:10]}

        Review the code snippet provided and assess.

        Code Snippet:
        {code_content}

        """


        messages = [
            {'role': 'user', 'content': content}
        ]

        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
        # 32021 is the id of the token
        outputs = self.model.generate(inputs, max_new_tokens=512, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=32021)
        response = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        print(response)
        if all(keyword not in response.split('\n')[0] for keyword in ["Yes", "correct", "correctly"]):
            self.run_process()
            
    
    def handle_error(self):
        content = f"""
        Consider the question: "{self.question}".

        Given the context, it appears that the most relevant task is time-series analysis.

        Your task is to verify if the following Python code correctly performs {title} on the dataframe 'df' to address the stated question.
        If there is any problem in the code. Respond in Yes or No. If yes then provide the corrected version of same function.

        Here are the first few samples of the 'df' dataframe:

        {self.data[6:10]}

        Review the code snippet provided and assess.

        Code Snippet:
        {code_content}

        """


        messages = [
            {'role': 'user', 'content': content}
        ]

        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
        # 32021 is the id of the token
        outputs = self.model.generate(inputs, max_new_tokens=512, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=32021)
        response = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        print(response)
        if all(keyword not in response.split('\n')[0] for keyword in ["Yes", "correct", "correctly"]):
            self.run_process()
            
            
    def run_generated_code(self):
        from generated_code import analysis

        try:
            analysis(self.data)
            self.code_verification(analysis(self.data))
        except Exception as e:
            code_snippet = handle_error()
            self.run_generated_code()
                
    def run_process(self):
        response = self.code_inference()
        code_snippet = self.save_function(response)
        self.run_generated_code()
        
        
## Usage
# from CodeGeneration import CodeGeneration

# code_generation = CodeGeneration(query, data)
# code_generation.run_process()