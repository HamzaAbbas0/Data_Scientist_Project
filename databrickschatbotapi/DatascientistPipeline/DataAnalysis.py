from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class DataAnalysis:
    def __init__(self, model_name="deepseek-ai/deepseek-coder-1.3b-instruct", gpu_device=2):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).cuda(device=gpu_device)

    def extact_and_save_function(self, response, filename="data_analysis.py"):
        if "```python" in response:
            start = response.split("```python")
            code_snippet = start[1].split('```')[0]

            # Save to a text file with the function name
            with open(filename, "w") as file:
                file.write(code_snippet)

            print(f"Code snippet saved to '{filename}'")
            return code_snippet

    def code_verification(self, question, code_content, df):
        content = f"""
        Consider the question: "{question}".

        Your task is to verify if the following Python function correctly performs analysis on the dataframe 'df' to address the stated question.
        
        If there is any problem in the code, provide the corrected version of same function.

        Review the code snippet provided and assess.

        Code Snippet:
        {code_content}
        
        For more context, here are the first few samples of the 'df' dataframe:

        {df[6:10]}

        """

        messages = [{'role': 'user', 'content': content}]

        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.model.device)
        # 32021 is the id of the token
        outputs = self.model.generate(inputs, max_new_tokens=512, do_sample=True, top_k=50, top_p=0.95,
                                      num_return_sequences=1, eos_token_id=32021)
        response = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        print(response)
        if all(keyword not in response.split('\n')[0] for keyword in ["Yes", "correct", "correctly"]):
            self.data_clean_process(question, df, "data_cleaning")

    def data_analysis_inference(self, question, df):
        content = f"""

        Consider the question: "{question}".

        Generate python function 'analysis' to answer the questions. The code may include graphs, charts or any visualization etc.

        \n\nHere are first few samples of cleaned dataframe:
        {df[6:10]}

        """

        messages = [{'role': 'user', 'content': content}]

        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.model.device)
        # 32021 is the id of the token
        outputs = self.model.generate(inputs, max_new_tokens=512, do_sample=True, top_k=50, top_p=0.95,
                                      num_return_sequences=1, eos_token_id=32021)
        print(self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))
        response = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        return response

    def run_data_analysis_process(self, question, df, code_snippet):
        from data_analysis import analysis

        try:
            analysis(df)
            self.code_verification(question, code_snippet, df)
        except Exception as e:
            code_snippet = self.handle_error(question, df, e, code_snippet)
            self.run_data_analysis_or_modeling(question, df, code_snippet)

    def data_analysis(self, question, df):
        response = self.data_analysis_inference(question, df)
        code_snippet = self.extact_and_save_function(response)
        self.run_data_analysis_process(question, df, code_snippet)

        
