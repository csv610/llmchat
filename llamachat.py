
import ollama
import sys

class LlamaModel:
    def __init__(self, model_name="llama3.1"):
        self.model_name = model_name
        self.messages = []

    def get_response(self, user_input):
        # Corrected the variable name from 'content' to 'user_input'
        prompt = [{'role': 'user', 'content': user_input}]
        
        # Call the Ollama chat function and get the response
        response = ollama.chat(model=self.model_name, messages=prompt)
        return response['message']['content']

if __name__ == "__main__":
    llama = LlamaModel()

    question = sys.argv[1]
    
    response = llama.get_response(question)
    print(response)
