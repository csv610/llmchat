# sl_metallama3_2.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

from huggingface_hub import login

login(token=os.getenv('HUGGINGFACE_API_KEY'))

class LlamaModel:  # Renamed class
    def __init__(self, model_id="meta-llama/Meta-Llama-3-8B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        self.messages = [
            {"role": "system", "content": "You are a helpful and knowledgeable assistant. Your goal is to provide accurate information, answer questions, and engage in friendly conversation on a wide range of topics."}
        ]

    def generate_response(self, user_message, max_new_tokens, temperature):
        self.messages.append({"role": "user", "content": user_message})  # Add user message here

        input_ids = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)

# Streamlit app
def main():
    st.title("Llama Chat")
    llama_model = LlamaModel()  # Updated instantiation

    user_message = st.text_input("You:")
    max_new_tokens = st.number_input("Max New Tokens:", min_value=1, max_value=128000, value=1000)
    temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.6)

    if st.button("Send"):
        if user_message:
            response = llama_model.generate_response(user_message, max_new_tokens, temperature)
            st.text_area("Pirate Chatbot:", value=response, height=200)
        else:
            st.warning("Please enter a message.")

if __name__ == "__main__":
    main()
