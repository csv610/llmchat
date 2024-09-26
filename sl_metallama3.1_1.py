import streamlit as st
import transformers
import torch

class LlamaModel:
    def __init__(self):
        self.model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

    def generate_response(self, user_input, max_new_tokens, temperature):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input},
        ]

        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.pipeline(
            messages,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
        )
        return outputs[0]["generated_text"][-1]

# Streamlit interface
st.title("Pirate Chatbot")
chatbot = LlamaModel()
user_input = st.text_input("Ask the pirate chatbot:")
max_new_tokens = st.number_input("Max New Tokens:", min_value=1, value=256)
temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.6)

if user_input:
    response = chatbot.generate_response(user_input, max_new_tokens, temperature)
    st.write(response)
