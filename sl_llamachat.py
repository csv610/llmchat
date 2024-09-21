import streamlit as st
import ollama
import sys

@st.cache_resource
def get_llama_model(model_name, temperature, max_tokens):
    return LlamaModel(model_name=model_name, temperature=temperature, max_tokens=max_tokens)

class LlamaModel:
    def __init__(self, model_name="llama2", temperature=0.7, max_tokens=None):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.messages = []

    def get_response(self, user_input):
        prompt = [{'role': 'user', 'content': user_input}]
        response = ollama.chat(
            model=self.model_name, 
            messages=prompt,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        )
        return response['message']['content']

def main():
    st.title("LlamaModel Chat Interface")

    # Sidebar for model configuration
    st.sidebar.header("Model Configuration")
    model_name = st.sidebar.selectbox("Select Model", ["llama3.1"], index=0)
    temperature = st.sidebar.slider("Temperature", 0.1, 1.0, 0.5, 0.1)
    max_tokens = st.sidebar.number_input("Max Tokens", min_value=1, max_value=128000, value=2000, step=100)

    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Initialize LlamaModel
    llama = get_llama_model(model_name, temperature, max_tokens)

    # Chat interface
    user_input = st.text_input("You:", key="user_input")

    if st.button("Send"):
        if user_input:
            st.session_state.chat_history.append(("You", user_input))
            with st.spinner('Generating response...'):
                response = llama.get_response(user_input)
            st.session_state.chat_history.append(("llama", response))
            user_input = ""  # Reset user input

    # Add button to clear chat history
    if st.button("Clear History"):
        st.session_state.chat_history = []

    # Display chat history
    for role, message in st.session_state.chat_history:
        st.write(f"{role}: {message}")
        

if __name__ == "__main__":
    main()
