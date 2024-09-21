import streamlit as st
import ollama
import sys
import logging  # {{ edit_1 }}

class LlamaModel:
    def __init__(self, model_name="llama2", temperature=0.5, max_tokens=1000):
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
    
@st.cache_resource
def get_llama_model(model_name, temperature, max_tokens):
    return LlamaModel(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
    
def generate_response(llama, user_input):
    try:
        return llama.get_response(user_input)
    except Exception as e:
        logging.error(f"Error generating response: {e}")  # Log the error
        st.error("Sorry, I couldn't generate a response.")  # Keep user-facing error message
        return "Sorry, I couldn't generate a response."

# {{ edit_2 }}
logging.basicConfig(filename='llamachat.log', filemode='w', level=logging.INFO)  # Log to file in write mode

def main():

    # Sidebar for model configuration
    st.sidebar.header("Llama Chat")
    model_name = st.sidebar.selectbox("Select Model", ["llama3.1"], index=0)
    temperature = st.sidebar.slider("Temperature", 0.1, 1.0, 0.5, 0.1)
    max_tokens = st.sidebar.number_input("Max Tokens", min_value=1, max_value=128000, value=2000, step=100)

    # Add button to clear chat history in the sidebar
    if st.sidebar.button("Clear History"):
        st.session_state.chat_history = []

    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Initialize LlamaModel
    llama = get_llama_model(model_name, temperature, max_tokens)

    # Chat interface
    user_input = st.text_input("You:", key="user_input", value="")  # Clear input field after sending

    # Automatically send message when user input is provided
    if user_input:
        st.session_state.chat_history.append(("**You**", user_input))
        with st.spinner('Generating response...'):  # Spinner moved outside the function
            response = generate_response(llama, user_input)  # Updated to use the new function
        st.session_state.chat_history.append(("**Llama**", response))

    # Display chat history
    for role, message in st.session_state.chat_history:
        st.write(f"{role}: {message}")

if __name__ == "__main__":
    main()
