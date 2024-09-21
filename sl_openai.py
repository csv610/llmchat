import PyPDF2
import pdfplumber
import base64
from PIL import Image
from io import BytesIO
import streamlit as st
from openai import OpenAI
import os

# Helper function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Helper function to encode images as base64
def encode_image_to_base64(image_file):
    image = Image.open(image_file)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Function to interact with OpenAI's ChatCompletion API
def ask_llm(question, model="gpt-4", temperature=0.5, max_tokens=1000):
    client = OpenAI()  # This will use the OPENAI_API_KEY environment variable
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Streamlit UI
st.title("OpenAI LLM")

# Check for API key in environment variable
if 'OPENAI_API_KEY' not in os.environ:
    st.error("OPENAI_API_KEY not found in environment variables. Please set it before running the application.")
    st.stop()

# Sidebar with model settings and history
st.sidebar.header("Model Settings")
model_choice = st.sidebar.selectbox("Choose the Model", options=["gpt-4", "gpt-4o", "gpt-4o-mini"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5, help="Control the randomness of the model")
max_tokens = st.sidebar.number_input("Max Tokens", min_value=10, max_value=128000, value=1000, step=1000, help="Limit the length of the response")

# Response History in Sidebar
st.sidebar.header("Response History")

# Initialize session state for history and visibility
if "history" not in st.session_state:
    st.session_state.history = []

if "show_history" not in st.session_state:
    st.session_state.show_history = True

# History management buttons
col1, col2 = st.sidebar.columns(2)
if col1.button("Show History"):
    st.session_state.show_history = True
if col2.button("Hide History"):
    st.session_state.show_history = False

col3, col4 = st.sidebar.columns(2)
if col3.button("Clear History"):
    st.session_state.history = []
if col4.button("Delete History"):
    st.session_state.history = []
    st.session_state.show_history = False

# Display history if show_history is True
if st.session_state.show_history and len(st.session_state.history) > 0:
    for i, entry in enumerate(st.session_state.history[::-1]):
        st.sidebar.subheader(f"Interaction {len(st.session_state.history) - i}")
        st.sidebar.write(f"**Q**: {entry['question'][:50]}...")
        st.sidebar.write(f"**A**: {entry['response'][:50]}...")
        if st.sidebar.button(f"Show full interaction {len(st.session_state.history) - i}"):
            st.sidebar.text_area("Full Question", entry['question'], height=100)
            st.sidebar.text_area("Full Answer", entry['response'], height=200)
elif st.session_state.show_history:
    st.sidebar.write("No history available yet.")

# File upload (PDF, Images)
uploaded_file = st.file_uploader("Upload a PDF or an Image", type=["pdf", "png", "jpg", "jpeg"])

# Input text for asking questions
st.header("Ask a Question")
input_text = st.text_area("Enter your question:", value="", height=100, help="Type the question you want to ask the model")

# Button to send the query
if st.button("Submit"):
    if input_text.strip() == "" and not uploaded_file:
        st.warning("Please enter a question or upload a file.")
    else:
        # If a PDF is uploaded, extract text
        if uploaded_file is not None:
            file_type = uploaded_file.type
            if file_type == "application/pdf":
                pdf_text = extract_text_from_pdf(uploaded_file)
                input_text += "\n\nExtracted Text from PDF:\n" + pdf_text
            elif file_type in ["image/png", "image/jpeg"]:
                image_base64 = encode_image_to_base64(uploaded_file)
                input_text += f"\n\nUploaded Image (base64): {image_base64}"

        # Get the response from the LLM
        response = ask_llm(input_text, model=model_choice, temperature=temperature, max_tokens=max_tokens)
        
        # Display the response
        st.subheader("Model's Response")
        st.write(response)

        # Store the query and response in session state for history
        st.session_state.history.append({"question": input_text, "response": response})

# Export option
if st.sidebar.button("Export Conversation"):
    if len(st.session_state.history) > 0:
        history_str = "\n\n".join([f"Q: {h['question']}\nA: {h['response']}" for h in st.session_state.history])
        b64 = base64.b64encode(history_str.encode()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="conversation.txt">Download Conversation</a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)
    else:
        st.sidebar.write("No conversation history to export.")
