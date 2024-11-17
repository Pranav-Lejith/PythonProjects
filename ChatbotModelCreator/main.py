import streamlit as st
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import tempfile

# Title
st.title("AI Chatbot Model Creator")

# Sidebar options
st.sidebar.header("Chatbot Model Settings")
chatbot_type = st.sidebar.selectbox("Choose chatbot type", ["Default Chatbot", "Custom Chatbot"])

# Load Blenderbot model and tokenizer
MODEL_NAME = 'facebook/blenderbot-400M-distill'
tokenizer = BlenderbotTokenizer.from_pretrained(MODEL_NAME)
model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_NAME)

# Test chatbot section
st.subheader("Test your Chatbot")
user_input = st.text_input("Ask something")
if user_input:
    inputs = tokenizer(user_input, return_tensors="pt")
    reply_ids = model.generate(**inputs)
    bot_response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    st.write(f"Chatbot response: {bot_response}")

# Function to generate chatbot usage code
def generate_usage_code():
    usage_code = f"""
# Usage code for your Chatbot Model
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Load your trained chatbot model
MODEL_NAME = '{MODEL_NAME}'
tokenizer = BlenderbotTokenizer.from_pretrained(MODEL_NAME)
model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_NAME)

# Function to chat with the bot
def chat_with_bot(user_input):
    inputs = tokenizer(user_input, return_tensors="pt")
    reply_ids = model.generate(**inputs)
    response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return response

# Example usage
user_input = "Hello!"
response = chat_with_bot(user_input)
print(f"Chatbot response: {{response}}")
"""
    return usage_code

# Download button for chatbot model and usage code
if st.button("Download Model & Usage Code"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_file:
        temp_file.write(generate_usage_code().encode('utf-8'))
        temp_file_path = temp_file.name
    
    st.write("Your chatbot model is ready for download!")
    
    st.download_button(
        label="Download Usage Code",
        data=open(temp_file_path, "r").read(),
        file_name="chatbot_usage.py",
        mime="text/x-python"
    )
