import streamlit as st
import google.generativeai as ai

# Set up the page configuration
st.set_page_config(page_title="Chatbot UI", layout="wide")

# Sidebar for user API key input and conversation selection
st.sidebar.title("Chatbot Settings")
api_key = st.sidebar.text_input("Enter your API Key", type="password", help="Input your Google Generative AI API key.")

# Store session state for chat history and conversations
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "current_conversation" not in st.session_state:
    st.session_state.current_conversation = None

# Function to start a new conversation
def start_new_conversation():
    conversation_id = f"Conversation {len(st.session_state.conversations) + 1}"
    st.session_state.conversations[conversation_id] = []
    st.session_state.current_conversation = conversation_id

# Sidebar for conversation management
st.sidebar.title("Conversations")
if st.sidebar.button("New Conversation"):
    start_new_conversation()

# Display the current conversation history
st.title("Chatbot Conversation")
if st.session_state.current_conversation:
    for message in st.session_state.conversations[st.session_state.current_conversation]:
        st.write(message)
else:
    st.write("No conversation selected. Start a new conversation or select an existing one.")

# Input area for user messages
message = st.text_input("You:", key="message_input", placeholder="Type your message here...")
if st.button("Send") and message:
    if api_key:
        ai.configure(api_key=api_key)
        model = ai.GenerativeModel("gemini-pro")
        chat = model.start_chat()
        response = chat.send_message(message)
        user_message = f"You: {message}"
        bot_response = f"Chatbot: {response.text}"
        st.session_state.conversations[st.session_state.current_conversation].append(user_message)
        st.session_state.conversations[st.session_state.current_conversation].append(bot_response)
        st.experimental_rerun()
    else:
        st.error("Please enter a valid API key.")

# Display existing conversations in the sidebar
st.sidebar.title("Existing Conversations")
for conversation_id in st.session_state.conversations.keys():
    if st.sidebar.button(conversation_id):
        st.session_state.current_conversation = conversation_id
        st.experimental_rerun()
