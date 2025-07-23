import streamlit as st
from key import GOOGLE_API_KEY
import google.generativeai as genai
import os
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage

# Set Gemini API Key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Gemini model and chat
model = genai.GenerativeModel("gemini-2.0-pro")  # You can also try "gemini-1.5-flash"
chat = model.start_chat(history=[])

# Setup session memory
if 'memory' not in st.session_state:
    st.session_state['memory'] = ConversationBufferMemory()

# Function to build Gemini-compatible message list from memory
def get_message_history(memory):
    messages = []
    for message in memory.chat_memory.messages:
        if isinstance(message, HumanMessage):
            messages.append({"role": "user", "parts": [message.content]})
        elif isinstance(message, AIMessage):
            messages.append({"role": "model", "parts": [message.content]})
    return messages

# Function to get Gemini response with context
def get_gemini_response(question, memory):
    # Get chat history as Gemini-style parts
    history = get_message_history(memory)

    # Add current user question
    history.append({"role": "user", "parts": [question]})

    # Send to Gemini
    response = chat.send_message(history)

    # Extract text
    response_text = response.text

    # Save in memory
    memory.chat_memory.add_message(HumanMessage(content=question))
    memory.chat_memory.add_message(AIMessage(content=response_text))

    return response_text

# Streamlit UI
st.set_page_config(page_title="Q&A Chatbot")
st.header("🤖 Chatbot for User Queries")

user_input = st.text_input("Your Question:", key="input")
submit = st.button("Ask")

if submit and user_input:
    response = get_gemini_response(user_input, st.session_state['memory'])

    # Show response
    st.subheader("💬 Bot's Answer")
    st.write(response)

# Display chat history
st.subheader("📜 Chat History")
for msg in st.session_state['memory'].chat_memory.messages:
    if isinstance(msg, HumanMessage):
        st.markdown(f"**You**: {msg.content}")
    elif isinstance(msg, AIMessage):
        st.markdown(f"**Bot**: {msg.content}")
