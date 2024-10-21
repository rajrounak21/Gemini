import streamlit as st
from key import GOOGLE_API_KEY
import google.generativeai as genai
import os
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage

# Set the API key for Google Gemini
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to load Gemini Pro model and get response
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

# Initialize ConversationBufferMemory to store chat history
if 'memory' not in st.session_state:
    st.session_state['memory'] = ConversationBufferMemory()


# Function to get full history as string for context
def get_full_conversation(memory):
    conversation_history = ""
    for message in memory.chat_memory.messages:
        if isinstance(message, HumanMessage):
            conversation_history += f"You: {message.content}\n"
        elif isinstance(message, AIMessage):
            conversation_history += f"Bot: {message.content}\n"
    return conversation_history


# Function to get Gemini model's response and use memory for context
def get_gemini_response(question, memory):
    # Get full conversation history to pass as context
    full_conversation = get_full_conversation(memory)

    # Update the memory with the current question
    memory.chat_memory.add_message(HumanMessage(content=question))

    # Pass the conversation history as part of the input
    response = chat.send_message(f"{full_conversation}You: {question}", stream=True)

    # Collect the full response text
    response_text = "".join(chunk.text for chunk in response)

    # Update the memory with the bot's response
    memory.chat_memory.add_message(AIMessage(content=response_text))

    return response_text


# Streamlit page configuration
st.set_page_config(page_title="Q&A CHATBOT")

st.header("CHATBOT FOR USERS QUERY QUESTION")

# Input field for the user
user_input = st.text_input("Input:", key="input")
submit = st.button("Ask the Question")

# If the user submits a question
if submit and user_input:
    # Get the model's response using conversation buffer memory
    response = get_gemini_response(user_input, st.session_state['memory'])

    # Append the user's input and response to the chat history
    st.session_state['memory'].chat_memory.add_message(HumanMessage(content=user_input))
    st.session_state['memory'].chat_memory.add_message(AIMessage(content=response))

    # Display the response
    st.subheader("The Response is")
    st.write(response)

# Display the full chat history
st.subheader("The Chat History is")
for message in st.session_state['memory'].chat_memory.messages:
    if isinstance(message, HumanMessage):
        st.write(f"You: {message.content}")
    elif isinstance(message, AIMessage):
        st.write(f"Bot: {message.content}")