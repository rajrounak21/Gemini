# Q&A Chatbot with Google Gemini and Streamlit
import streamlit as st
from PIL import Image
import google.generativeai as genai

# Set the API key for Google Gemini
from key import GOOGLE_API_KEY
import os
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-1.5-flash')

# Function to get Gemini response for both text and image
def get_gemini_response(input_text, image=None):
    if image is not None:
        # Pass the image as a PIL Image object directly to Gemini
        response = model.generate_content([input_text, image])
    else:
        # Generate response using only text
        response = model.generate_content(input_text)

    return response.text

# Streamlit app layout
st.set_page_config(page_title="Gemini Image Q&A Chatbot")
st.header("Gemini Application")

# Text input
input_text = st.text_input("Enter your question:", key="text_input")

# File uploader for image
uploaded_file = st.sidebar.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

# Button to submit the query
submit_query = st.button("Get Answer !!")

# Display response when the button is clicked
if submit_query:
    if input_text and uploaded_file:
        response = get_gemini_response(input_text, image)
        st.subheader("Response from Gemini:")
        st.write(response)
    elif input_text:
        response = get_gemini_response(input_text)
        st.subheader("Response from Gemini:")
        st.write(response)
    else:
        st.warning("Please provide a text input or an image for analysis.")
