import streamlit as st
import requests
import os

# Read from environment or fallback to localhost
API_URL = os.getenv("API_URL", "http://localhost:8000/ask")

st.title("Indian Labour Law RAG Chatbot")

question = st.text_input("Ask a question:")

if st.button("Submit") and question.strip():
    response = requests.post(API_URL, json={"question": question})
    if response.status_code == 200:
        data = response.json()
        st.write("### Answer")
        st.write(data["answer"])
        st.write("### Sources")
        st.write(", ".join(data["sources"]))
    else:
        st.error("Error contacting the API")
