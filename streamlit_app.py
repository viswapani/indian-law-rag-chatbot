import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Indian Labour Law RAG Chatbot")

if "upload_success" not in st.session_state:
    st.session_state["upload_success"] = False

mode = st.radio("Select Mode:", ["Default Knowledge Base", "Upload Custom PDF"])

# ----------- Default Knowledge Base -----------
if mode == "Default Knowledge Base":
    st.subheader("Ask a Question on Default Indian Labour Laws")
    question = st.text_input("Enter your question:")
    if st.button("Submit Question") and question.strip():
        response = requests.post(f"{API_URL}/ask", data={"question": question})
        if response.status_code == 200:
            data = response.json()
            st.write("### Answer")
            st.write(data.get("answer", "No answer returned."))
        else:
            st.error(f"Error {response.status_code}: {response.text}")

# ----------- Upload and Immediate Follow-up -----------
elif mode == "Upload Custom PDF":
    st.subheader("Upload Your PDF and Optionally Ask a Follow-Up")

    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
    follow_up_question = st.text_input("Optionally, enter a follow-up question now:")

    if uploaded_file and st.button("Upload and Process (with optional follow-up)"):
        with st.spinner("Uploading and processing..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = requests.post(f"{API_URL}/upload", files=files)
                if response.status_code == 200:
                    st.success(f"File '{uploaded_file.name}' processed successfully.")
                    st.session_state["upload_success"] = True

                    # Automatically ask follow-up if provided
                    if follow_up_question.strip():
                        st.write(f"Sending follow-up to {API_URL}/ask")
                        follow_up_response = requests.post(f"{API_URL}/ask", data={"question": follow_up_question})
                        if follow_up_response.status_code == 200:
                            data = follow_up_response.json()
                            st.write("### Answer")
                            st.write(data.get("answer", "No answer returned."))
                        else:
                            st.error(f"Error {follow_up_response.status_code}: {follow_up_response.text}")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
                    st.session_state["upload_success"] = False

            except Exception as e:
                st.error(f"Exception: {str(e)}")
                st.session_state["upload_success"] = False
