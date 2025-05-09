import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

# Configure Gemini API Key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Path to your PDF or documents folder
DOCUMENTS_FOLDER = "data"

# Target save location
SAVE_PATH = "vector_stores/gemini_faiss_index"

# Prepare documents
all_docs = []
for file_name in os.listdir(DOCUMENTS_FOLDER):
    if file_name.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DOCUMENTS_FOLDER, file_name))
        docs = loader.load()
        all_docs.extend(docs)

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(all_docs)

# Generate embeddings using Gemini
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Build vector store
vector_store = FAISS.from_documents(split_docs, embeddings)

# Save the store
vector_store.save_local(SAVE_PATH)

print(f"✅ Gemini FAISS index saved to {SAVE_PATH}")
