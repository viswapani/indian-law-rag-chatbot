import os
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def preprocess_pdfs(pdf_folder="data/"):
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            path = os.path.join(pdf_folder, filename)
            text = extract_text_from_pdf(path)
            chunks = splitter.split_text(text)
            for chunk in chunks:
                all_chunks.append({"source": filename, "content": chunk})

    return all_chunks
