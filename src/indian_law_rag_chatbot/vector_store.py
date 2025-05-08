import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from indian_law_rag_chatbot.preprocess import preprocess_pdfs

load_dotenv()

def build_vector_store(pdf_folder="data/", faiss_folder="faiss_index"):
    chunks = preprocess_pdfs(pdf_folder)

    embeddings = OpenAIEmbeddings()
    texts = [chunk["content"] for chunk in chunks]
    metadatas = [{"source": chunk["source"]} for chunk in chunks]

    db = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    db.save_local(faiss_folder)
    print(f"Vector store saved to: {faiss_folder}")

if __name__ == "__main__":
    build_vector_store()
