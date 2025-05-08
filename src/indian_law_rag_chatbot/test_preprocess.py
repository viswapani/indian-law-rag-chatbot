from indian_law_rag_chatbot.preprocess import preprocess_pdfs

if __name__ == "__main__":
    chunks = preprocess_pdfs()
    print(f"Loaded {len(chunks)} chunks")
