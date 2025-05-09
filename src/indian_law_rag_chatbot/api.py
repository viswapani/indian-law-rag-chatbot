from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import shutil
import os

app = FastAPI(title="Indian Labour Law RAG API")

# Paths
PREBUILT_FAISS_PATH = "vector_stores/openai_faiss_index"
LATEST_UPLOAD_PATH = "vector_stores/latest_uploaded_faiss"

# ---------------------- Utility: Build and Save Vector Store ----------------------
def build_vector_store(file_path, save_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(split_docs, embeddings)
    vector_store.save_local(save_path)
    return vector_store

# ---------------------- Utility: Load Vector Store ----------------------
def load_vector_store(path):
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

# ---------------------- /upload: Process and Save Latest ----------------------
@app.post("/upload")
async def upload_pdf(file: UploadFile):
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        build_vector_store(file_location, LATEST_UPLOAD_PATH)
        return JSONResponse(content={"message": f"File '{file.filename}' processed and saved for QA."})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        os.remove(file_location)

# ---------------------- /ask: Check Latest Upload First ----------------------
@app.post("/ask")
async def ask_default(question: str = Form(...)):
    try:
        if os.path.exists(LATEST_UPLOAD_PATH):
            vector_store = load_vector_store(LATEST_UPLOAD_PATH)
        elif os.path.exists(PREBUILT_FAISS_PATH):
            vector_store = load_vector_store(PREBUILT_FAISS_PATH)
        else:
            raise ValueError("No vector store found for querying.")

        llm = ChatOpenAI(model_name="gpt-3.5-turbo")
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        answer = qa_chain.run(question)

        return JSONResponse(content={"answer": answer})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
