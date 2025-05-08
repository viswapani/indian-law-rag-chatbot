from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="Indian Labour Law RAG API")

# Initialize on startup
@app.on_event("startup")
def load_chain():
    global qa_chain
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(query: Query):
    try:
        result = qa_chain(query.question)
        answer = result["result"]
        sources = [doc.metadata["source"] for doc in result["source_documents"]]
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
