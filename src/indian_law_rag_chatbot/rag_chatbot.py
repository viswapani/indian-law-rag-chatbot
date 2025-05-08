import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()  # Load OPENAI_API_KEY from .env

# def load_vector_store(faiss_folder="faiss_index"):
    # embeddings = OpenAIEmbeddings()
    # return FAISS.load_local(faiss_folder, embeddings)

def load_vector_store(faiss_folder="faiss_index"):
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(
        faiss_folder,
        embeddings,
        allow_dangerous_deserialization=True  # ADD THIS
    )

def create_rag_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

def ask_question(chain, query):
    result = chain(query)
    answer = result["result"]
    sources = [doc.metadata["source"] for doc in result["source_documents"]]
    return answer, sources

if __name__ == "__main__":
    store = load_vector_store()
    chain = create_rag_chain(store)

    print("Indian Labour Law RAG Chatbot")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Ask your question: ")
        if query.lower() in ["exit", "quit"]:
            break
        answer, sources = ask_question(chain, query)
        print("\n Answer:")
        print(answer)
        print("\n Sources:", sources)
        print("\n" + "-"*60 + "\n")
