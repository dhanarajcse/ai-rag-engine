from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"}  # 🔥 force CPU (important for cloud)
    )

def create_vectorstore(docs):
    embeddings = get_embeddings()
    return FAISS.from_documents(docs, embeddings)