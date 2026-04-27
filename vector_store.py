from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=MODEL_NAME)


def create_vectorstore(docs):
    embeddings = get_embeddings()
    return FAISS.from_documents(docs, embeddings)