import streamlit as st
from rag_pipeline import chunk_documents, retrieve_docs, build_prompt
from vector_store import create_vectorstore
from llm_client import call_llm

from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader

st.title("🤖 RAG Chatbot")

# -------------------------
# Upload Files
# -------------------------
files = st.file_uploader(
    "Upload files",
    type=["pdf", "txt", "csv"],
    accept_multiple_files=True
)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if files:
    docs = []

    for file in files:
        with open(file.name, "wb") as f:
            f.write(file.getvalue())

        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(file.name)
        elif file.name.endswith(".txt"):
            loader = TextLoader(file.name)
        else:
            loader = CSVLoader(file.name)

        loaded_docs = loader.load()

        # ✅ FIX: ensure source metadata is set
        for d in loaded_docs:
            d.metadata["source"] = file.name

        docs.extend(loaded_docs)

    chunks = chunk_documents(docs)

    st.session_state.vectorstore = create_vectorstore(chunks)

    st.success("Documents processed!")

# -------------------------
# Chat
# -------------------------
query = st.chat_input("Ask your question")

if query:
    if st.session_state.vectorstore is None:
        st.warning("Upload files first")
    else:
        docs = retrieve_docs(st.session_state.vectorstore, query)

        context = "\n".join([d.page_content for d in docs])

        prompt = build_prompt(context, query)

        answer = call_llm(prompt)

        st.write("### Answer")
        st.write(answer)

        # -------------------------
        # FIX: REMOVE DUPLICATE SOURCES
        # -------------------------
        sources = sorted(set(
            d.metadata.get("source", "file")
            for d in docs
        ))

        st.write("### Sources")

        # cleaner UI display
        st.markdown("**Source Documents:**")
        st.write(", ".join(sources))