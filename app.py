import streamlit as st
import os
import tempfile

from rag_pipeline import chunk_documents, retrieve_docs, build_prompt
from vector_store import create_vectorstore
from llm_client import call_llm
from csv_handler import handle_csv_query

from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader

st.title("🤖 RAG Chatbot")

# -------------------------
# INIT SESSION STATE
# -------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "csv_file" not in st.session_state:
    st.session_state.csv_file = None

# -------------------------
# Upload Files
# -------------------------
files = st.file_uploader(
    "Upload files",
    type=["pdf", "txt", "csv"],
    accept_multiple_files=True
)

if files:
    docs = []
    csv_file_path = None

    for file in files:
        # ✅ Use temp file (avoids overwrite + safer in cloud)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.getvalue())
            file_path = tmp.name

        # Load based on file type
        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)

        elif file.name.endswith(".txt"):
            loader = TextLoader(file_path)

        elif file.name.endswith(".csv"):
            loader = CSVLoader(file_path)
            csv_file_path = file_path   # ✅ store CSV path

        else:
            continue

        loaded_docs = loader.load()

        # Add metadata
        for d in loaded_docs:
            d.metadata["source"] = file.name

        docs.extend(loaded_docs)

    # Create vector store
    chunks = chunk_documents(docs)
    st.session_state.vectorstore = create_vectorstore(chunks)

    # Save CSV file path
    st.session_state.csv_file = csv_file_path

    st.success("Documents processed successfully!")

# -------------------------
# Chat
# -------------------------
query = st.chat_input("Ask your question")

if query:
    if st.session_state.vectorstore is None:
        st.warning("Please upload files first")
    else:
        with st.spinner("Thinking..."):

            # -------------------------
            # SMART ROUTING (CSV vs RAG)
            # -------------------------
            if st.session_state.get("csv_file") and any(
                word in query.lower()
                for word in ["total", "sum", "show", "list", "all"]
            ):
                answer = handle_csv_query(query, st.session_state.csv_file)
                sources = ["CSV Data"]

            else:
                docs = retrieve_docs(st.session_state.vectorstore, query)

                context = "\n".join(
                    d.page_content.strip() for d in docs
                )

                prompt = build_prompt(context, query)

                answer = call_llm(prompt)

                # Deduplicate sources
                sources = sorted(set(
                    d.metadata.get("source", "file")
                    for d in docs
                ))

        # -------------------------
        # DISPLAY
        # -------------------------
        st.write("### Answer")
        st.write(answer)

        if sources:
            st.markdown("**📄 Source Documents:**")
            st.write(", ".join(sources))