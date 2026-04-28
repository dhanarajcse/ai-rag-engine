import streamlit as st
import tempfile

from rag_pipeline import chunk_documents, retrieve_docs, build_prompt
from vector_store import create_vectorstore
from llm_client import call_llm

from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader

st.title("🤖 RAG Chatbot")

# -------------------------
# INIT SESSION STATE
# -------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# -------------------------
# Upload Files
# -------------------------
files = st.file_uploader(
    "Upload files",
    type=["pdf", "txt"],  # ✅ CSV removed (optional)
    accept_multiple_files=True
)

if files:
    docs = []

    for file in files:
        # temp file (safe for cloud)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.getvalue())
            file_path = tmp.name

        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)

        elif file.name.endswith(".txt"):
            loader = TextLoader(file_path)

        # (Optional) If you still want CSV as text via RAG
        elif file.name.endswith(".csv"):
            loader = CSVLoader(file_path)

        else:
            continue

        loaded_docs = loader.load()

        # add metadata
        for d in loaded_docs:
            d.metadata["source"] = file.name

        docs.extend(loaded_docs)

    # create vector store
    chunks = chunk_documents(docs)
    st.session_state.vectorstore = create_vectorstore(chunks)

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

            docs = retrieve_docs(st.session_state.vectorstore, query)

            context = "\n".join(
                d.page_content.strip() for d in docs
            )

            prompt = build_prompt(context, query)

            answer = call_llm(prompt)

            # deduplicate sources
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