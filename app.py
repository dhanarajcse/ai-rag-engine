import streamlit as st
import tempfile
import os

from rag_pipeline import chunk_documents
from vector_store import create_vectorstore
from agent import run_agent

from langchain_community.document_loaders import PyPDFLoader, TextLoader

st.set_page_config(page_title="Agentic RAG Chatbot", layout="wide")

st.title("🤖 Agentic RAG Chatbot")

# -------------------------
# INIT SESSION STATE
# -------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "all_docs" not in st.session_state:
    st.session_state.all_docs = []

if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = set()

if "show_clear_msg" not in st.session_state:
    st.session_state.show_clear_msg = False

if "show_reset_msg" not in st.session_state:
    st.session_state.show_reset_msg = False

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


# -------------------------
# SHOW TOAST MESSAGES
# -------------------------
if st.session_state.get("show_clear_msg"):
    st.toast("🧹 Chat cleared successfully")
    st.session_state.show_clear_msg = False

if st.session_state.get("show_reset_msg"):
    st.toast("🔄 Documents reset successfully")
    st.session_state.show_reset_msg = False


# -------------------------
# FILE UPLOAD
# -------------------------
files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}"
)

if files:
    new_files = [
        f for f in files
        if f.name not in st.session_state.uploaded_file_names
    ]

    if new_files:
        new_docs = []

        for file in new_files:
            suffix = os.path.splitext(file.name)[1].lower()

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file.getvalue())
                file_path = tmp.name

            if suffix == ".pdf":
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path, encoding="utf-8")

            loaded_docs = loader.load()

            for d in loaded_docs:
                d.metadata["source"] = file.name

            loaded_docs = [
                d for d in loaded_docs
                if d.page_content and d.page_content.strip()
            ]

            new_docs.extend(loaded_docs)
            st.session_state.uploaded_file_names.add(file.name)

        if new_docs:
            st.session_state.all_docs.extend(new_docs)

            with st.spinner("🔄 Processing documents..."):
                chunks = chunk_documents(st.session_state.all_docs)
                st.session_state.vectorstore = create_vectorstore(chunks)

            st.success(f"✅ {len(new_files)} new file(s) processed!")
        else:
            st.warning("⚠ No readable text found in the uploaded file(s).")


# -------------------------
# CHAT UI
# -------------------------
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

        if chat["role"] == "assistant" and chat.get("sources"):
            st.markdown("**📄 Sources:**")
            st.write(", ".join(chat["sources"]))

query = st.chat_input("Ask your question...")

if query:
    st.session_state.chat_history.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    if st.session_state.vectorstore is None:
        response = "⚠ Please upload documents first"
        sources = []
    else:
        with st.spinner("🤖 Agent is thinking..."):
            response, sources = run_agent(
                query,
                st.session_state.vectorstore,
                st.session_state.all_docs
            )

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response,
        "sources": sources
    })

    with st.chat_message("assistant"):
        st.markdown(response)

        if sources:
            st.markdown("**📄 Sources:**")
            st.write(", ".join(sources))


# -------------------------
# ACTION BUTTONS
# -------------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.show_clear_msg = True
        st.rerun()

with col2:
    if st.button("🔄 Reset Documents"):
        st.session_state.vectorstore = None
        st.session_state.all_docs = []
        st.session_state.chat_history = []
        st.session_state.uploaded_file_names = set()
        st.session_state.uploader_key += 1
        st.session_state.show_reset_msg = True
        st.rerun()