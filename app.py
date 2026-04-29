import streamlit as st
import tempfile

from rag_pipeline import chunk_documents, retrieve_docs, build_prompt
from vector_store import create_vectorstore
from llm_client import call_llm
from agent import run_agent

from langchain_community.document_loaders import PyPDFLoader, TextLoader

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("🤖 AI RAG Chatbot")

# -------------------------
# INIT SESSION STATE
# -------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "all_docs" not in st.session_state:
    st.session_state.all_docs = []

# ✅ Track uploaded files
if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = set()

# ✅ UI message flags
if "show_clear_msg" not in st.session_state:
    st.session_state.show_clear_msg = False

if "show_reset_msg" not in st.session_state:
    st.session_state.show_reset_msg = False

# ✅ NEW: uploader key (for clearing UI)
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
# RAG FUNCTION (Agent uses this)
# -------------------------
def rag_answer(query, vectorstore):
    docs = retrieve_docs(vectorstore, query)

    context = "\n".join(d.page_content.strip() for d in docs)

    prompt = build_prompt(context, query)
    answer = call_llm(prompt)

    sources = sorted(set(
        d.metadata.get("source", "file") for d in docs
    ))

    return answer, sources


# -------------------------
# FILE UPLOAD (ONLY NEW FILES)
# -------------------------
files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}"  # 🔥 key fix
)

if files:
    new_files = [
        f for f in files
        if f.name not in st.session_state.uploaded_file_names
    ]

    if new_files:
        new_docs = []

        for file in new_files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(file.getvalue())
                file_path = tmp.name

            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path)

            loaded_docs = loader.load()

            for d in loaded_docs:
                d.metadata["source"] = file.name

            new_docs.extend(loaded_docs)

            # track processed file
            st.session_state.uploaded_file_names.add(file.name)

        # append docs
        st.session_state.all_docs.extend(new_docs)

        # rebuild vectorstore
        with st.spinner("🔄 Processing documents..."):
            chunks = chunk_documents(st.session_state.all_docs)
            st.session_state.vectorstore = create_vectorstore(chunks)

        st.success(f"✅ {len(new_files)} new file(s) processed!")


# -------------------------
# CHAT UI
# -------------------------
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

query = st.chat_input("Ask your question...")

if query:
    st.session_state.chat_history.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    if st.session_state.vectorstore is None:
        response = "⚠ Please upload documents first"
        sources = []
    else:
        with st.spinner("🤖 Thinking..."):
            response, sources = run_agent(
                query,
                st.session_state.vectorstore,
                rag_answer
            )

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response
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

# ✅ CLEAR CHAT
with col1:
    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.show_clear_msg = True
        st.rerun()

# ✅ RESET DOCUMENTS (FULL RESET + CLEAR UPLOADER UI)
with col2:
    if st.button("🔄 Reset Documents"):
        st.session_state.vectorstore = None
        st.session_state.all_docs = []
        st.session_state.chat_history = []
        st.session_state.uploaded_file_names = set()

        # 🔥 reset uploader UI
        st.session_state.uploader_key += 1

        st.session_state.show_reset_msg = True
        st.rerun()