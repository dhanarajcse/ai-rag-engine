# 🤖 AI RAG Engine — Intelligent Document Chat Assistant

An **Agentic RAG (Retrieval-Augmented Generation)** chatbot that allows users to upload documents and interact with them using natural language.

Built with **Streamlit, LangChain, FAISS, Hugging Face Embeddings, and Groq LLM**, this app delivers fast, context-aware answers from our data.

---

## 🚀 Live Demo
👉 https://ai-rag-engine.streamlit.app/

---

## ✨ Key Features

- 📄 Upload multiple **PDF / TXT documents**
- 💬 Ask questions based on document content
- 🧠 Context-aware responses using RAG pipeline
- 🔍 Semantic search with FAISS vector store
- ⚡ Fast inference using Groq LLM
- 📌 Source-aware answers for transparency
- 🤖 Agent-based query routing (RAG / Summary / General)

---

## 🧠 Agent-Based Query Routing

The system uses a lightweight agent to intelligently route queries:

- **RAG Tool** → Answers document-based questions  
- **Summarize Tool** → Generates summaries  
- **General Tool** → Handles non-document queries  

---

## 🏗️ Architecture Overview

```
User Query
   ↓
Streamlit UI
   ↓
Agent Router
   ↓
Tool Selection (RAG / Summary / General)
   ↓
Document Processing → Chunking → Embeddings
   ↓
FAISS Vector Store
   ↓
Context Retrieval
   ↓
Groq LLM
   ↓
Final Answer + Sources
```

---

## 🛠️ Tech Stack

- Python 3.11  
- Streamlit  
- LangChain  
- FAISS  
- Hugging Face Transformers / Sentence Transformers  
- Groq API  
- PyMuPDF / PyPDF  

---

## 📁 Project Structure

```
ai-rag-engine/
│
├── app.py                 # Streamlit UI (main entry point)
├── agent.py               # Agent logic (decision-making / routing)
├── tools.py               # Custom tools
├── rag_pipeline.py        # RAG pipeline
├── vector_store.py        # FAISS setup
├── llm_client.py          # LLM integration
├── requirements.txt       # Dependencies
├── README.md

```

---

## ⚙️ Setup Instructions

```bash
git clone https://github.com/dhanarajcse/ai-rag-engine
cd ai-rag-engine
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔐 Environment Setup

Create `.streamlit/secrets.toml`:

```
GROQ_API_KEY="your_api_key_here"
```

---

## 📌 Future Enhancements

- Chat history memory  
- Multi-user authentication  
- Support for more file formats (DOCX, CSV)  
- Deployment with Docker + AWS  

---


## 👨‍💻 Author

```
Dhanaraj K
Full Stack Developer
```
