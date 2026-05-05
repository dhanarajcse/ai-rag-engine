🤖 AI RAG Engine — Intelligent Document Chat Assistant

An Agentic RAG (Retrieval-Augmented Generation) chatbot that lets users upload documents and interact with them using natural language.

Built with Streamlit, LangChain, FAISS, Hugging Face embeddings, and Groq LLM, this app provides fast, context-aware answers from your data.


🚀 Live Demo
👉 https://ai-rag-engine.streamlit.app/


✨ Key Features
📄 Upload multiple PDF/TXT documents
💬 Ask questions based on document content
🧠 Context-aware responses using RAG pipeline
🔍 Semantic search with FAISS vector store
⚡ Fast inference using Groq LLM
📌 Source-aware answers for transparency


🧠 Agent-Based Query Routing

The system includes a lightweight agent layer to intelligently route queries:

RAG Tool → Answers document-based questions
Summarize Tool → Generates summaries
General Tool → Handles non-document queries


🏗️ Architecture Overview
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
FAISS Vector Store → Context Retrieval
   ↓
Groq LLM
   ↓
Final Answer + Sources


🛠️ Tech Stack

Python 3.11
Streamlit
LangChain
FAISS
Hugging Face Transformers
Sentence Transformers
Groq API
PyMuPDF / PyPDF

📁 Project Structure

ai-rag-engine/
│
├── app.py # Streamlit UI (main entry point)
├── agent.py # Agent logic (decision-making / tool routing)
├── tools.py # Custom tools used by the agent
├── rag_pipeline.py # RAG pipeline (retrieval + prompt handling)
├── vector_store.py # FAISS vector database setup
├── llm_client.py # LLM API integration
├── requirements.txt # Project dependencies
├── README.md # Project documentation
    

⚙️ Setup & Installation
git clone https://github.com/your-username/ai-rag-engine.git
cd ai-rag-engine

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt


🔐 Configuration

Create a file:
.streamlit/secrets.toml

Add your Groq API key:
GROQ_API_KEY = "your_groq_api_key_here"

▶️ Run Locally
streamlit run app.py

☁️ Deployment (Streamlit Cloud)
1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Create a new app
4. Add GROQ_API_KEY in secrets
    GROQ_API_KEY = "your_groq_api_key_here"


👨‍💻 Author
Dhanaraj K
AI & Full Stack Developer