🤖 AI RAG Engine — Intelligent Document Chat Assistant

    Ask questions from your documents and get accurate, context-aware AI responses powered by Retrieval-Augmented Generation.

🚀 Live Demo
    https://ai-rag-engine.streamlit.app

📸 Demo Preview


⚡ Features

    📄 Upload multiple documents (PDF, TXT, CSV)
    💬 Chat with your documents in natural language
    🧠 AI-powered contextual answers (RAG pipeline)
    🔍 Semantic search using FAISS vector database
    📚 Source-based responses for transparency
    📊 Handles large documents efficiently
    🧹 Clear chat history option

🏗️ System Architecture
    User Query
        ↓
    Streamlit UI
        ↓
    Document Loader (PDF / TXT / CSV)
        ↓
    Text Chunking (LangChain)
        ↓
    Embeddings (Sentence Transformers)
        ↓
    FAISS Vector Store
        ↓
    Top-K Retrieval
        ↓
    LLM (Groq / OpenAI)
        ↓
    Final Answer + Sources

🧰 Tech Stack

    🐍 Python
    🎈 Streamlit
    🧠 LangChain
    📦 FAISS (Vector Database)
    🤗 Sentence Transformers
    ⚡ Groq / OpenAI API
    📄 PyMuPDF / PyPDF

📁 Project Structure
        ai-rag-engine/
    │
    ├── app.py                  # Streamlit UI
    ├── rag_pipeline.py         # RAG logic (retrieval + prompts)
    ├── vector_store.py        # FAISS vector store setup
    ├── llm_client.py          # LLM API integration
    ├── requirements.txt
    │
    ├── .streamlit/
    │   └── secrets.toml       # API keys (not pushed to GitHub)
    │
    ├── assets/
    │   └── demo.png           # screenshots
    │
    └── README.md

⚙️ Installation (Local Setup)
    1. Clone repository
        git clone https://github.com/your-username/ai-rag-engine.git
        cd ai-rag-engine

    2. Create virtual environment
        python -m venv venv
        venv\Scripts\activate   # Windows

    3. Install dependencies
        pip install -r requirements.txt

    🔐 Environment Setup
        Create .streamlit/secrets.toml
        GROQ_API_KEY = "your_api_key_here"

    ▶️ Run Application
        streamlit run app.py

    ☁️ Deployment (Streamlit Cloud)
        Push code to GitHub
        Go to https://streamlit.io/cloud
        Create new app
        Add repository
        Add secrets:
                GROQ_API_KEY = "xxx"

    Deploy 🚀

    🧠 How It Works

        User uploads documents
        Documents are split into chunks
        Embeddings are generated
        FAISS stores vectors
        User asks a question
        Relevant chunks are retrieved
        LLM generates answer using context

    👨‍💻 Author

        Dhanaraj Kathirvel
        AI & Full Stack Developer