from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------------
# CHUNKING (improved)
# -------------------------
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,        # increased for better context
        chunk_overlap=150      # better continuity between chunks
    )
    return splitter.split_documents(docs)


# -------------------------
# RETRIEVAL (optimized)
# -------------------------
def retrieve_docs(vectorstore, query, k=3):
    docs = vectorstore.similarity_search(query, k=k)

    # optional: remove duplicate content chunks
    seen = set()
    unique_docs = []

    for d in docs:
        text = d.page_content.strip()
        if text not in seen:
            seen.add(text)
            unique_docs.append(d)

    return unique_docs


# -------------------------
# PROMPT (strong + structured)
# -------------------------
def build_prompt(context, question):
    return f"""
You are a precise AI assistant for document Q&A.

RULES:
- Answer ONLY using the given context.
- Do NOT repeat the question.
- Do NOT continue incomplete sentences.
- If answer is not in context, say exactly: "Not found in context"
- Keep answer clear, short, and structured (3-5 lines max).

QUESTION:
{question}

CONTEXT:
{context}

FINAL ANSWER:
"""