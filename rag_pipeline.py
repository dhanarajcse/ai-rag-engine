from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------------
# CHUNKING (balanced for text + tables)
# -------------------------
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,      # balanced (not too big, not too small)
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)


# -------------------------
# RETRIEVAL (clean + relevant)
# -------------------------
def retrieve_docs(vectorstore, query, k=6):
    results = vectorstore.similarity_search_with_score(query, k=k)

    # sort by similarity score (lower is better)
    results.sort(key=lambda x: x[1])

    docs = []
    seen = set()

    for doc, _ in results:
        text = doc.page_content.strip()

        # remove duplicates
        if text not in seen:
            seen.add(text)
            docs.append(doc)

    return docs


# -------------------------
# PROMPT (table + text optimized)
# -------------------------
def build_prompt(context, question):
    return f"""
You are a highly accurate AI assistant.

The provided context may contain:
- plain text
- tabular data (rows/columns like CSV or tables)

INSTRUCTIONS:
- Use ONLY the given context
- If data exists, DO NOT say "Not found"
- Carefully match keys (e.g., Ride No → Earnings)
- Do NOT skip rows in tables
- Do NOT mix multiple answers
- Provide a clear and complete answer

IF NOT FOUND:
Return exactly: Not found in context

QUESTION:
{question}

CONTEXT:
{context}

FINAL ANSWER:
"""