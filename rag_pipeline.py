from langchain_text_splitters import RecursiveCharacterTextSplitter


# -------------------------
# CHUNKING (stable + context-safe)
# -------------------------
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(docs)


# -------------------------
# RETRIEVAL (HYBRID + FALLBACK)
# -------------------------
def retrieve_docs(vectorstore, query, k=8):
    query_lower = query.lower().strip()
    query_words = query_lower.split()

    # 🔥 Step 1: get large candidate pool
    results = vectorstore.similarity_search(query, k=20)

    keyword_docs = []
    partial_docs = []
    semantic_docs = []

    for doc in results:
        text = doc.page_content.lower()

        # ✅ STRONG MATCH (all words present)
        if all(word in text for word in query_words):
            keyword_docs.append(doc)

        # ✅ PARTIAL MATCH (any word present)
        elif any(word in text for word in query_words):
            partial_docs.append(doc)

        # ✅ PURE SEMANTIC
        else:
            semantic_docs.append(doc)

    # 🔥 Priority order (VERY IMPORTANT)
    combined = keyword_docs + partial_docs + semantic_docs

    # 🔥 Remove duplicates
    seen = set()
    final_docs = []

    for doc in combined:
        content = doc.page_content.strip()

        if content not in seen:
            seen.add(content)
            final_docs.append(doc)

        if len(final_docs) >= k:
            break

    return final_docs


# -------------------------
# PROMPT (strict but practical)
# -------------------------
def build_prompt(context, question):
    return f"""
You are a precise document-based AI assistant.

RULES:
- Answer ONLY using the provided context
- If definition exists, return it clearly
- Do NOT miss key terms or lines
- Do NOT combine unrelated content
- Keep answer short (2–4 lines)

IMPORTANT:
- If relevant information exists → answer it
- Only say "Not found in context" if nothing relevant is present

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""