from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(docs)


def retrieve_docs(vectorstore, query, k=4):
    docs = vectorstore.similarity_search(query, k=k)
    return docs


def build_prompt(context, question):
    return f"""
Answer using only the context.

Context:
{context}

Question:
{question}

If not found say "Not found in context"
"""