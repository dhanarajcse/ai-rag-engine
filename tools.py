from llm_client import call_llm
from rag_pipeline import retrieve_docs


MAX_CONTEXT_CHARS = 12000


def get_uploaded_file_names(all_docs):
    return sorted(set(
        d.metadata.get("source", "")
        for d in all_docs
        if d.metadata.get("source")
    ))


def get_docs_for_file(file_name, all_docs):
    return [
        d for d in all_docs
        if d.metadata.get("source") == file_name
    ]


def detect_mentioned_files(query, all_docs):
    query_lower = query.lower()
    file_names = get_uploaded_file_names(all_docs)

    return [
        file_name
        for file_name in file_names
        if file_name.lower() in query_lower
    ]


def build_context_from_docs(docs):
    seen = set()
    context_parts = []

    for doc in docs:
        content = doc.page_content.strip()

        if content and content not in seen:
            seen.add(content)
            source = doc.metadata.get("source", "file")
            context_parts.append(f"[Source: {source}]\n{content}")

    return "\n\n".join(context_parts)


def get_document_hints(context):
    lines = [
        line.strip()
        for line in context.splitlines()
        if line.strip()
    ]

    hints = []

    greeting_words = ("hi ", "hello ", "dear ", "to ")
    closing_words = ("thank you", "thanks", "regards", "sincerely", "from ")

    for line in lines[:5]:
        lower_line = line.lower()
        if lower_line.startswith(greeting_words):
            hints.append(f"Possible recipient/audience: {line}")

    for line in lines[-5:]:
        lower_line = line.lower()
        if lower_line.startswith(closing_words) or lower_line.startswith("team "):
            hints.append(f"Possible sender/provider: {line}")

    return "\n".join(hints)


def answer_from_context(query, context):
    if not context:
        return "Not found in context"

    hints = get_document_hints(context)

    prompt = f"""
You are a reliable document question-answering assistant.

Use ONLY the provided context and document hints.

Guidelines:
- Answer using facts from the context.
- Match meaning, not only exact words.
- If the question asks "for whom", "to whom", "assigned to whom", or "who is it for", identify the recipient/audience.
- If the question asks "who assigned", "who gave", "who provided", or "who sent", identify the sender/provider.
- Greetings can indicate the recipient/audience.
- Signatures or closing lines can indicate the sender/provider.
- If the answer is still not supported, say: Not found in context.
- Keep the answer short and clear.

DOCUMENT HINTS:
{hints}

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    return call_llm(prompt)


def answer_from_specific_files(query, all_docs):
    mentioned_files = detect_mentioned_files(query, all_docs)

    answers = []
    sources = []

    for file_name in mentioned_files:
        file_docs = get_docs_for_file(file_name, all_docs)
        context = build_context_from_docs(file_docs)
        answer = answer_from_context(query, context)

        answers.append(f"**{file_name}:**\n{answer}")
        sources.append(file_name)

    return "\n\n".join(answers), sources


def rag_tool(query, vectorstore, all_docs=None):
    if all_docs:
        mentioned_files = detect_mentioned_files(query, all_docs)

        if mentioned_files:
            return answer_from_specific_files(query, all_docs)

        full_context = build_context_from_docs(all_docs)

        if len(full_context) <= MAX_CONTEXT_CHARS:
            answer = answer_from_context(query, full_context)
            sources = get_uploaded_file_names(all_docs)

            return answer, sources

    docs = retrieve_docs(vectorstore, query)
    context = build_context_from_docs(docs)

    answer = answer_from_context(query, context)

    sources = sorted(set(
        d.metadata.get("source", "file") for d in docs
    ))

    return answer, sources


def summarize_tool(all_docs):
    context = build_context_from_docs(all_docs)

    if not context:
        return "No readable document content found.", []

    prompt = f"""
Summarize the uploaded document content clearly.

Use only the content below.
Keep the summary short and useful.

CONTENT:
{context}
"""

    return call_llm(prompt), get_uploaded_file_names(all_docs)


def general_tool(query):
    prompt = f"""
Answer this question clearly and briefly:

{query}
"""
    return call_llm(prompt), []
