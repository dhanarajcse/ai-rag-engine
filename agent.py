from tools import rag_tool, summarize_tool, general_tool


def decide_tool(query: str, has_docs: bool) -> str:
    query_lower = query.lower().strip()

    summary_words = [
        "summarize",
        "summary",
        "overview",
        "brief",
        "short notes",
        "main points",
        "key points"
    ]

    general_words = [
        "who are you",
        "what can you do",
        "help",
        "hello",
        "hi"
    ]

    if any(word in query_lower for word in summary_words):
        return "SUMMARY"

    if any(word in query_lower for word in general_words):
        return "GENERAL"

    if has_docs:
        return "RAG"

    return "GENERAL"


def run_agent(query, vectorstore, all_docs):
    has_docs = vectorstore is not None and len(all_docs) > 0

    tool = decide_tool(query, has_docs)

    if tool == "SUMMARY":
        return summarize_tool(all_docs)

    if tool == "GENERAL":
        return general_tool(query)

    return rag_tool(query, vectorstore, all_docs)
