import streamlit as st
from llm_client import call_llm

# -------------------------
# DECIDE WHICH TOOL TO USE
# -------------------------
def decide_tool(query: str) -> str:
    router_prompt = f"""
You are a router.

Decide the best tool for the user query.

Available tools:
1. RAG → for answering questions from uploaded documents (PDF, TXT, CSV-as-text)

Rules:
- If the question requires information from documents → return: RAG
- Otherwise → return: RAG

Return ONLY one word: RAG

Query:
{query}
"""
    decision = call_llm(router_prompt)
    return (decision or "").strip().upper()


# -------------------------
# AGENT EXECUTION
# -------------------------
def run_agent(query, vectorstore, rag_fn):
    """
    rag_fn: function(query, vectorstore) -> (answer, sources)
    """
    return rag_fn(query, vectorstore)
