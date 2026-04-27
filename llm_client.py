import streamlit as st
import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def call_llm(prompt):
    try:
        # ✅ Safe secret access
        api_key = st.secrets.get("GROQ_API_KEY", None)

        if not api_key:
            return "❌ Missing GROQ_API_KEY in Streamlit secrets"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an intelligent assistant. "
                        "Answer ONLY using provided context. "
                        "If not available, say 'Not found in context'."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,   # 🔽 slightly lower = more accurate RAG answers
            "max_tokens": 512     # 🔥 prevents overly long responses
        }

        # 🚀 API call with timeout
        res = requests.post(
            GROQ_URL,
            headers=headers,
            json=payload,
            timeout=30
        )

        # 🔍 Try parsing JSON safely
        try:
            data = res.json()
        except Exception:
            return f"❌ Invalid API response: {res.text}"

        # ❌ HTTP error handling
        if res.status_code != 200:
            return f"❌ Groq API Error {res.status_code}: {data}"

        # ✅ SUCCESS
        if isinstance(data, dict) and "choices" in data:
            return data["choices"][0]["message"]["content"].strip()

        # ❌ API ERROR RESPONSE
        if isinstance(data, dict) and "error" in data:
            return f"❌ Groq Error: {data['error'].get('message', data['error'])}"

        # ❌ UNKNOWN RESPONSE
        return f"❌ Unexpected response format: {data}"

    except requests.exceptions.Timeout:
        return "❌ Request timeout. Please try again."

    except requests.exceptions.RequestException as e:
        return f"❌ Network error: {str(e)}"

    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"