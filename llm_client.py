import streamlit as st
import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# ✅ Reuse session (faster API calls)
session = requests.Session()

def call_llm(prompt):
    try:
        api_key = st.secrets.get("GROQ_API_KEY")

        if not api_key:
            return "❌ GROQ_API_KEY not found in Streamlit secrets"

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
                        "You are a precise RAG assistant. "
                        "Answer ONLY from the given context. "
                        "Do not hallucinate. "
                        "If answer is missing, say: 'Not found in context'."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,
            "max_tokens": 512,
            "top_p": 0.9
        }

        # 🚀 API call (reused session)
        res = session.post(
            GROQ_URL,
            headers=headers,
            json=payload,
            timeout=60
        )

        # ❌ Handle HTTP errors early
        if res.status_code != 200:
            try:
                error_data = res.json()
            except Exception:
                error_data = res.text
            return f"❌ Groq API Error {res.status_code}: {error_data}"

        # ✅ Parse JSON
        data = res.json()

        # ✅ Extract safely
        choices = data.get("choices")
        if choices and len(choices) > 0:
            return choices[0].get("message", {}).get("content", "").strip()

        # ❌ API error format
        if "error" in data:
            return f"❌ Groq Error: {data['error'].get('message', data['error'])}"

        return f"❌ Unexpected response: {data}"

    except requests.exceptions.Timeout:
        return "❌ Request timeout. Please try again."

    except requests.exceptions.ConnectionError:
        return "❌ Connection error. Check internet or API availability."

    except requests.exceptions.RequestException as e:
        return f"❌ Request error: {str(e)}"

    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"