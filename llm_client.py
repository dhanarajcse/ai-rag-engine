import streamlit as st
import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def call_llm(prompt):
    try:
        api_key = st.secrets["GROQ_API_KEY"]

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers based on provided context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3
        }

        # 🚀 Add timeout for Streamlit stability
        res = requests.post(
            GROQ_URL,
            headers=headers,
            json=payload,
            timeout=30
        )

        # 🔍 Parse response safely
        data = res.json()

        # ✅ SUCCESS CASE
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]

        # ❌ API ERROR CASE
        if "error" in data:
            return f"Groq Error: {data['error'].get('message', data['error'])}"

        # ❌ UNKNOWN RESPONSE
        return f"Unexpected response: {data}"

    except requests.exceptions.Timeout:
        return "Error: Request timeout. Please try again."

    except Exception as e:
        return f"Exception: {str(e)}"