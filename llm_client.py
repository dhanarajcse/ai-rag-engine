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
            "model": "llama3-8b-8192",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }

        res = requests.post(GROQ_URL, headers=headers, json=payload)

        # 🔍 Convert response safely
        data = res.json()

        # ✅ SUCCESS CASE
        if "choices" in data:
            return data["choices"][0]["message"]["content"]

        # ❌ API ERROR CASE
        elif "error" in data:
            return f"Error: {data['error']['message']}"

        # ❌ UNKNOWN RESPONSE
        else:
            return f"Unexpected response: {data}"

    except Exception as e:
        return f"Exception: {str(e)}"