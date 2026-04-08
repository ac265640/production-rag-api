# generation/generator.py

import requests
from config import settings


def build_prompt(query: str, contexts: list[str]) -> str:
    """
    Builds grounding prompt for LLM
    """
    context_text = "\n\n".join(contexts)

    prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the context below.
If the answer is not present, say: "Not in context".

Context:
{context_text}

Question:
{query}

Answer:
"""
    return prompt


def generate(query: str, contexts: list[str]) -> str:
    """
    Calls Groq API to generate answer
    """

    prompt = build_prompt(query, contexts)

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {settings.GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": settings.LLM_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": settings.LLM_TEMPERATURE
    }

    response = requests.post(url, headers=headers, json=data)

    try:
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print("Groq API Error:", response.text)
        return "Error generating response"