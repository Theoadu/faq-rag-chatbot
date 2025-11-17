import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

def evaluate_answer(user_question: str, system_answer: str, chunks_related: list) -> dict:
    chunks_text = "\n---\n".join(chunks_related)
    prompt = f"""
You are an AI quality auditor. Score the RAG system's response on a scale of 0–10 based on:
1. Relevance: Do the provided chunks address the user's question?
2. Accuracy: Does the system answer correctly reflect the chunk content?
3. Completeness: Is all key information from the chunks included?

User Question: {user_question}
System Answer: {system_answer}
Retrieved Chunks:
{chunks_text}

Respond ONLY with a JSON object: {{"score": integer (0-10), "reason": "brief explanation"}}
"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return {"score": result["score"], "reason": result["reason"]}
    except Exception as e:
        return {"score": 0, "reason": f"Evaluator error: {str(e)}"}