
# UPDATED VERSION - fixes memory text issues and repetition

from flask import Flask, request, jsonify
from openai import OpenAI
import os
import random
import json
import re
from datetime import datetime, timezone

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

USER_MEMORY = {}

MAX_MEMORY_ITEMS = 12
MAX_RECENT_REPLIES = 4
MAX_RECENT_TOPICS = 6
MAX_RELEVANT_MEMORIES_FOR_PROMPT = 3


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def safe_json_loads(text):
    try:
        return json.loads(text)
    except Exception:
        return None


def clean_text(text, limit=240):
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    text = text.rstrip(".")  # remove trailing dots
    return text[:limit]


def sanitize_fact(text, user):
    """
    Removes weird formatting like 'tim's dog died..'
    """
    if not text:
        return ""

    text = text.strip().rstrip(".")

    # remove username possession like "tim's"
    text = re.sub(fr"\b{re.escape(user.lower())}'s\b", "their", text.lower())

    return text


def get_user_memory(user):
    if user not in USER_MEMORY:
        USER_MEMORY[user] = {
            "user": user,
            "seen_count": 0,
            "memory_items": [],
            "recent_replies": []
        }
    return USER_MEMORY[user]


def extract_memory_with_model(user, message):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "Extract a meaningful memory if present."
                },
                {
                    "role": "user",
                    "content": f"{user}: {message}"
                }
            ]
        )

        raw = response.choices[0].message.content or "{}"
        data = safe_json_loads(raw)

        if not data or not data.get("should_store"):
            return None

        fact_text = clean_text(data.get("fact", ""))
        fact_text = sanitize_fact(fact_text, user)

        if not fact_text:
            return None

        return {
            "text": fact_text,
            "importance": float(data.get("importance", 0.6))
        }

    except Exception:
        return None


def store_memory_fact(memory, fact):
    if not fact:
        return

    # prevent duplicates
    existing = [x["text"] for x in memory["memory_items"]]
    if fact["text"] in existing:
        return

    memory["memory_items"].append(fact)
    memory["memory_items"] = sorted(
        memory["memory_items"],
        key=lambda x: x.get("importance", 0),
        reverse=True
    )[:MAX_MEMORY_ITEMS]


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user = str(data.get("user", "viewer")).strip()
    message = str(data.get("message", "")).strip()

    memory = get_user_memory(user)

    extracted_fact = extract_memory_with_model(user, message)
    if extracted_fact:
        store_memory_fact(memory, extracted_fact)

    # simple reply for now
    if memory["memory_items"]:
        fact = memory["memory_items"][0]["text"]
        reply = f"I'm really sorry. It makes sense you'd feel sad after {fact}."
    else:
        reply = "I'm here with you."

    memory["recent_replies"].append(reply)
    memory["recent_replies"] = memory["recent_replies"][-MAX_RECENT_REPLIES:]

    return jsonify({
        "reply": reply,
        "memory": memory
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
