from flask import Flask, request, jsonify
from openai import OpenAI
import os
import json
import re
from datetime import datetime, timezone

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

USER_MEMORY = {}

MAX_MEMORY_ITEMS = 15
MAX_RECENT_TURNS = 8


# ---------------- UTIL ----------------

def now_iso():
    return datetime.now(timezone.utc).isoformat()


def normalize(text):
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def is_fragment(message):
    words = message.strip().split()
    return len(words) <= 3


# ---------------- MEMORY ----------------

def get_user_memory(user):
    if user not in USER_MEMORY:
        USER_MEMORY[user] = {
            "memory_items": [],
            "recent_turns": [],
            "conversation_summary": ""
        }
    return USER_MEMORY[user]


def store_memory(memory, fact):
    if not fact or not fact.get("text"):
        return

    text = normalize(fact["text"])

    banned = [
        "feeling sad", "feels sad", "feeling emotional",
        "owns a cat", "owns a dog", "likes animals"
    ]

    if any(b in text for b in banned):
        return

    existing = [normalize(x["text"]) for x in memory["memory_items"]]

    if text in existing:
        return

    memory["memory_items"].append(fact)
    memory["memory_items"] = memory["memory_items"][-MAX_MEMORY_ITEMS:]


# ---------------- MEMORY EXTRACTION ----------------

def extract_memory(user, message):
    prompt = f"""
Extract memory ONLY if this is a CLEAR personal fact.

MESSAGE:
{message}

RULES:
- Only store real facts (job, events)
- DO NOT guess
- DO NOT infer
- DO NOT store fragments
- DO NOT store topics

Return JSON:
{{
 "should_store": true/false,
 "text": "...",
 "topic": "...",
 "importance": 0-1
}}
"""

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        data = json.loads(res.choices[0].message.content)

        if data.get("should_store"):
            memory = get_user_memory(user)
            store_memory(memory, {
                "text": data["text"],
                "topic": data.get("topic", ""),
                "importance": data.get("importance", 0.5),
                "created_at": now_iso()
            })

    except:
        pass


# ---------------- CONTEXT ----------------

def update_conversation(memory, user_msg, bot_reply):
    memory["recent_turns"].append({
        "user": user_msg,
        "assistant": bot_reply
    })
    memory["recent_turns"] = memory["recent_turns"][-MAX_RECENT_TURNS:]


def build_summary(memory):
    turns = memory["recent_turns"][-4:]

    if not turns:
        return ""

    text = "\n".join([f"user: {t['user']}" for t in turns])

    prompt = f"""
Summarise this conversation in ONE short sentence:

{text}
"""

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return res.choices[0].message.content.strip()

    except:
        return ""


# ---------------- REPLY ----------------

def generate_reply(user, message):
    memory = get_user_memory(user)

    summary = memory.get("conversation_summary", "")
    recent = memory.get("recent_turns", [])

    use_memory = not is_fragment(message)
    memories = memory["memory_items"][-4:] if use_memory else []

    system_prompt = """
You are Aria, a natural, confident Twitch chat personality.

CRITICAL RULES:

1. Stay tightly grounded in the LAST message.
- Do NOT drift.
- Do NOT expand beyond what the user said.

2. Fragment handling:
- If message is short or unclear (e.g. "a cat", "him", "that job"):
- DO NOT assume meaning.
- React or ask for clarification instead.
- NEVER invent context or ownership.

3. Emotional responses:
- Stay in the feeling.
- DO NOT give advice.
- DO NOT try to fix the situation.
- NO therapy phrases (e.g. "I'm here for you", "you’re not alone").

4. Keep replies SHORT:
- 1 sentence preferred
- 2 max
- No long explanations

5. Tone:
- Human, casual, slightly playful
- Not an assistant
- Not overly supportive

6. Memory:
- Only use memory if clearly relevant
- Never force it

GOOD EXAMPLES:
- "Yeah… that sounds like a rough one."
- "A cat? Wait — what’s the story there?"
- "Oof, I get why that stuck with you."

BAD:
- Giving advice
- Making assumptions
- Overexplaining
"""

    user_prompt = f"""
User: {user}

Conversation summary:
{summary}

Recent conversation:
{recent}

Relevant memory:
{memories}

New message:
{message}

Reply as Aria.
"""

    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7
    )

    return res.choices[0].message.content.strip()


# ---------------- ROUTE ----------------

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user = data.get("user")
    message = data.get("message")

    if not user or not message:
        return jsonify({"error": "Missing user or message"}), 400

    memory = get_user_memory(user)

    # extract memory safely
    extract_memory(user, message)

    # generate reply
    reply = generate_reply(user, message)

    # update conversation
    update_conversation(memory, message, reply)

    # update summary
    memory["conversation_summary"] = build_summary(memory)

    return jsonify({
        "reply": reply,
        "memory": memory
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
