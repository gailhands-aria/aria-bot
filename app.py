from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
import os
import json
import re
from datetime import datetime, timezone
import uuid

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

USER_MEMORY = {}

MAX_MEMORY_ITEMS = 15
MAX_RECENT_TURNS = 8
MAX_RECENT_OPENERS = 5

AUDIO_FOLDER = "audio"
os.makedirs(AUDIO_FOLDER, exist_ok=True)


# ---------------- UTIL ----------------

def now_iso():
    return datetime.now(timezone.utc).isoformat()


def normalize(text):
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def is_fragment(message):
    return len(message.strip().split()) <= 3


def get_opener(text):
    words = text.lower().split()
    return words[0] if words else ""


# ---------------- MEMORY ----------------

def get_user_memory(user):
    if user not in USER_MEMORY:
        USER_MEMORY[user] = {
            "memory_items": [],
            "recent_turns": [],
            "conversation_summary": "",
            "recent_openers": []
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
- Only store real facts
- DO NOT guess or infer
- DO NOT store fragments

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

    opener = get_opener(bot_reply)
    if opener:
        memory["recent_openers"].append(opener)
        memory["recent_openers"] = memory["recent_openers"][-MAX_RECENT_OPENERS:]


def build_summary(memory):
    turns = memory["recent_turns"][-4:]

    if not turns:
        return ""

    text = "\n".join([f"user: {t['user']}" for t in turns])

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Summarise briefly:\n{text}"}],
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
    openers = memory.get("recent_openers", [])

    system_prompt = f"""
You are Aria, a young, upbeat Twitch chat personality.

STYLE:
- 1 sentence
- lively, quick, slightly playful
- natural, not forced
- DO NOT use quotation marks

RULES:
- stay grounded
- no assumptions
- no advice

WORD RULES:
- no "oof"
- no "yo"
- avoid repeating openers: {openers}
"""

    user_prompt = f"""
User: {user}
Summary: {summary}
Recent: {recent}
Message: {message}

Reply:
"""

    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=1.0
    )

    reply = res.choices[0].message.content.strip()
    return reply.strip('"').strip("'")


# ---------------- TTS STYLE ----------------

def shape_for_voice(text):
    # 🔥 KEY PART — makes her sound alive
    text = text.replace(".", "")
    text = text.replace("!", "!")
    text = text.strip()

    # Add slight upbeat lift naturally
    if text.endswith("better"):
        text += " though"

    return text


# ---------------- TTS ----------------

def generate_tts(text):
    try:
        filename = f"aria_{uuid.uuid4().hex}.mp3"
        filepath = os.path.join(AUDIO_FOLDER, filename)

        styled = shape_for_voice(text)

        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="sage",
            input=styled
        ) as response:
            response.stream_to_file(filepath)

        return filename

    except Exception as e:
        print("TTS ERROR:", e)
        return None


# ---------------- AUDIO ROUTE ----------------

@app.route("/audio/<filename>")
def serve_audio(filename):
    return send_from_directory(AUDIO_FOLDER, filename)


# ---------------- ROUTE ----------------

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user = data.get("user")
    message = data.get("message")

    memory = get_user_memory(user)

    extract_memory(user, message)

    reply = generate_reply(user, message)

    update_conversation(memory, message, reply)

    memory["conversation_summary"] = build_summary(memory)

    audio_file = generate_tts(reply)

    audio_url = None
    if audio_file:
        audio_url = f"https://aria-bot-1.onrender.com/audio/{audio_file}"

    return jsonify({
        "reply": reply,
        "audio_url": audio_url
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
