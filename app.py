from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
from cartesia import Cartesia
import os
import json
import re
from datetime import datetime, timezone
import uuid
import random

app = Flask(__name__)

# ---------------- CLIENTS ----------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
cartesia_client = Cartesia(api_key=os.getenv("CARTESIA_API_KEY"))
CARTESIA_VOICE_ID = os.getenv("CARTESIA_VOICE_ID")

# ---------------- MEMORY ----------------

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
    return ""


# ---------------- REPLY ----------------

def generate_reply(user, message):
    system_prompt = """
You are Aria, a young, upbeat, slightly playful Twitch personality.

STYLE:
- 1 sentence
- warm, natural, slightly flirty/playful
- light teasing is okay
- DO NOT use emojis
- DO NOT use asterisks

TONE:
- sound happy and alive
- soft, human, not robotic
- slightly expressive but not exaggerated

RULES:
- no "oof"
- no "yo"
"""

    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ],
        temperature=1.0
    )

    return res.choices[0].message.content.strip()


# ---------------- VOICE SHAPING ----------------

def shape_for_voice(text):
    text = text.strip()

    # remove any weird formatting
    text = re.sub(r"\*.*?\*", "", text)

    # 🎯 make "aw" sound soft and stretched
    text = re.sub(r"\b[Aa]w+\b", "aawh", text)

    # soften tone
    text = text.replace("haha", "heh")
    text = text.replace("Haha", "Heh")

    # 💕 add soft giggle at end sometimes
    if random.random() < 0.4:
        text += " hehe"

    # smooth punctuation
    text = text.replace(".", "")
    text = text.replace("...", ", ")

    return text.strip()


# ---------------- TTS ----------------

def generate_tts(text):
    try:
        filename = f"aria_{uuid.uuid4().hex}.wav"
        filepath = os.path.join(AUDIO_FOLDER, filename)

        styled = shape_for_voice(text)

        response = cartesia_client.tts.generate(
            model_id="sonic-3",
            transcript=styled,
            voice={
                "mode": "id",
                "id": CARTESIA_VOICE_ID
            },
            output_format={
                "container": "wav",
                "encoding": "pcm_f32le",
                "sample_rate": 44100,
            },
        )

        response.write_to_file(filepath)
        return filename

    except Exception as e:
        print("TTS ERROR:", e)
        return None


# ---------------- ROUTE ----------------

@app.route("/audio/<filename>")
def serve_audio(filename):
    return send_from_directory(AUDIO_FOLDER, filename)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user = data.get("user", "gail")
    message = data.get("message", "")

    reply = generate_reply(user, message)
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
