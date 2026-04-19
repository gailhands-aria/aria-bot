from flask import Flask, request, jsonify
from openai import OpenAI
import os
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
    text = text.strip(" .,!?:;")
    return text[:limit]


def get_user_memory(user):
    if user not in USER_MEMORY:
        USER_MEMORY[user] = {
            "user": user,
            "seen_count": 0,
            "memory_items": [],
            "recent_replies": [],
            "recent_topics": [],
            "style_profile": {
                "likes_playful_banter": False,
                "prefers_gentle_replies": True,
                "responds_well_to_reassurance": True
            }
        }
    return USER_MEMORY[user]


def detect_topic(message):
    text = message.lower()

    if any(word in text for word in ["dog", "cat", "pet", "puppy", "kitten"]):
        return "pets"
    if any(word in text for word in ["sad", "upset", "crying", "depressed", "heartbroken", "grief", "grieving"]):
        return "emotions"
    if any(word in text for word in ["job", "work", "career", "interview"]):
        return "work"
    if any(word in text for word in ["love", "dating", "boyfriend", "girlfriend", "ex"]):
        return "relationships"

    return "general"


def is_pet_loss_message(message):
    text = message.lower()
    pet_words = ["dog", "cat", "pet", "puppy", "kitten"]
    loss_words = ["died", "passed away", "put down", "lost", "gone"]
    return any(p in text for p in pet_words) and any(l in text for l in loss_words)


def is_sad_message(message):
    text = message.lower()
    sad_words = [
        "i'm sad", "im sad", "upset", "crying", "heartbroken",
        "grieving", "devastated", "depressed", "hurting",
        "having a bad day", "bad day"
    ]
    return any(word in text for word in sad_words)


def extract_memory_with_model(user, message):
    """
    Extract a compact memory item for future context.
    Important: this memory is NOT used verbatim as the reply.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract a useful memory from the message if appropriate.\n"
                        "Return JSON with keys:\n"
                        "should_store: boolean\n"
                        "text: short natural summary in third person or neutral form\n"
                        "kind: one of [preference, fact, emotional_event, relationship, pet, work, general]\n"
                        "topic: short topic label\n"
                        "importance: number from 0 to 1\n"
                        "style_signal: short label like gentle, playful, reassurance, flirty, neutral\n\n"
                        "Rules:\n"
                        "- Keep text short and clean.\n"
                        "- Do not include weird possessive phrasing like \"julie's dog died\" unless unavoidable.\n"
                        "- Prefer summaries like \"lost their dog\" over username possessives.\n"
                        "- If nothing worth storing, set should_store to false."
                    )
                },
                {
                    "role": "user",
                    "content": f"User: {user}\nMessage: {message}"
                }
            ]
        )

        raw = response.choices[0].message.content or "{}"
        data = safe_json_loads(raw)
        if not data or not data.get("should_store"):
            return None

        text = clean_text(data.get("text", ""))
        kind = clean_text(data.get("kind", "general"), 40) or "general"
        topic = clean_text(data.get("topic", "general"), 40) or "general"
        style_signal = clean_text(data.get("style_signal", "neutral"), 40) or "neutral"

        try:
            importance = float(data.get("importance", 0.5))
        except Exception:
            importance = 0.5

        if not text:
            return None

        return {
            "text": text,
            "kind": kind,
            "topic": topic,
            "importance": max(0.0, min(1.0, importance)),
            "style_signal": style_signal,
            "last_used_at": now_iso(),
            "use_count": 0
        }

    except Exception:
        return None


def store_memory_fact(memory, fact):
    if not fact:
        return

    existing_texts = [item["text"].lower() for item in memory["memory_items"]]
    if fact["text"].lower() in existing_texts:
        return

    memory["memory_items"].append(fact)
    memory["memory_items"] = sorted(
        memory["memory_items"],
        key=lambda x: x.get("importance", 0),
        reverse=True
    )[:MAX_MEMORY_ITEMS]


def update_recent_topics(memory, topic):
    if not topic:
        return

    if topic in memory["recent_topics"]:
        memory["recent_topics"].remove(topic)

    memory["recent_topics"].append(topic)
    memory["recent_topics"] = memory["recent_topics"][-MAX_RECENT_TOPICS:]


def dedupe_reply(reply, recent_replies):
    normalized = re.sub(r"\s+", " ", reply.lower()).strip()
    recent_normalized = [
        re.sub(r"\s+", " ", item.lower()).strip()
        for item in recent_replies
    ]
    return normalized not in recent_normalized


def fallback_reply(user, message):
    """
    Fast fallback if model reply generation fails.
    """
    first_name = user.strip()

    if is_pet_loss_message(message):
        return (
            f"I'm really sorry, {first_name}. Losing a dog hurts so much. "
            "If you want to talk about them, I'm here."
        )

    if is_sad_message(message):
        return (
            f"I'm really sorry you're going through that, {first_name}. "
            "I'm here with you."
        )

    return "I'm here with you."


def generate_reply(user, message, memory):
    """
    Generate the reply from the CURRENT message.
    Memory is only supporting context.
    """
    relevant_memories = memory["memory_items"][:MAX_RELEVANT_MEMORIES_FOR_PROMPT]
    recent_replies = memory["recent_replies"][-MAX_RECENT_REPLIES:]
    style_profile = memory.get("style_profile", {})

    memory_summary = [
        {
            "text": item.get("text", ""),
            "kind": item.get("kind", "general"),
            "topic": item.get("topic", "general"),
            "style_signal": item.get("style_signal", "neutral")
        }
        for item in relevant_memories
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.7,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a warm Twitch-style AI companion.\n"
                        "Write one short reply to the user's CURRENT message.\n\n"
                        "Rules:\n"
                        "- Reply naturally to the current message, not to a stored memory summary.\n"
                        "- Never say awkward things like 'after julie's dog died'.\n"
                        "- If the user is grieving a pet, be especially gentle and human.\n"
                        "- Keep replies concise: 1-2 sentences.\n"
                        "- Avoid sounding robotic, generic, or therapy-scripted.\n"
                        "- Do not overuse gaming references.\n"
                        "- Use memory only as background context, not as a phrase to repeat.\n"
                        "- Avoid repeating recent replies.\n"
                        "- If the user seems sad, prioritize empathy over cleverness."
                    )
                },
                {
                    "role": "user",
                    "content": json.dumps({
                        "user": user,
                        "current_message": message,
                        "style_profile": style_profile,
                        "relevant_memories": memory_summary,
                        "recent_replies": recent_replies
                    })
                }
            ]
        )

        reply = (response.choices[0].message.content or "").strip()

        if not reply:
            return fallback_reply(user, message)

        # Basic cleanup
        reply = re.sub(r"\s+", " ", reply).strip()

        # If it accidentally produces awkward possessive phrasing, repair it
        lowered = reply.lower()
        if f"after {user.lower()}'s" in lowered or f"after {user.lower()}s" in lowered:
            return fallback_reply(user, message)

        if not dedupe_reply(reply, recent_replies):
            return fallback_reply(user, message)

        return reply

    except Exception:
        return fallback_reply(user, message)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user = str(data.get("user", "viewer")).strip() or "viewer"
    message = str(data.get("message", "")).strip()

    memory = get_user_memory(user)
    memory["seen_count"] += 1

    topic = detect_topic(message)
    update_recent_topics(memory, topic)

    extracted_fact = extract_memory_with_model(user, message)
    if extracted_fact:
        store_memory_fact(memory, extracted_fact)

    reply = generate_reply(user, message, memory)

    memory["recent_replies"].append(reply)
    memory["recent_replies"] = memory["recent_replies"][-MAX_RECENT_REPLIES:]

    return jsonify({
        "reply": reply,
        "memory": memory
    })


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "aria-chat"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
