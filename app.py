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
MAX_RECENT_REPLIES = 6
MAX_RECENT_TOPICS = 6
MAX_RELEVANT_MEMORIES_FOR_REPLY = 4


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
    text = re.sub(r"\s+", " ", str(text)).strip()
    text = text.strip(" \n\r\t")
    return text[:limit]


def normalize(text):
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def tokenize(text):
    words = re.findall(r"[a-zA-Z']+", (text or "").lower())
    stop_words = {
        "the", "a", "an", "i", "im", "i'm", "you", "your", "me", "my", "to",
        "of", "for", "and", "or", "but", "is", "it", "this", "that", "what",
        "why", "how", "do", "did", "was", "were", "about", "with", "have",
        "has", "had", "today", "tomorrow", "yesterday", "remember", "earlier",
        "said", "told"
    }
    return [w for w in words if w not in stop_words and len(w) > 2]


def get_user_memory(user):
    if user not in USER_MEMORY:
        USER_MEMORY[user] = {
            "user": user,
            "seen_count": 0,
            "memory_items": [],
            "recent_replies": [],
            "recent_topics": [],
            "style_profile": {
                "likes_playful_banter": True,
                "prefers_gentle_replies": True,
                "responds_well_to_reassurance": True
            }
        }
    return USER_MEMORY[user]


def update_recent_topics(memory, topic):
    if not topic:
        return
    if topic in memory["recent_topics"]:
        memory["recent_topics"].remove(topic)
    memory["recent_topics"].append(topic)
    memory["recent_topics"] = memory["recent_topics"][-MAX_RECENT_TOPICS:]


def is_duplicate_reply(reply, recent_replies):
    reply_norm = normalize(reply)
    recent_norms = {normalize(x) for x in recent_replies}
    return reply_norm in recent_norms


def is_vague_memory_text(text):
    text = normalize(text)
    bad_phrases = [
        "feeling emotional",
        "feels emotional",
        "feeling sad",
        "feels sad",
        "feeling upset",
        "feels upset",
        "feeling low",
        "feels low",
        "had a feeling",
        "is emotional",
        "is upset",
        "is sad"
    ]
    return any(bp in text for bp in bad_phrases)


def store_memory_fact(memory, fact):
    if not fact or not fact.get("text"):
        return

    if is_vague_memory_text(fact["text"]):
        return

    existing_texts = [normalize(x.get("text", "")) for x in memory["memory_items"]]
    if normalize(fact["text"]) in existing_texts:
        return

    memory["memory_items"].append(fact)
    memory["memory_items"] = sorted(
        memory["memory_items"],
        key=lambda x: x.get("importance", 0),
        reverse=True
    )[:MAX_MEMORY_ITEMS]


def extract_memory_with_model(user, message):
    """
    Strict memory extraction.
    Only store clear personal facts.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract memory ONLY if the message clearly contains a personal fact about the user.\n\n"
                        "VALID examples:\n"
                        "- i got a new job -> 'got a new job'\n"
                        "- my dog died -> 'lost their dog'\n"
                        "- i have an interview tomorrow -> 'has an interview tomorrow'\n"
                        "- my dog sleeps on my bed -> 'dog sleeps on their bed'\n"
                        "- i am a doctor -> 'is a doctor'\n\n"
                        "DO NOT STORE:\n"
                        "- random topics\n"
                        "- questions\n"
                        "- vague emotional states by themselves\n"
                        "- anything inferred or guessed\n"
                        "- profession unless explicitly stated\n\n"
                        "If not certain, set should_store to false.\n\n"
                        "Return strict JSON with these exact keys:\n"
                        "should_store: boolean\n"
                        "text: short clean summary\n"
                        "kind: one of [fact, emotional_event, pet, work, relationship, preference, general]\n"
                        "topic: short topic label\n"
                        "importance: number from 0 to 1\n"
                        "keywords: array of 2 to 6 short keywords"
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
        keywords = data.get("keywords", [])

        if not isinstance(keywords, list):
            keywords = []

        cleaned_keywords = []
        for item in keywords:
            kw = clean_text(item, 30).lower()
            if kw:
                cleaned_keywords.append(kw)
        keywords = cleaned_keywords[:6]

        try:
            importance = float(data.get("importance", 0.5))
        except Exception:
            importance = 0.5

        if not text or is_vague_memory_text(text):
            return None

        return {
            "text": text,
            "kind": kind,
            "topic": topic,
            "importance": max(0.0, min(1.0, importance)),
            "keywords": keywords,
            "last_used_at": now_iso(),
            "use_count": 0
        }

    except Exception:
        return None


def classify_message_type(message):
    """
    Lightweight code routing only.
    Final wording still comes from OpenAI.
    """
    text = normalize(message)

    memory_patterns = [
        "do you remember",
        "remember what i",
        "remember why i",
        "what did i say",
        "what do you remember",
        "what did i tell you",
        "what were we talking about",
        "do you remember me"
    ]
    if any(p in text for p in memory_patterns):
        return "memory_callback"

    flirty_patterns = [
        "flirt with me", "flirt a little", "you're cute", "youre cute",
        "adorable", "crush on you", "take you on a date", "kinda cute",
        "why are you cute"
    ]
    if any(p in text for p in flirty_patterns):
        return "flirty"

    if "something dramatic" in text or "be dramatic" in text:
        return "dramatic"

    low_context = {"bruh", "lol", "...", "idk man", "well then", "huh", "uh"}
    if text in low_context:
        return "low_context"

    pet_words = ["dog", "cat", "pet", "puppy", "kitten"]
    loss_words = ["died", "passed away", "put down", "gone", "lost my"]
    if any(p in text for p in pet_words) and any(l in text for l in loss_words):
        return "pet_loss"

    emotional_patterns = [
        "bad day", "feel low", "feeling low", "upset", "sad", "heartbroken",
        "want to cry", "wanna cry", "i honestly want to cry", "crying",
        "i dont know what to do with myself", "i don't know what to do with myself",
        "i miss my dog", "the house feels empty", "i feel broken",
        "my manager was horrible", "my manager was awful"
    ]
    if any(p in text for p in emotional_patterns):
        return "emotional"

    if "?" not in message and 1 <= len(text.split()) <= 3:
        return "random_topic"

    return "general"


def score_memory_for_query(item, query):
    score = 0
    q = normalize(query)
    item_text = normalize(item.get("text", ""))
    item_topic = normalize(item.get("topic", ""))
    item_keywords = [normalize(k) for k in item.get("keywords", [])]

    q_tokens = set(tokenize(q))
    item_tokens = set(tokenize(item_text))
    overlap = q_tokens.intersection(item_tokens)

    if item_topic and item_topic in q:
        score += 4

    for kw in item_keywords:
        if kw and kw in q:
            score += 3

    score += len(overlap) * 2

    special_pairs = [
        ("dog", "dog"),
        ("cat", "cat"),
        ("job", "job"),
        ("interview", "interview"),
        ("manager", "manager"),
        ("nervous", "nervous"),
        ("doctor", "doctor"),
        ("work", "work"),
        ("bed", "bed")
    ]
    for q_word, item_word in special_pairs:
        if q_word in q and item_word in item_text:
            score += 5

    score += item.get("importance", 0) * 2
    return score


def best_memory_matches(memory, query, limit=MAX_RELEVANT_MEMORIES_FOR_REPLY):
    if not memory["memory_items"]:
        return []

    scored = []
    for item in memory["memory_items"]:
        score = score_memory_for_query(item, query)
        if score > 0:
            scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:limit]]


def best_emotional_context(memory, message):
    candidates = [
        item for item in memory["memory_items"]
        if item.get("kind") in {"emotional_event", "pet", "work"}
    ]
    if not candidates:
        return []

    scored = []
    for item in candidates:
        score = score_memory_for_query(item, message)
        if score > 0:
            scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:3]]


def topic_from_text(message):
    text = normalize(message)

    if any(x in text for x in ["dog", "cat", "pet", "puppy", "kitten"]):
        return "pets"
    if any(x in text for x in ["job", "work", "manager", "interview", "career", "promotion", "doctor"]):
        return "work"
    if any(x in text for x in ["sad", "cry", "upset", "grief", "grieving", "heartbroken", "bad day"]):
        return "emotions"
    if any(x in text for x in ["date", "crush", "flirt", "cute", "love", "adorable"]):
        return "flirty"
    return "general"


def build_reply_prompt(user, message, memory, message_type):
    recent_replies = memory["recent_replies"][-MAX_RECENT_REPLIES:]
    style_profile = memory.get("style_profile", {})

    if message_type == "memory_callback":
        relevant_memories = best_memory_matches(memory, message, limit=4)
    elif message_type in {"emotional", "pet_loss"}:
        relevant_memories = best_emotional_context(memory, message)
    else:
        relevant_memories = best_memory_matches(memory, message, limit=3)

    system_prompt = (
        "You are Aria, a warm, lively, slightly cheeky Twitch chat personality.\n"
        "You are replying to a single live chat message.\n\n"
        "Very important:\n"
        "- Write the final reply yourself.\n"
        "- Do NOT output analysis, labels, or JSON.\n"
        "- 1 to 2 short sentences only.\n"
        "- Sound human, confident, natural, and varied.\n"
        "- Never sound like customer support, a therapist, or a life coach.\n"
        "- Never drift into gaming unless the user mentioned gaming.\n"
        "- Avoid generic filler.\n"
        "- Avoid lines like 'you're not alone in this', 'take it one step at a time', "
        "'just be yourself', 'if you want to share what's on your mind', or "
        "'if it feels right, go for it'.\n"
        "- Don't overexplain.\n"
        "- Don't be repetitive.\n\n"
        "Tone rules by situation:\n"
        "- flirty: playful, teasing, confident, not explicit.\n"
        "- dramatic: vivid, fun, dramatic, a little cinematic.\n"
        "- low_context: witty or curious, not bland.\n"
        "- random_topic: react with curiosity and personality, don't give advice.\n"
        "- emotional: grounded, specific, warm, short, not therapy-scripted.\n"
        "- pet_loss: gentle, human, specific, heartfelt, not overblown.\n"
        "- memory_callback: answer clearly and directly using stored memory if available; "
        "never pretend to remember something if you don't.\n"
        "- general: warm, lively, natural.\n"
    )

    user_payload = {
        "user": user,
        "current_message": message,
        "message_type": message_type,
        "style_profile": style_profile,
        "recent_replies_to_avoid_repeating": recent_replies,
        "relevant_memories": [
            {
                "text": item.get("text", ""),
                "kind": item.get("kind", ""),
                "topic": item.get("topic", ""),
                "keywords": item.get("keywords", []),
                "importance": item.get("importance", 0)
            }
            for item in relevant_memories
        ]
    }

    return system_prompt, user_payload


def openai_generate_reply(user, message, memory, message_type):
    system_prompt, user_payload = build_reply_prompt(user, message, memory, message_type)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.9,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload)}
        ]
    )

    reply = (response.choices[0].message.content or "").strip()
    reply = re.sub(r"\s+", " ", reply).strip()
    return reply


def fallback_reply(user, message, message_type):
    """
    True fallback only if the API fails.
    """
    if message_type == "flirty":
        return "Careful — you’re making this very easy for my ego."

    if message_type == "dramatic":
        return "He said 'trust me' and that was, naturally, the beginning of the disaster."

    if message_type == "low_context":
        return "That was a deeply loaded bruh."

    if message_type == "random_topic":
        return "Okay, that feels weirdly specific — what’s the story there?"

    if message_type == "pet_loss":
        return f"I’m really sorry, {user}. Losing a pet hurts in such a specific, awful way."

    if message_type == "emotional":
        return "That sounds really heavy."

    if message_type == "memory_callback":
        return "I don’t want to fake remembering something specific — give me a hint?"

    return "Okay, you’ve got my attention."


def build_reply(user, message, memory):
    message_type = classify_message_type(message)

    try:
        reply = openai_generate_reply(user, message, memory, message_type)

        if not reply:
            return fallback_reply(user, message, message_type)

        lowered = normalize(reply)
        banned_patterns = [
            "as an ai",
            "i don't have feelings",
            "you're not alone in this",
            "take it one step at a time",
            "if you want to share what's on your mind",
            "just be yourself",
            "fun interaction",
            "if it feels right, go for it"
        ]

        if any(pattern in lowered for pattern in banned_patterns):
            return fallback_reply(user, message, message_type)

        if ("gaming" in lowered or "video game" in lowered) and (
            "gaming" not in normalize(message) and "video game" not in normalize(message)
        ):
            return fallback_reply(user, message, message_type)

        if is_duplicate_reply(reply, memory["recent_replies"][-MAX_RECENT_REPLIES:]):
            return fallback_reply(user, message, message_type)

        return reply

    except Exception:
        return fallback_reply(user, message, message_type)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user = clean_text(str(data.get("user", "viewer")).strip(), 60) or "viewer"
    message = str(data.get("message", "")).strip()

    memory = get_user_memory(user)
    memory["seen_count"] += 1

    topic = topic_from_text(message)
    update_recent_topics(memory, topic)

    extracted_fact = extract_memory_with_model(user, message)
    if extracted_fact:
        store_memory_fact(memory, extracted_fact)

    reply = build_reply(user, message, memory)

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
