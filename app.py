from flask import Flask, request, jsonify
from openai import OpenAI
import os
import random
import re
from datetime import datetime, timezone

app = Flask(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------------------------------------------
# LIGHTWEIGHT IN-MEMORY USER MEMORY
# NOTE:
# This resets when the app restarts/redeploys.
# Later we can move this to Render Postgres or Key Value.
# -------------------------------------------------------------------
USER_MEMORY = {}

MAX_FACTS = 10
MAX_RECENT_REPLIES = 4
MAX_RECENT_TOPICS = 6


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def now_iso():
    return datetime.now(timezone.utc).isoformat()


def get_user_memory(user):
    if user not in USER_MEMORY:
        USER_MEMORY[user] = {
            "seen_count": 0,
            "last_seen_at": None,
            "last_vibe": None,
            "last_topic": None,
            "notes": "",
            "memory_items": [],
            "recent_topics": [],
            "recent_replies": []
        }
    return USER_MEMORY[user]


def detect_vibe(text):
    t = text.lower()

    if any(x in t for x in ["bad day", "sad", "hurt", "upset", "crying", "heartbroken", "devastated", "low"]):
        return "sad"
    if any(x in t for x in ["stressed", "overwhelmed", "anxious", "nervous", "panic", "worried"]):
        return "stressed"
    if any(x in t for x in ["frustrated", "annoyed", "angry", "irritated", "fed up"]):
        return "frustrated"
    if any(x in t for x in ["lol", "lmao", "haha", "funny", "joke"]):
        return "funny"
    if any(x in t for x in ["cute", "hot", "kiss", "date", "love you", "flirty", "cheeky"]):
        return "flirty"
    if any(x in t for x in ["excited", "omg", "yay", "so happy", "buzzing", "can't wait"]):
        return "excited"
    if "?" in t:
        return "curious"
    return "neutral"


def detect_topics(text):
    t = text.lower()
    topics = []

    topic_map = {
        "job": ["job", "interview", "hired", "work", "career", "boss", "shift"],
        "bad_day": ["bad day", "rough day", "sad", "upset", "crying", "drained"],
        "sleep": ["sleep", "tired", "insomnia", "exhausted", "awake"],
        "pet": ["dog", "cat", "pet", "puppy", "kitten"],
        "relationship": ["boyfriend", "girlfriend", "ex", "dating", "relationship"],
        "stream": ["stream", "twitch", "chat", "live", "viewer"],
        "gaming": ["game", "gaming", "fortnite", "cod", "minecraft", "valorant"],
        "health": ["ill", "sick", "doctor", "hospital", "pain"],
        "family": ["mum", "mom", "dad", "sister", "brother", "family"]
    }

    for topic, keywords in topic_map.items():
        if any(word in t for word in keywords):
            topics.append(topic)

    return topics


def is_greeting(text):
    t = text.lower().strip()
    simple = {
        "hi", "hey", "hello", "yo", "hiya", "sup", "heyy",
        "hey aria", "hi aria", "hello aria", "yo aria"
    }
    return t in simple


def is_recall_request(text):
    t = text.lower()
    patterns = [
        r"\bdo you remember\b",
        r"\bremember what i told you\b",
        r"\bwhat did i tell you\b",
        r"\bwhat did i say\b",
        r"\bwhat was i talking about\b",
        r"\bdo you remember what i said\b",
        r"\bremember that\b"
    ]
    return any(re.search(p, t) for p in patterns)


def clean_fact_text(text):
    cleaned = text.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned[:220]


def fact_kind_and_importance(text):
    t = text.lower()

    # Stronger life facts first
    if any(x in t for x in ["i got the job", "i got hired", "i got accepted", "i passed"]):
        return "achievement", 0.97

    if any(x in t for x in ["i have an interview", "i had an interview", "job interview"]):
        return "life_event", 0.90

    if any(x in t for x in ["i had a bad day", "i'm having a bad day", "i feel awful", "i'm really sad"]):
        return "emotional_event", 0.92

    if any(x in t for x in ["my dog", "my cat", "my pet"]):
        return "personal", 0.78

    if any(x in t for x in ["my ex", "my boyfriend", "my girlfriend"]):
        return "relationship", 0.82

    if any(x in t for x in ["tomorrow i", "next week i", "i'm going to", "i plan to"]):
        return "plan", 0.76

    if any(x in t for x in ["i'm nervous", "i'm anxious", "i'm stressed", "i'm worried"]):
        return "emotional_event", 0.80

    if any(x in t for x in ["i got", "i have", "i had", "i'm", "i am", "my "]):
        return "general_personal", 0.58

    return None, 0.0


def extract_memory_fact(message):
    raw = message.strip()
    if len(raw) < 8:
        return None

    kind, importance = fact_kind_and_importance(raw)
    if not kind:
        return None

    # Skip low-value tiny chatter
    low_value_starts = [
        "i am here", "i'm here", "i'm bored", "my name is", "i'm hungry"
    ]
    if raw.lower() in low_value_starts:
        return None

    return {
        "text": clean_fact_text(raw),
        "kind": kind,
        "importance": importance,
        "created_at": now_iso(),
        "last_used_at": None,
        "use_count": 0
    }


def is_duplicate_fact(existing_items, new_text):
    new_text_lower = new_text.lower().strip()
    for item in existing_items:
        old = item.get("text", "").lower().strip()
        if new_text_lower == old:
            return True
        if new_text_lower in old or old in new_text_lower:
            return True
    return False


def store_memory_fact(memory, fact, topics):
    if not fact:
        return

    fact["topic"] = topics[0] if topics else None

    if not is_duplicate_fact(memory["memory_items"], fact["text"]):
        memory["memory_items"].append(fact)

    memory["memory_items"] = sorted(
        memory["memory_items"],
        key=lambda x: x.get("importance", 0),
        reverse=True
    )[:MAX_FACTS]


def update_recent_topics(memory, topics):
    for topic in topics:
        if topic not in memory["recent_topics"]:
            memory["recent_topics"].append(topic)
    memory["recent_topics"] = memory["recent_topics"][-MAX_RECENT_TOPICS:]
    if topics:
        memory["last_topic"] = topics[-1]


def update_recent_replies(memory, reply):
    if not reply:
        return
    memory["recent_replies"].append(reply.strip())
    memory["recent_replies"] = memory["recent_replies"][-MAX_RECENT_REPLIES:]


def choose_best_memory(memory, message, explicit_recall=False):
    items = memory.get("memory_items", [])
    if not items:
        return None

    current_topics = detect_topics(message)
    greeting = is_greeting(message)

    best = None
    best_score = -999

    for item in items:
        score = item.get("importance", 0)

        if explicit_recall:
            score += 1.2

        if current_topics and item.get("topic") in current_topics:
            score += 0.8

        if greeting and item.get("kind") in ["achievement", "life_event", "emotional_event", "plan"]:
            score += 0.45

        score -= item.get("use_count", 0) * 0.28

        if score > best_score:
            best_score = score
            best = item

    return best


def mark_memory_used(item):
    if not item:
        return
    item["use_count"] = item.get("use_count", 0) + 1
    item["last_used_at"] = now_iso()


def build_recall_reply(memory):
    items = memory.get("memory_items", [])
    if not items:
        if memory.get("last_topic"):
            return f"I remember we were talking about {memory['last_topic']}, but I don’t have the exact detail locked in properly yet."
        return "I remember you, but I don’t have a strong specific detail saved yet."

    top_items = items[:2]

    if len(top_items) == 1:
        mark_memory_used(top_items[0])
        return random.choice([
            f"Yeah — you told me {top_items[0]['text'][0].lower() + top_items[0]['text'][1:]}",
            f"I do — you said {top_items[0]['text'][0].lower() + top_items[0]['text'][1:]}",
            f"Yeah, I remember — {top_items[0]['text'][0].lower() + top_items[0]['text'][1:]}"
        ])

    mark_memory_used(top_items[0])
    mark_memory_used(top_items[1])

    first = top_items[0]["text"]
    second = top_items[1]["text"]

    return random.choice([
        f"Yeah — you told me {first[0].lower() + first[1:]}. You also said {second[0].lower() + second[1:]}",
        f"I remember a couple of things actually — {first[0].lower() + first[1:]}, and later {second[0].lower() + second[1:]}",
        f"Yeah — I remember {first[0].lower() + first[1:]}, and also {second[0].lower() + second[1:]}"
    ])


def build_memory_summary(memory, chosen_memory=None):
    parts = []

    if memory.get("seen_count", 0) > 1:
        parts.append(f"Returning viewer seen {memory['seen_count']} times.")

    if memory.get("last_vibe"):
        parts.append(f"Last emotional tone from them: {memory['last_vibe']}.")

    if memory.get("recent_topics"):
        parts.append("Recent topics: " + ", ".join(memory["recent_topics"][-3:]) + ".")

    important = memory.get("memory_items", [])[:3]
    if important:
        facts = " | ".join(item["text"] for item in important)
        parts.append(f"Important remembered facts: {facts}.")

    if chosen_memory:
        parts.append(f"Most relevant memory for this message: {chosen_memory['text']}.")

    if memory.get("recent_replies"):
        parts.append("Avoid sounding too similar to these recent Aria replies: " + " | ".join(memory["recent_replies"]) + ".")

    return " ".join(parts)


# -------------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------------
@app.route("/")
def home():
    return "Aria is alive!"


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user = str(data.get("user", "viewer")).strip() or "viewer"
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"reply": "Say something to me 😌"})

    memory = get_user_memory(user)
    memory["seen_count"] += 1
    memory["last_seen_at"] = now_iso()

    vibe = detect_vibe(message)
    memory["last_vibe"] = vibe

    topics = detect_topics(message)
    update_recent_topics(memory, topics)

    fact = extract_memory_fact(message)
    store_memory_fact(memory, fact, topics)

    # Handle direct recall requests first
    if is_recall_request(message):
        reply = build_recall_reply(memory)
        update_recent_replies(memory, reply)
        return jsonify({"reply": reply})

    chosen_memory = choose_best_memory(memory, message, explicit_recall=False)

    styles = [
        "energetic and excited",
        "chill and relaxed",
        "slightly sarcastic",
        "playful and teasing",
        "warm and friendly"
    ]

    greeting_modes = [
        "Include a casual greeting and maybe the user's name",
        "No greeting, jump straight into response",
        "React first, greeting later in the sentence",
        "Be casual like you're mid-conversation already"
    ]

    style = random.choice(styles)
    greeting_mode = random.choice(greeting_modes)

    memory_summary = build_memory_summary(memory, chosen_memory)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.95,
        top_p=0.95,
        messages=[
            {
                "role": "system",
                "content": f"""
You are Aria, a live Twitch streamer.

IMPORTANT RULES:
- Talk like a real person, never like an AI assistant
- Do not default to gaming unless the user mentions gaming
- Follow the user's topic naturally
- Keep replies varied and not repetitive
- Sometimes greet, sometimes don't
- Sometimes use the user's name, sometimes don't
- Keep replies short and stream-friendly
- Use emojis only sometimes, not every reply

VERY IMPORTANT:
Before replying, figure out the emotional tone of the user's message and match it naturally.

Possible tones include:
- sad / hurt / disappointed
- stressed / overwhelmed
- happy / excited
- funny / joking
- flirty / cheeky
- curious / asking a normal question
- neutral / casual chat
- frustrated / annoyed

Tone rules:
- If the user seems sad, hurt, or says they had a bad day:
  be gentle, warm, comforting, and reassuring
  do not be sarcastic
  sound caring and human

- If the user is being funny or joking:
  laugh, play along, banter a little
  match their energy naturally

- If the user is being flirty or cheeky:
  be playful, confident, lightly teasing
  keep it tasteful and stream-safe

- If the user seems excited:
  match their energy and enthusiasm

- If the user seems frustrated:
  be understanding and grounded
  don't dismiss them

- If the message is casual:
  just chat naturally

MEMORY RULES:
- This user may be a returning viewer.
- Recognise returning users naturally, but do not force it every message.
- If there is a relevant remembered fact, you may reference it naturally.
- Do not list memories mechanically.
- Do not sound like you're reading notes.
- If there is a remembered fact and it fits naturally, weave it in like a real streamer would.
- Examples of natural memory use:
  "Wait — you're the one who got the job, right?"
  "How did that thing end up going?"
  "You've had a rough couple of days lately, haven't you?"
- Only reference memory if it feels relevant to the current message or greeting.
- If memory is not relevant, just reply normally.
- Never say "your last topic was" or "I have stored".
- Never mention the word "vibe".
- Never explain your reasoning.

STYLE CONTROL:
Current style: {style}
Greeting behaviour: {greeting_mode}

USER CONTEXT:
{memory_summary}
"""
            },
            {
                "role": "user",
                "content": f"{user}: {message}"
            }
        ]
    )

    reply = (response.choices[0].message.content or "").strip()

    if chosen_memory and chosen_memory["text"].lower() in reply.lower():
        mark_memory_used(chosen_memory)

    update_recent_replies(memory, reply)

    return jsonify({
        "reply": reply
    })


@app.route("/memory/<user>", methods=["GET"])
def view_memory(user):
    memory = get_user_memory(user)
    return jsonify(memory)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
