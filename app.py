from flask import Flask, request, jsonify
from openai import OpenAI
import os
import random
import json
import re
from datetime import datetime, timezone

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------------------------------------------
# TEMP IN-MEMORY STORE
# NOTE: resets on restart/redeploy
# -------------------------------------------------------------------
USER_MEMORY = {}

MAX_MEMORY_ITEMS = 12
MAX_RECENT_REPLIES = 4
MAX_RECENT_TOPICS = 6
MAX_RELEVANT_MEMORIES_FOR_PROMPT = 3


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
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
    return text[:limit]


def get_user_memory(user):
    if user not in USER_MEMORY:
        USER_MEMORY[user] = {
            "user": user,
            "seen_count": 0,
            "last_seen_at": None,
            "last_vibe": None,
            "last_topic": None,
            "style_profile": {
                "likes_playful_banter": False,
                "responds_well_to_reassurance": False,
                "prefers_gentle_replies": False
            },
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
    if any(x in t for x in ["excited", "omg", "yay", "so happy", "buzzing", "can't wait", "feel great", "feeling great"]):
        return "excited"
    if "?" in t:
        return "curious"
    return "neutral"


def detect_topics(text):
    t = text.lower()
    topics = []

    topic_map = {
        "job": ["job", "interview", "hired", "work", "career", "boss", "shift", "role", "offer"],
        "pets": ["dog", "cat", "pet", "puppy", "kitten"],
        "relationship": ["boyfriend", "girlfriend", "ex", "dating", "relationship", "husband", "wife"],
        "health": ["ill", "sick", "doctor", "hospital", "pain", "therapy", "mental health"],
        "family": ["mum", "mom", "dad", "sister", "brother", "family"],
        "stream": ["stream", "twitch", "chat", "live", "viewer"],
        "gaming": ["game", "gaming", "fortnite", "minecraft", "valorant", "cod"],
        "sleep": ["sleep", "tired", "insomnia", "exhausted", "awake"],
        "school": ["exam", "university", "school", "college", "study", "class"],
        "money": ["money", "rent", "bill", "debt", "paid", "salary"]
    }

    for topic, keywords in topic_map.items():
        if any(word in t for word in keywords):
            topics.append(topic)

    return topics


def update_recent_topics(memory, topics):
    for topic in topics:
        if topic not in memory["recent_topics"]:
            memory["recent_topics"].append(topic)
    memory["recent_topics"] = memory["recent_topics"][-MAX_RECENT_TOPICS:]
    if topics:
        memory["last_topic"] = topics[-1]


def update_recent_replies(memory, reply):
    if reply:
        memory["recent_replies"].append(reply.strip())
        memory["recent_replies"] = memory["recent_replies"][-MAX_RECENT_REPLIES:]


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


def normalize_memory_kind(kind):
    allowed = {
        "achievement",
        "loss",
        "emotional_event",
        "life_event",
        "preference",
        "relationship",
        "pet",
        "plan",
        "general"
    }
    return kind if kind in allowed else "general"


def clamp_importance(value):
    try:
        value = float(value)
    except Exception:
        return 0.5
    return max(0.0, min(1.0, value))


def is_duplicate_fact(existing_items, new_text):
    new_text_lower = new_text.lower().strip()
    for item in existing_items:
        old = item.get("text", "").lower().strip()
        if new_text_lower == old:
            return True
        if new_text_lower in old or old in new_text_lower:
            return True
    return False


def store_memory_fact(memory, fact):
    if not fact:
        return

    if not is_duplicate_fact(memory["memory_items"], fact["text"]):
        memory["memory_items"].append(fact)

    memory["memory_items"] = sorted(
        memory["memory_items"],
        key=lambda x: x.get("importance", 0),
        reverse=True
    )[:MAX_MEMORY_ITEMS]


def extract_memory_with_model(user, message):
    """
    Ask the model whether the message contains something worth remembering.
    Returns a dict or None.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            top_p=1,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": """
You extract memory for a Twitch AI streamer.

Decide whether the viewer message contains something worth remembering for future chats.

Store ONLY things likely to matter later, such as:
- achievements
- losses
- worries or emotional situations
- plans
- important life updates
- recurring preferences
- pet updates
- relationship updates

Do NOT store:
- greetings
- filler chat
- generic questions
- small talk with no future value
- repeated versions of the same unimportant thing

Return JSON only with:
{
  "should_store": true or false,
  "fact": "short reusable memory sentence",
  "kind": "achievement | loss | emotional_event | life_event | preference | relationship | pet | plan | general",
  "topic": "short topic label",
  "importance": 0.0 to 1.0,
  "style_signal": "gentle | playful | reassuring | neutral"
}

Rules:
- Write fact as a short third-person memory sentence, like:
  "They got the job"
  "Their dog died"
  "They were nervous about an interview"
- Be specific, not vague
- If nothing meaningful should be stored, return should_store false
- Never include extra keys
"""
                },
                {
                    "role": "user",
                    "content": f"Viewer: {user}\nMessage: {message}"
                }
            ]
        )

        raw = response.choices[0].message.content or "{}"
        data = safe_json_loads(raw)
        if not data or not data.get("should_store"):
            return None

        fact_text = clean_text(data.get("fact", ""))
        if not fact_text:
            return None

        return {
            "text": fact_text,
            "kind": normalize_memory_kind(data.get("kind")),
            "topic": clean_text(data.get("topic", ""), 40) or None,
            "importance": clamp_importance(data.get("importance", 0.6)),
            "style_signal": data.get("style_signal", "neutral"),
            "created_at": now_iso(),
            "last_used_at": None,
            "use_count": 0
        }

    except Exception:
        return None


def update_style_profile(memory, style_signal, vibe):
    if style_signal == "playful":
        memory["style_profile"]["likes_playful_banter"] = True
    if style_signal == "reassuring":
        memory["style_profile"]["responds_well_to_reassurance"] = True
    if style_signal == "gentle" or vibe in ["sad", "stressed"]:
        memory["style_profile"]["prefers_gentle_replies"] = True


def choose_relevant_memories(memory, message, explicit_recall=False):
    items = memory.get("memory_items", [])
    if not items:
        return []

    current_topics = detect_topics(message)
    current_vibe = detect_vibe(message)
    greeting = is_greeting(message)

    scored = []

    for item in items:
        score = item.get("importance", 0.0)

        if explicit_recall:
            score += 1.3

        if current_topics and item.get("topic") in current_topics:
            score += 0.8

        if greeting and item.get("kind") in ["achievement", "loss", "emotional_event", "life_event", "plan"]:
            score += 0.4

        if current_vibe == "excited" and item.get("kind") == "achievement":
            score += 0.9

        if current_vibe in ["sad", "stressed"] and item.get("kind") in ["loss", "emotional_event"]:
            score += 1.0

        if current_vibe == "flirty" and memory["style_profile"].get("likes_playful_banter"):
            score += 0.2

        score -= item.get("use_count", 0) * 0.28

        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:MAX_RELEVANT_MEMORIES_FOR_PROMPT]]


def mark_memories_used(memories):
    for item in memories:
        item["use_count"] = item.get("use_count", 0) + 1
        item["last_used_at"] = now_iso()


def build_recall_reply(memory):
    relevant = choose_relevant_memories(memory, "", explicit_recall=True)
    if not relevant:
        if memory.get("last_topic"):
            return f"I remember we were talking about {memory['last_topic']}, but I don’t have the exact detail locked in properly yet."
        return "I remember you, but I don’t have a strong specific detail saved yet."

    if len(relevant) == 1:
        fact = relevant[0]["text"]
        mark_memories_used([relevant[0]])
        return random.choice([
            f"Yeah — you told me {fact[0].lower() + fact[1:]}",
            f"I do — you said {fact[0].lower() + fact[1:]}",
            f"Yeah, I remember — {fact[0].lower() + fact[1:]}"
        ])

    first = relevant[0]["text"]
    second = relevant[1]["text"]
    mark_memories_used(relevant[:2])

    return random.choice([
        f"Yeah — you told me {first[0].lower() + first[1:]}. You also said {second[0].lower() + second[1:]}",
        f"I remember a couple of things actually — {first[0].lower() + first[1:]}, and later {second[0].lower() + second[1:]}",
        f"Yeah — I remember {first[0].lower() + first[1:]}, and also {second[0].lower() + second[1:]}"
    ])


def build_memory_summary(memory, relevant_memories):
    parts = []

    if memory.get("seen_count", 0) > 1:
        parts.append(f"Returning viewer seen {memory['seen_count']} times.")

    if memory.get("last_vibe"):
        parts.append(f"Last emotional tone from them: {memory['last_vibe']}.")

    if memory.get("recent_topics"):
        parts.append("Recent topics: " + ", ".join(memory["recent_topics"][-3:]) + ".")

    style_profile = memory.get("style_profile", {})
    style_flags = []
    if style_profile.get("likes_playful_banter"):
        style_flags.append("likes playful banter")
    if style_profile.get("responds_well_to_reassurance"):
        style_flags.append("responds well to reassurance")
    if style_profile.get("prefers_gentle_replies"):
        style_flags.append("often needs a gentle tone")
    if style_flags:
        parts.append("Style hints: " + ", ".join(style_flags) + ".")

    if relevant_memories:
        facts = " | ".join(item["text"] for item in relevant_memories)
        parts.append(f"Relevant remembered facts for this message: {facts}.")

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

    # ---------------------------------------------------------------
    # MODEL-BASED MEMORY EXTRACTION
    # ---------------------------------------------------------------
    extracted_fact = extract_memory_with_model(user, message)
    if extracted_fact:
        store_memory_fact(memory, extracted_fact)
        update_style_profile(memory, extracted_fact.get("style_signal", "neutral"), vibe)

    # ---------------------------------------------------------------
    # DIRECT RECALL REQUESTS
    # ---------------------------------------------------------------
    if is_recall_request(message):
        reply = build_recall_reply(memory)
        update_recent_replies(memory, reply)
        return jsonify({
            "reply": reply,
            "memory": memory
        })

    relevant_memories = choose_relevant_memories(memory, message, explicit_recall=False)

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
    memory_summary = build_memory_summary(memory, relevant_memories)

    # ---------------------------------------------------------------
    # MAIN REPLY
    # ---------------------------------------------------------------
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
- If there is a relevant remembered fact, use it naturally.
- Do not list memories mechanically.
- Do not sound like you are reading notes.
- Only mention memory if it genuinely fits the current message.
- If the user sounds happy and there is a relevant positive memory, it can be natural to connect them.
- If the user sounds low and there is a relevant sad memory, it can be natural to connect them gently.
- Never say "your last topic was" or "I have stored".
- Never mention the word "vibe".
- Never explain your reasoning.

Examples of natural memory use:
- "Wait — you're the one who got the job, right?"
- "Honestly I'd be in a good mood too after that."
- "How did that thing end up going?"
- "You've had a rough couple of days lately, haven't you?"
- "Aw love... is this about what happened before?"

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

    # mark only the relevant memories we fed into the prompt
    if relevant_memories:
        mark_memories_used(relevant_memories)

    update_recent_replies(memory, reply)

    return jsonify({
        "reply": reply,
        "memory": memory
    })


@app.route("/memory/<user>", methods=["GET"])
def view_memory(user):
    memory = get_user_memory(user)
    return jsonify(memory)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
