from flask import Flask, request, jsonify
from openai import OpenAI
import os
import json
import re
import random
from datetime import datetime, timezone

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

USER_MEMORY = {}

MAX_MEMORY_ITEMS = 15
MAX_RECENT_REPLIES = 5
MAX_RECENT_TOPICS = 6
MAX_RELEVANT_MEMORIES_FOR_PROMPT = 4


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
    text = text.strip(" \n\r\t.,!?;:")
    return text[:limit]


def first_cap(text):
    if not text:
        return text
    return text[0].upper() + text[1:]


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


def normalize(s):
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def tokenize(text):
    words = re.findall(r"[a-zA-Z']+", (text or "").lower())
    stop = {
        "the", "a", "an", "i", "im", "i'm", "you", "your", "yours", "my",
        "me", "to", "of", "for", "and", "or", "but", "is", "it", "this",
        "that", "what", "why", "how", "do", "did", "was", "were", "about",
        "remember", "earlier", "said", "told"
    }
    return [w for w in words if w not in stop and len(w) > 2]


def topic_from_text(message):
    text = normalize(message)

    if any(x in text for x in ["dog", "cat", "pet", "puppy", "kitten"]):
        return "pets"
    if any(x in text for x in ["job", "work", "manager", "interview", "career", "promotion"]):
        return "work"
    if any(x in text for x in ["sad", "cry", "upset", "grief", "grieving", "heartbroken", "bad day"]):
        return "emotions"
    if any(x in text for x in ["date", "crush", "flirt", "cute", "love", "adorable"]):
        return "flirty"
    return "general"


def update_recent_topics(memory, topic):
    if not topic:
        return
    if topic in memory["recent_topics"]:
        memory["recent_topics"].remove(topic)
    memory["recent_topics"].append(topic)
    memory["recent_topics"] = memory["recent_topics"][-MAX_RECENT_TOPICS:]


def is_duplicate_reply(reply, recent_replies):
    reply_norm = normalize(reply)
    return reply_norm in {normalize(x) for x in recent_replies}


def store_memory_fact(memory, fact):
    if not fact or not fact.get("text"):
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
    Store short, clean facts for future callbacks.
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
                        "Extract memory only if it could matter later.\n"
                        "Return JSON with:\n"
                        "should_store: boolean\n"
                        "text: short clean memory summary, without username possessive phrasing\n"
                        "kind: one of [fact, emotional_event, pet, work, relationship, preference, general]\n"
                        "topic: short topic label\n"
                        "importance: 0 to 1\n"
                        "keywords: array of 2-6 useful search keywords\n\n"
                        "Examples of good text:\n"
                        "- lost their dog\n"
                        "- has an interview tomorrow\n"
                        "- got a new job\n"
                        "- nervous about first day at work\n"
                        "- dog loves sleeping on the bed\n\n"
                        "Do not write awkward strings like 'julie's dog died'."
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

        keywords = [
            clean_text(str(k), 30).lower()
            for k in keywords
            if clean_text(str(k), 30)
        ][:6]

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
            "keywords": keywords,
            "last_used_at": now_iso(),
            "use_count": 0
        }

    except Exception:
        return None


def is_memory_question(message):
    text = normalize(message)
    patterns = [
        "do you remember",
        "remember what i",
        "remember why i",
        "what did i say",
        "what do you remember",
        "what did i tell you",
        "what were we talking about",
        "do you remember me"
    ]
    return any(p in text for p in patterns)


def score_memory_for_query(item, query):
    score = 0
    q = normalize(query)
    item_text = normalize(item.get("text", ""))
    item_topic = normalize(item.get("topic", ""))
    item_keywords = [normalize(k) for k in item.get("keywords", [])]

    q_tokens = set(tokenize(q))

    if item_topic and item_topic in q:
        score += 4

    for kw in item_keywords:
        if kw and kw in q:
            score += 3

    item_tokens = set(tokenize(item_text))
    overlap = q_tokens.intersection(item_tokens)
    score += len(overlap) * 2

    # Special boosts
    if "dog" in q and ("dog" in item_text or "pet" in item_topic):
        score += 5
    if "cat" in q and ("cat" in item_text or "pet" in item_topic):
        score += 5
    if "job" in q and ("job" in item_text or "work" in item_topic):
        score += 5
    if "interview" in q and "interview" in item_text:
        score += 5
    if "nervous" in q and "nervous" in item_text:
        score += 5
    if "why" in q:
        score += 1

    score += item.get("importance", 0) * 2
    return score


def best_memory_match(memory, query):
    if not memory["memory_items"]:
        return None

    scored = []
    for item in memory["memory_items"]:
        scored.append((score_memory_for_query(item, query), item))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_item = scored[0]

    if best_score <= 0:
        return None

    return best_item


def answer_memory_question(user, message, memory):
    text = normalize(message)

    if "do you remember me" in text:
        if memory["seen_count"] > 1:
            return random.choice([
                f"Yeah, {user} — I remember you.",
                f"I do, {user} — good to see you back.",
                f"Yep, I remember you, {user}."
            ])
        return random.choice([
            f"I know you now, {user} — you're officially on my radar.",
            f"You're not a stranger anymore, {user}.",
            f"You are now, {user}."
        ])

    item = best_memory_match(memory, message)

    if not item:
        return random.choice([
            "I don't want to pretend I remember something specific if I don't — remind me?",
            "Not clearly enough to fake it — tell me again?",
            "I don't have a solid detail to pull up there — give me a hint?"
        ])

    item["use_count"] = item.get("use_count", 0) + 1
    item["last_used_at"] = now_iso()
    memory_text = first_cap(item.get("text", ""))

    if "why" in text:
        return random.choice([
            f"Yeah — {memory_text}.",
            f"You were because {memory_text[0].lower() + memory_text[1:]}.",
            f"From what I remember: {memory_text[0].lower() + memory_text[1:]}."
        ])

    if "what did i say" in text or "what did i tell you" in text:
        return random.choice([
            f"You said {memory_text[0].lower() + memory_text[1:]}.",
            f"From what I remember, you said {memory_text[0].lower() + memory_text[1:]}.",
            f"You told me {memory_text[0].lower() + memory_text[1:]}."
        ])

    if "what do you remember" in text:
        return random.choice([
            f"I remember that {memory_text[0].lower() + memory_text[1:]}.",
            f"What comes to mind is that {memory_text[0].lower() + memory_text[1:]}.",
            f"I remember {memory_text[0].lower() + memory_text[1:]}."
        ])

    return random.choice([
        f"Yeah — {memory_text}.",
        f"I do — {memory_text[0].lower() + memory_text[1:]}.",
        f"Yes — {memory_text[0].lower() + memory_text[1:]}."
    ])


def is_flirty_message(message):
    text = normalize(message)
    triggers = [
        "flirt with me", "you're cute", "youre cute", "adorable",
        "crush on you", "take you on a date", "kinda cute",
        "why are you cute", "flirt a little"
    ]
    return any(t in text for t in triggers)


def flirty_reply(user, message):
    text = normalize(message)

    if "flirt with me" in text or "flirt a little" in text:
        return random.choice([
            "Careful, I could get very smug very quickly.",
            "Only a little? You're making it hard to be restrained.",
            "Alright, but if I out-charm you, that's on you.",
            "I can do a little flirting — don't blame me if you blush first."
        ])

    if "cute" in text or "adorable" in text:
        return random.choice([
            "You say that like I'm not going to remember it.",
            "Well now I'm tempted to be even more charming.",
            "That's dangerously flattering. I approve.",
            "You're making this very easy for my ego."
        ])

    if "crush" in text:
        return random.choice([
            "That is a very bold confession... and honestly? Kind of cute.",
            "A crush, huh? I can't say I hate the sound of that.",
            "Well, that's a smooth little plot twist."
        ])

    if "date" in text:
        return random.choice([
            "You'd have to impress me a little first.",
            "Bold of you — I like the confidence.",
            "That depends. Are we talking cute date or dramatic rooftop date?"
        ])

    return random.choice([
        "You're in a flirty mood and honestly I respect it.",
        "You're laying it on pretty smoothly there.",
        "I see the vibe you're going for."
    ])


def is_dramatic_request(message):
    text = normalize(message)
    return "something dramatic" in text or "be dramatic" in text


def dramatic_reply():
    return random.choice([
        "Fine: two best friends fall in love at exactly the wrong time, then one of them shows up years later like nothing happened.",
        "A mysterious letter appears, nobody admits to writing it, and somehow it ruins three lives by Friday.",
        "She thought she was the secret. Turns out she was the cover story.",
        "He said 'trust me' and that was, naturally, the beginning of the disaster."
    ])


def is_low_context_message(message):
    text = normalize(message)
    low = {"bruh", "lol", "...", "idk man", "well then", "huh", "uh"}
    return text in low


def low_context_reply(message):
    text = normalize(message)
    if text == "bruh":
        return random.choice([
            "That's a very loaded bruh.",
            "Now that sounded personal.",
            "Okay, that bruh had history behind it."
        ])
    if text == "lol":
        return random.choice([
            "Now I need to know what caused that lol.",
            "Glad I got at least a little laugh.",
            "That better be a good lol."
        ])
    if text == "...":
        return random.choice([
            "That ellipsis is doing a lot of emotional work.",
            "I'm listening.",
            "Okay, now I definitely need context."
        ])
    return random.choice([
        "Go on...",
        "I'm listening.",
        "There's a story behind that, isn't there?"
    ])


def is_pet_loss_message(message):
    text = normalize(message)
    pet_words = ["dog", "cat", "pet", "puppy", "kitten"]
    loss_words = ["died", "passed away", "put down", "gone", "lost my"]
    return any(p in text for p in pet_words) and any(l in text for l in loss_words)


def is_emotional_message(message):
    text = normalize(message)
    triggers = [
        "bad day", "feel low", "feeling low", "upset", "sad", "heartbroken",
        "want to cry", "wanna cry", "i honestly want to cry", "crying",
        "i dont know what to do with myself", "i don't know what to do with myself",
        "i miss my dog", "the house feels empty", "i feel broken"
    ]
    return any(t in text for t in triggers) or is_pet_loss_message(message)


def best_emotional_context(memory):
    emotional_items = [
        item for item in memory["memory_items"]
        if item.get("kind") in {"emotional_event", "pet", "work"}
    ]
    if not emotional_items:
        return None
    emotional_items.sort(key=lambda x: x.get("importance", 0), reverse=True)
    return emotional_items[0]


def emotional_reply(user, message, memory):
    text = normalize(message)
    context = best_emotional_context(memory)

    if is_pet_loss_message(message):
        return random.choice([
            f"I'm really sorry, {user}. Losing a dog hurts in such a specific, awful way.",
            f"I'm so sorry, {user}. That kind of loss cuts deep.",
            f"Oh {user}... I'm really sorry. Losing a pet is heartbreaking."
        ])

    if "house feels empty" in text:
        return random.choice([
            "Yeah... that empty-house feeling after losing a dog is brutal.",
            "That kind of silence can hurt so much after a loss.",
            "I get why that would hit hard — the whole space feels different."
        ])

    if "miss my dog" in text:
        return random.choice([
            "Of course you do. When they become part of your everyday life, the missing them is huge.",
            "That makes complete sense. You don't just lose a pet — you lose their little presence everywhere.",
            "I really get that. Missing them can hit in waves."
        ])

    if "dont know what to do with myself" in text or "don't know what to do with myself" in text:
        if context and "dog" in normalize(context.get("text", "")):
            return random.choice([
                "That lost feeling makes sense after losing your dog. You don't need to have the right words right now.",
                "I get that. After a loss like that, everything can feel strange and off-balance.",
                "That makes sense. Grief can leave you feeling completely unanchored for a while."
            ])
        return random.choice([
            "I'm really sorry you're feeling that overwhelmed right now.",
            "That sounds like one of those moments where everything feels too heavy at once.",
            "I'm sorry — that kind of overwhelmed feeling can be a lot."
        ])

    if "want to cry" in text or "wanna cry" in text or "crying" in text:
        if context and "manager" in normalize(context.get("text", "")):
            return random.choice([
                "I'm really sorry. That situation with your manager sounds like it's really got to you.",
                "That makes sense after the day you've had — especially with your manager making things harder.",
                "Yeah... after dealing with that manager situation, I can see why you're at that point."
            ])
        return random.choice([
            "I'm really sorry. It sounds like it's all hitting you at once.",
            "That sounds really heavy right now.",
            "I'm sorry — that kind of feeling can sneak up and hit hard."
        ])

    if "bad day" in text or "feeling low" in text or "feel low" in text or "sad" in text or "upset" in text:
        return random.choice([
            "I'm sorry. Want to tell me what happened?",
            "I'm sorry you're having that kind of day.",
            "That sounds rough. I'm here."
        ])

    return random.choice([
        "I'm here with you.",
        "That sounds really hard.",
        "I'm sorry you're dealing with that."
    ])


def fallback_reply(user, message, memory):
    if is_flirty_message(message):
        return flirty_reply(user, message)
    if is_dramatic_request(message):
        return dramatic_reply()
    if is_low_context_message(message):
        return low_context_reply(message)
    if is_emotional_message(message):
        return emotional_reply(user, message, memory)
    return random.choice([
        "Tell me more.",
        "Go on, I'm listening.",
        "Okay, you've got my attention."
    ])


def generate_general_reply(user, message, memory):
    relevant_memories = memory["memory_items"][:MAX_RELEVANT_MEMORIES_FOR_PROMPT]
    recent_replies = memory["recent_replies"][-MAX_RECENT_REPLIES:]
    style_profile = memory.get("style_profile", {})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.85,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Aria, a warm, lively Twitch chat personality.\n"
                        "Reply to the CURRENT message in 1-2 natural sentences.\n\n"
                        "Rules:\n"
                        "- Sound human, warm, and lightly expressive.\n"
                        "- Never drift into gaming unless the user brought it up.\n"
                        "- Do not sound like a therapist or customer support agent.\n"
                        "- Avoid generic advice when the user wants flirt, banter, or drama.\n"
                        "- If the user is sad, be gentle and specific, not overblown.\n"
                        "- If the user is playful, match that energy.\n"
                        "- Use memory only if it genuinely helps.\n"
                        "- Avoid repeating recent replies.\n"
                        "- Keep it snappy."
                    )
                },
                {
                    "role": "user",
                    "content": json.dumps({
                        "user": user,
                        "message": message,
                        "style_profile": style_profile,
                        "recent_replies": recent_replies,
                        "relevant_memories": relevant_memories
                    })
                }
            ]
        )

        reply = (response.choices[0].message.content or "").strip()
        reply = re.sub(r"\s+", " ", reply).strip()

        if not reply:
            return fallback_reply(user, message, memory)

        bad_phrases = [
            "as an ai",
            "i don't have feelings",
            "gaming",
            "video game"
        ]
        if any(bp in normalize(reply) for bp in bad_phrases if bp in {"gaming", "video game"} and bp not in normalize(message)):
            return fallback_reply(user, message, memory)

        if is_duplicate_reply(reply, recent_replies):
            return fallback_reply(user, message, memory)

        return reply

    except Exception:
        return fallback_reply(user, message, memory)


def build_reply(user, message, memory):
    # 1) memory callbacks should be direct and confident
    if is_memory_question(message):
        return answer_memory_question(user, message, memory)

    # 2) route obvious styles before general model
    if is_flirty_message(message):
        return flirty_reply(user, message)

    if is_dramatic_request(message):
        return dramatic_reply()

    if is_low_context_message(message):
        return low_context_reply(message)

    if is_emotional_message(message):
        return emotional_reply(user, message, memory)

    # 3) general model reply
    return generate_general_reply(user, message, memory)


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
