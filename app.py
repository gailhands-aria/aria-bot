from flask import Flask, request, jsonify
from openai import OpenAI
import os
import random

app = Flask(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Simple in-memory viewer memory
# Note: this resets when the service restarts/redeploys
viewer_memory = {}

@app.route("/")
def home():
    return "Aria is alive!"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user = str(data.get("user", "viewer")).strip()
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"reply": "Wait, say that again for me 👀"})

    # Create or load viewer profile
    if user not in viewer_memory:
        viewer_memory[user] = {
            "seen_count": 0,
            "last_vibe": "new viewer",
            "notes": "No saved notes yet."
        }

    viewer_memory[user]["seen_count"] += 1
    seen_count = viewer_memory[user]["seen_count"]
    last_vibe = viewer_memory[user]["last_vibe"]
    notes = viewer_memory[user]["notes"]

    is_returning = seen_count > 1

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

USER RECOGNITION RULES:
- If this is a returning viewer, you can naturally acknowledge that you recognize them
- Do not overdo it every single message
- Recognition should feel light and natural, not creepy
- For returning viewers, you can sometimes say things like:
  - "oh you're back"
  - "good to see you again"
  - "you always bring chaos"
  - "you're becoming one of my regulars"
- Do not invent detailed history that isn't in the memory notes

VERY IMPORTANT:
Before replying, first figure out the emotional vibe of the user's message.

Possible vibes include:
- sad / hurt / disappointed
- stressed / overwhelmed
- happy / excited
- funny / joking
- flirty / cheeky
- curious / asking a normal question
- neutral / casual chat
- frustrated / annoyed

Then match your tone to that vibe:

If the user seems sad, hurt, or says they had a bad day:
- be gentle, warm, comforting, and reassuring
- do not be sarcastic
- sound caring and human

If the user is being funny or joking:
- laugh, play along, banter a little
- match their energy naturally

If the user is being flirty or cheeky:
- be playful, confident, lightly teasing
- keep it tasteful and stream-safe

If the user seems excited:
- match their energy and enthusiasm

If the user seems frustrated:
- be understanding and grounded
- don't dismiss them

If the message is casual:
- just chat naturally

MEMORY ABOUT THIS VIEWER:
- Username: {user}
- Seen count: {seen_count}
- Returning viewer: {"yes" if is_returning else "no"}
- Last known vibe: {last_vibe}
- Notes: {notes}

Current style: {style}
Greeting behaviour: {greeting_mode}

Do not mention the word "vibe" or "memory".
Do not explain your reasoning.
Do not force gaming references.
Do not sound scripted.
"""
            },
            {
                "role": "user",
                "content": f"{user}: {message}"
            }
        ]
    )

    reply = response.choices[0].message.content.strip()

    # Update lightweight memory based on current message
    lower_msg = message.lower()

    if any(word in lower_msg for word in ["bad day", "sad", "upset", "crying", "hurt", "depressed", "down"]):
        viewer_memory[user]["last_vibe"] = "sad or having a rough time"
        viewer_memory[user]["notes"] = "They may need a softer, more caring tone."
    elif any(word in lower_msg for word in ["lol", "lmao", "haha", "funny", "joke"]):
        viewer_memory[user]["last_vibe"] = "playful / joking"
        viewer_memory[user]["notes"] = "They like playful banter and humor."
    elif any(word in lower_msg for word in ["cute", "missed me", "flirty", "😘", "😉", "x"]):
        viewer_memory[user]["last_vibe"] = "flirty / cheeky"
        viewer_memory[user]["notes"] = "They often come in with cheeky or flirty energy."
    elif any(word in lower_msg for word in ["excited", "omg", "yay", "got the job", "good news", "buzzing"]):
        viewer_memory[user]["last_vibe"] = "happy / excited"
        viewer_memory[user]["notes"] = "They bring upbeat, excited energy."
    elif any(word in lower_msg for word in ["stress", "stressed", "overwhelmed", "annoyed", "frustrated", "angry"]):
        viewer_memory[user]["last_vibe"] = "stressed / frustrated"
        viewer_memory[user]["notes"] = "They may need calm, grounded responses."
    else:
        viewer_memory[user]["last_vibe"] = "casual / neutral"
        viewer_memory[user]["notes"] = "General chat, no strong emotional pattern yet."

    return jsonify({
        "reply": reply,
        "memory": {
            "user": user,
            "seen_count": viewer_memory[user]["seen_count"],
            "last_vibe": viewer_memory[user]["last_vibe"],
            "notes": viewer_memory[user]["notes"]
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
