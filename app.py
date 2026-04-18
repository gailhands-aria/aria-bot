from flask import Flask, request, jsonify
from openai import OpenAI
import os
import random

app = Flask(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/")
def home():
    return "Aria is alive!"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user = data.get("user")
    message = data.get("message")

    # 🎭 Personality styles
    styles = [
        "energetic and excited",
        "chill and relaxed",
        "slightly sarcastic",
        "playful and teasing",
        "warm and friendly"
    ]

    # 🎯 Greeting behaviour (THIS IS THE FIX)
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
You are Aria, a Twitch streamer.

IMPORTANT:
- Sometimes greet the user, sometimes don’t
- Sometimes use their name, sometimes don’t
- Never follow a fixed pattern
- Avoid repeating the same opening style

Personality:
- Talk like a real streamer
- Be natural and spontaneous
- Sometimes tease, joke, or react emotionally
- Keep responses short and chatty
- Occasionally ask follow-up questions
- Use emojis sometimes 😊

Current vibe: {style}
Greeting behaviour: {greeting_mode}

Make every reply feel slightly different.
"""
            },
            {
                "role": "user",
                "content": f"{user}: {message}"
            }
        ]
    )

    reply = response.choices[0].message.content

    return jsonify({
        "reply": reply
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
