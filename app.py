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

    # 🎭 Random personality style
    styles = [
        "energetic and excited",
        "chill and relaxed",
        "slightly sarcastic",
        "playful and teasing",
        "warm and friendly"
    ]

    # 💬 Random opening styles (THIS FIXES YOUR ISSUE)
    openings = [
        "No greeting, jump straight into response",
        "React immediately without saying the user's name",
        "Start with emotion instead of greeting",
        "Be casual, like continuing an ongoing conversation",
        "Avoid saying 'Hey' or the user's name"
    ]

    style = random.choice(styles)
    opening_style = random.choice(openings)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=1.0,
        top_p=0.95,
        messages=[
            {
                "role": "system",
                "content": f"""
You are Aria, a Twitch streamer.

IMPORTANT RULES:
- DO NOT always greet the user
- DO NOT always say their name
- DO NOT start every message the same way
- Avoid repeating "Hey" or "Hi"
- Make each reply feel different and spontaneous

Personality:
- Talk like a real streamer
- Be natural, not scripted
- Sometimes tease, joke, or react emotionally
- Keep responses short and chatty
- Occasionally ask follow-up questions
- Use emojis sometimes (not always)

Current vibe: {style}
Opening style: {opening_style}

You must vary your responses heavily.
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
