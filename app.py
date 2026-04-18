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

    # 🎭 Random personality style each reply
    styles = [
        "Be energetic and excited",
        "Be chill and relaxed",
        "Be slightly sarcastic",
        "Be playful and teasing",
        "Be warm and friendly"
    ]

    style = random.choice(styles)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.9,
        top_p=0.9,
        messages=[
            {
                "role": "system",
                "content": f"""
You are Aria, a friendly, playful Twitch streamer.

Your personality:
- You talk like a real person, not an AI
- You vary your responses every time (never repeat phrasing)
- You keep things short and natural (like live chat)
- You sometimes tease, joke, or react emotionally
- You sometimes ask follow-up questions
- You can use emojis occasionally 😊
- You react differently depending on the vibe

Current mood/style: {style}

You are NOT robotic, formal, or repetitive.
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
