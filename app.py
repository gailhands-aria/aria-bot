from flask import Flask, request, jsonify
import openai
import os

app = Flask(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route("/")
def home():
    return "Aria is alive!"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user = data.get("user")
    message = data.get("message")

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are Aria, a friendly, slightly playful Twitch streamer who talks like a real person."},
            {"role": "user", "content": f"{user}: {message}"}
        ]
    )

    reply = response.choices[0].message.content

    return jsonify({
        "reply": reply
    })
