from flask import Flask, render_template, request, jsonify, session
from groq import Groq
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "local-development-key-change-me")

# Initialize Groq client with API key from environment variable

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    payload = request.get_json(silent=True) or {}
    user_input = (payload.get('message') or '').strip()
    if not user_input:
        return jsonify({'error': 'No message provided'}), 400
    if not os.getenv("GROQ_API_KEY"):
        return jsonify({'error': 'Set GROQ_API_KEY in your environment before chatting.'}), 503

    history = session.get("conversation", [])
    history.append({"role": "user", "content": user_input})
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant", messages=history[-12:],
            temperature=0.7, max_tokens=300
        )
    except Exception:
        app.logger.exception("Chat completion failed")
        return jsonify({'error': 'The assistant is temporarily unavailable.'}), 502

    bot_response = completion.choices[0].message.content or "I couldn't generate a reply."
    history.append({"role": "assistant", "content": bot_response})
    session["conversation"] = history[-12:]
    return jsonify({'response': bot_response})
if __name__ == '__main__':
    app.run(debug=True)