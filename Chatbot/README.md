# Study Buddy

A small Flask chat interface powered by Groq. Conversation context is stored per browser session and capped to the latest six exchanges.

1. Install dependencies from `requirements.txt`.
2. Copy `.env.example` to `.env`; set `GROQ_API_KEY` and a private `FLASK_SECRET_KEY`.
3. Run `python app.py` from this folder.
