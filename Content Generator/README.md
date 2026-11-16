# ContextCraft

A Flask content creator that retrieves related saved generations to provide context for new drafts.

1. Install dependencies from `requirements.txt`.
2. Copy `.env.example` to `.env` and set `GROQ_API_KEY`.
3. Run `python app.py` from this folder.

The vector index and generation history are stored locally in `vector_store/`.
