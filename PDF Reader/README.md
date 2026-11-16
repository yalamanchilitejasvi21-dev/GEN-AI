# PDF Reader

A command-line document Q&A helper. Add PDF or TXT files to `documents/`, set `GROQ_API_KEY` using `.env.example`, then run `python main.py`.

Questions use TF-IDF retrieval over overlapping document passages. Basic arithmetic is parsed with a restricted expression evaluator.
