from flask import Flask, render_template, request, jsonify

from utils.embeddings import generate_embedding
from utils.vector_db import add_to_faiss, search_similar, get_all_history
from utils.groq_generator import generate_ai_content

app = Flask(__name__)


# ==================================================
# Home Page
# ==================================================
@app.route("/")
def index():
    return render_template("index.html")


# ==================================================
# History Page
# ==================================================
@app.route("/history")
def history():
    past_content = get_all_history()
    return render_template("history.html", history=past_content)


# ==================================================
# Generate Content (RAG pipeline)
# ==================================================
@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    topic = (data.get("topic") or "").strip()
    platform = (data.get("platform") or "").strip()
    tone = (data.get("tone") or "").strip()

    if not topic or not platform or not tone:
        return jsonify({"error": "topic, platform, and tone are all required."}), 400

    if len(topic) > 300:
        return jsonify({"error": "Please keep the topic under 300 characters."}), 400

    # --------------------------------------------
    # Generate Embedding for the topic
    # --------------------------------------------
    try:
        topic_embedding = generate_embedding(topic)
    except Exception as e:
        app.logger.exception("Embedding generation failed")
        return jsonify({"error": "Could not prepare the topic for retrieval."}), 502

    # --------------------------------------------
    # Retrieve similar past content (RAG context)
    # --------------------------------------------
    similar_contents = search_similar(topic_embedding, k=3)
    context = "\n\n".join(similar_contents) if similar_contents else "No prior related content."

    # --------------------------------------------
    # Create RAG Prompt
    # --------------------------------------------
    prompt = f"""Generate high quality {platform} content.
Topic: {topic}
Tone: {tone}
Use these previous related contents as inspiration:
{context}

Make the content unique, engaging, professional, and attractive."""

    # --------------------------------------------
    # Generate content via Groq
    # --------------------------------------------
    try:
        generated_content = generate_ai_content(prompt)
    except Exception:
        app.logger.exception("Content generation failed")
        return jsonify({"error": "Content generation failed. Check the API key and try again."}), 502

    # --------------------------------------------
    # Save this generation to the vector store
    # so future related topics can reference it
    # --------------------------------------------
    add_to_faiss(
        generated_content,
        topic_embedding,
        metadata={"topic": topic, "platform": platform, "tone": tone},
    )

    return jsonify({"content": generated_content})


if __name__ == "__main__":
    app.run(debug=True)