from sentence_transformers import SentenceTransformer

# Load the model once and reuse it (loading it per-request is slow)
_model = None

EMBEDDING_DIM = 384  # output size of all-MiniLM-L6-v2


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def generate_embedding(text):
    """Convert a piece of text into a 384-dim vector embedding."""
    model = _get_model()
    embedding = model.encode(text)
    return embedding