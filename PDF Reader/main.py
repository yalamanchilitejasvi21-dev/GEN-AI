# Document question and answering using RAG
import os
import re
import ast
import operator
from dotenv import load_dotenv
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# Load environment variables from .env file
load_dotenv()

# Global variable for the document folder
document_folder = "documents"  # Folder where the PDF documents are stored
documents = []       # List to store the loaded documents
chunks = []          # List to store the text chunks from the documents
chat_history = []    # List to store the chat history
vectorizer = TfidfVectorizer()  # TF-IDF vectorizer for the documents
chunk_vectors = None            # TF-IDF vectors of the chunks

PRONOUNS = {
    "i", "me", "my", "mine", "you", "your", "yours", "he", "him", "his",
    "she", "her", "hers", "it", "its", "we", "us", "our", "ours",
    "they", "them", "their", "theirs"
}


# Load documents
def load_documents():
    if not os.path.isdir(document_folder):
        print(f"'{document_folder}' folder not found. Creating it now — add PDF or text files there and rerun.")
        os.makedirs(document_folder)
        return
    for file_name in os.listdir(document_folder):
        file_path = os.path.join(document_folder, file_name)
        text = ""
        if file_name.lower().endswith(".pdf"):
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text  # was page_text() — calling a string as a function
        elif file_name.lower().endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
        if text.strip():
            documents.append({"file_name": file_name, "text": text.strip()})

# Split documents into overlapping chunks
def split_documents(chunk_size=500, overlap=100):
    for doc in documents:
        text = doc["text"]
        source = doc["file_name"]
        step = chunk_size - overlap
        for i in range(0, len(text), step):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                # use a consistent key ("content") everywhere
                chunks.append({"source": source, "content": chunk})


# Vectorize the chunks using TF-IDF
def create_chunk_vectors():
    global vectorizer, chunk_vectors
    chunk_texts = [chunk["content"] for chunk in chunks]
    vectorizer = TfidfVectorizer()
    chunk_vectors = vectorizer.fit_transform(chunk_texts)


# Retrieve relevant chunks based on the query using cosine similarity
def retrieve_relevant_chunks(query, top_k=3, threshold=0.1):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, chunk_vectors)
    # indices of the top_k most similar chunks, descending order of similarity
    ranked_indices = similarities.argsort()[0][-top_k:][::-1]

    best_per_doc = {}
    for idx in ranked_indices:
        score = similarities[0, idx]
        if score < threshold:
            # ranked_indices is already sorted descending, so anything
            # after this point will only be lower — safe to stop.
            break
        chunk = chunks[idx]
        source = chunk["source"]
        if source not in best_per_doc:
            best_per_doc[source] = {"content": chunk["content"], "score": score, "source": source}

    selected_chunks = sorted(best_per_doc.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    return selected_chunks


# Simple router: decide whether the query is a calculation or a document search
def decide_action(query):
    calculation_keywords = ["calculate", "compute", "evaluate", "determine", "find", "solve"]
    query_lower = query.lower()
    for keyword in calculation_keywords:
        if keyword in query_lower:
            return "calculate"
    return "document_search"


# Safely evaluate a basic arithmetic expression (no eval() on raw user input)
_ALLOWED_EXPR_RE = re.compile(r"^[0-9+\-*/().\s]+$")
_BINARY_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                     ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
                     ast.Mod: operator.mod, ast.Pow: operator.pow}

def _evaluate_expression(node):
    if isinstance(node, ast.Expression):
        return _evaluate_expression(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_expression(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left, right = _evaluate_expression(node.left), _evaluate_expression(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 8:
            raise ValueError("Exponent is too large")
        return _BINARY_OPERATORS[type(node.op)](left, right)
    raise ValueError("Unsupported expression")

def calculate_answer(query):
    expression = query.lower()
    for word in ["calculate", "compute", "evaluate", "determine", "find", "solve"]:
        expression = expression.replace(word, "")
    expression = expression.strip()

    if not expression or not _ALLOWED_EXPR_RE.match(expression):
        return "I'm sorry, I can only evaluate simple arithmetic expressions (digits and + - * / ( ))."

    try:
        result = _evaluate_expression(ast.parse(expression, mode="eval"))
        return f"The result of the calculation is: {result}"
    except Exception:
        return "I'm sorry, I couldn't evaluate the expression. Please check the syntax and try again."


# Check if the query relies on pronouns (i.e. refers back to prior context)
def is_pronoun_query(query):
    tokens = re.findall(r"[a-zA-Z']+", query.lower())
    return any(token in PRONOUNS for token in tokens)


# Expand a pronoun-heavy follow-up question using the previous answer
def build_search_query(query):
    if chat_history and is_pronoun_query(query):
        last_answer = chat_history[-1]
        last_answer_text = last_answer.replace("Assistant:", "", 1)
        first_sentence = last_answer_text.split(".")[0].strip()
        return f"{first_sentence} {query}"
    return query


# Build the prompt sent to the model, including retrieved context
def build_prompt(query, relevant_chunks):
    context_text = "\n\n".join(
        f"Source: {c['source']}\nContent: {c['content']}" for c in relevant_chunks
    )
    return (
        "You are a helpful assistant answering questions using only the provided context.\n"
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n\n"
        "Respond in exactly this format:\n"
        "Answer : <your answer>\n"
        "Source : <source file name(s)>"
    )


# Retrieve context, call Groq, and parse out the answer/source
def answer_query(query):
    search_query = build_search_query(query)
    relevant_chunks = retrieve_relevant_chunks(search_query)

    if not relevant_chunks:
        return "I couldn't find anything relevant in the documents.", "N/A"

    prompt = build_prompt(search_query, relevant_chunks)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Set GROQ_API_KEY in your environment before asking document questions.", "N/A"
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    raw = completion.choices[0].message.content.strip()

    answer, source = raw, "N/A"
    for line in raw.split("\n"):
        if line.startswith("Answer : "):
            answer = line.replace("Answer : ", "", 1).strip()
        elif line.startswith("Source : "):
            source = line.replace("Source : ", "", 1).strip()

    chat_history.append(f"Assistant: {answer}")
    return answer, source


# Main program
def main():
    print("\nLoading documents...")
    load_documents()
    if len(documents) == 0:
        print("No documents found in the 'documents' folder. Please add PDF or text files and try again.")
        return

    split_documents()
    create_chunk_vectors()
    print("Assistant is ready to answer your questions.")

    while True:
        query = input("Ask Question : ").strip()
        if query.lower() in ["quit", "exit"]:
            print("Exiting the assistant. Goodbye!")
            break

        if not query:
            continue

        action = decide_action(query)
        if action == "calculate":
            result = calculate_answer(query)
            print(result)
            continue

        answer, source = answer_query(query)
        print(f"Answer: {answer}")
        print(f"Source: {source}")


if __name__ == "__main__":
    main()