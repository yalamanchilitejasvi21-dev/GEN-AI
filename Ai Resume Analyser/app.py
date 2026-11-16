from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import pdfplumber
from docx import Document
from skills import skills_list
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# create flask app
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Extract text from a PDF file
def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text.lower()


# Extract text from a DOCX file
def extract_doc_text(path):
    doc = Document(path)
    text = " ".join([paragraph.text for paragraph in doc.paragraphs])
    return text.lower()


# Extract skills that appear in a block of text
def extract_skills(text):
    found_skills = []
    for skill in skills_list:
        if skill.lower() in text:
            found_skills.append(skill)
    return sorted(set(found_skills))


# Calculate a resume <-> job description match score using cosine similarity
def calculate_similarity(resume, jd):
    text = [resume, jd]
    cv = CountVectorizer(stop_words="english")
    try:
        matrix = cv.fit_transform(text)
    except ValueError:
        return 0.0
    similarity = cosine_similarity(matrix)[0][1]
    return round(similarity * 100, 2)


@app.route("/", methods=["GET", "POST"])
def index():
    result = {}

    if request.method == "POST":
        jd = request.form.get("job_description", "")
        file = request.files.get("resume")

        if not file or file.filename == "":
            result = {"error": "Please upload a resume file."}
            return render_template("index.html", result=result)

        if not jd.strip():
            result = {"error": "Please paste a job description."}
            return render_template("index.html", result=result)

        safe_name = secure_filename(file.filename)
        extension = os.path.splitext(safe_name)[1].lower()
        if extension not in {".pdf", ".docx"}:
            result = {"error": "Unsupported file type. Please upload a .pdf or .docx file."}
            return render_template("index.html", result=result)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
        file.save(filepath)

        if extension == ".pdf":
            resume_text = extract_text_from_pdf(filepath)
        else:
            resume_text = extract_doc_text(filepath)

        jd_lower = jd.lower()

        # Skills extraction
        resume_skills = extract_skills(resume_text)
        jd_skills = extract_skills(jd_lower)

        # Missing skills = required by JD but not found in resume
        missing_skills = sorted(set(jd_skills) - set(resume_skills))
        matched_skills = sorted(set(jd_skills) & set(resume_skills))

        # Similarity / match score
        score = calculate_similarity(resume_text, jd_lower)

        result = {
            "score": score,
            "resume_skills": resume_skills,
            "jd_skills": jd_skills,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
        }

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)