#from celery import Celery
import os
import gc
import re
import spacy
import PyPDF2
import docx
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from celery_config import celery

# Lazy loading models
nlp = None
bert_model = None

def load_spacy():
    """Lazy-load spaCy model."""
    global nlp
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")

def load_bert():
    """Lazy-load BERT model."""
    global bert_model
    if bert_model is None:
        bert_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2", device="cpu")

@celery.task(name="match_resumes")  # ✅ Explicit name
def match_resumes(job_description, resumes, score_threshold):
    """ Background task for resume matching """
    print("🔄 Matching resumes in Celery worker...")

    load_spacy()
    load_bert()
    
    job_description_embeddings = bert_model.encode(job_description, normalize_embeddings=True).reshape(1, -1)
    results = []

    for resume in resumes:
        file_path = resume["file_path"]

        try:
            text = extract_text_from_file(file_path)
            if not text.strip():
                print(f"⚠️ Skipping {file_path}: No text extracted")
                continue

            name = extract_name(text) or "Unknown"
            email = extract_email(text) or "N/A"
            phone = extract_phone_number(text) or "N/A"

            resume_embeddings = bert_model.encode(text, normalize_embeddings=True).reshape(1, -1)
            bert_score = cosine_similarity(job_description_embeddings, resume_embeddings)[0][0]
            bert_score = max(0.0, min(bert_score, 1.0))

            results.append({
                "resume": os.path.basename(file_path),
                "resume_path": f"/download-resume/{os.path.basename(file_path)}",  # ✅ Add download link
                "name": name,
                "email": email,
                "phone": phone,
                "bert_score": round(float(bert_score), 2),
            })

            os.remove(file_path)  # Delete processed file
            gc.collect()  # Free memory

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            continue

    selected_candidates = [res for res in results if res["bert_score"] >= score_threshold]
    #return {"selected_candidates": selected_candidates, "scores": results}
    return {"matching_results": results, "selected_candidates": selected_candidates}


def extract_text_from_file(file_path):
    """Extract text from different file formats."""
    text = ""

    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    elif file_path.endswith(".pdf"):
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        text = " ".join([para.text for para in doc.paragraphs])
    elif file_path.endswith((".png", ".jpg", ".jpeg")):
        text = pytesseract.image_to_string(Image.open(file_path))

    return text

def extract_name(text):
    """Extracts candidate's name using spaCy."""
    doc = nlp(text)
    names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    return names[0] if names else None

def extract_email(text):
    """Extract email from text."""
    emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return emails[0] if emails else None

def extract_phone_number(text):
    """Extract phone number from text."""
    phones = re.findall(r"\+?\d[\d -]{8,15}\d", text)
    return phones[0] if phones else None
