print("Starting app.py execution...")

from flask import Flask, request, render_template, jsonify, redirect, url_for, session, send_file
import os
import docx
import PyPDF2
import pickle
import pytesseract
import base64
import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from email.mime.text import MIMEText
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import re
import numpy as np
import sqlite3
import zipfile

print("Modules imported successfully!")

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

# Create Upload Folder for Resumes
UPLOAD_FOLDER = "uploaded_resumes"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure folder exists

# Connect to SQLite Database
db = sqlite3.connect("resume_db.sqlite", check_same_thread=False)
cursor = db.cursor()

# Create HR Table if it doesn’t exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS hr (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT NOT NULL,
    company_name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
)
""")
db.commit()

try:
    print("Loading models...")

    # Define paths
    zip_path = "models/bert_model.zip"
    extract_path = "models/"
    model_path = os.path.join(extract_path, "bert_model.pkl")

    # Extract bert_model.pkl if it hasn't been extracted yet
    if not os.path.exists(model_path):
        print("Extracting bert_model.zip...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("✅ Model extracted successfully!")

    # Load models
    vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
    bert_model = pickle.load(open(model_path, "rb"))
    nlp = spacy.load("en_core_web_sm")

    print("✅ Models loaded successfully!")

except Exception as e:
    print("❌ Error loading models:", e)
    exit(1)

# Gmail API Authentication
SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
print("Authenticating Gmail API...")
creds = None

try:
    if os.path.exists("token.json"):
        with open("token.json", "r") as token:
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", SCOPES)
            creds = flow.run_local_server(port=5000)
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    service = build("gmail", "v1", credentials=creds)
    print("✅ Gmail API authenticated!")
except Exception as e:
    print("❌ Error authenticating Gmail API:", e)
    exit(1)

# ========================== HR REGISTRATION & LOGIN ==========================

@app.route("/")
def home():
    return redirect(url_for("register"))

@app.route("/register", methods=["GET", "POST"])
def register():
    """HR Registration Page."""
    if request.method == "POST":
        full_name = request.form.get("full_name")
        company_name = request.form.get("company_name")
        email = request.form.get("email")
        password = request.form.get("password")

        if not full_name or not company_name or not email or not password:
            return jsonify({"error": "All fields are required"}), 400

        try:
            cursor.execute("INSERT INTO hr (full_name, company_name, email, password) VALUES (?, ?, ?, ?)", 
                           (full_name, company_name, email, password))
            db.commit()
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            return jsonify({"error": "Email already registered"}), 400

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    """HR Login Page."""
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        cursor.execute("SELECT full_name, company_name, email FROM hr WHERE email = ? AND password = ?", 
                       (email, password))
        user = cursor.fetchone()

        if user:
            session["hr_name"], session["hr_company"], session["user_email"] = user
            return redirect(url_for("index"))
        else:
            return jsonify({"error": "Invalid email or password"}), 401

    return render_template("login.html")

@app.route("/logout")
def logout():
    """Log out HR user."""
    session.clear()
    return redirect(url_for("login"))

@app.route("/index")
def index():
    """Render the main HR dashboard after login."""
    if "user_email" not in session:
        return redirect(url_for("login"))
    return render_template("index.html", hr_name=session["hr_name"], hr_company=session["hr_company"])

# ========================== RESUME MATCHING & PROCESSING ==========================

def extract_text_from_file(file):
    """Extract text from different file formats."""
    text = ""
    filename = file.filename
    if filename.endswith(".txt"):
        text = file.read().decode("utf-8")
    elif filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif filename.endswith(".docx"):
        doc = docx.Document(file)
        text = " ".join([para.text for para in doc.paragraphs])
    elif filename.endswith((".png", ".jpg", ".jpeg")):
        text = pytesseract.image_to_string(Image.open(file))
    return text

def extract_name(text):
    """Extracts the candidate's name using spaCy's Named Entity Recognition (NER)."""
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":  # Look for names recognized as 'PERSON'
            return ent.text
    return None

def extract_email(text):
    """Extract email from text."""
    emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return emails[0] if emails else None

def extract_phone_number(text):
    """Extract phone number from text."""
    phones = re.findall(r"\+?\d[\d -]{8,15}\d", text)
    return phones[0] if phones else None  # ✅ Returns the first valid phone number

@app.route("/matcher", methods=["POST"])
def matcher():
    """Matches resumes with job description and saves them."""
    if "user_email" not in session:
        return jsonify({"error": "User not logged in"}), 401

    job_description = request.form.get("job_description")
    uploaded_files = request.files.getlist("resumes")
    score_threshold = float(request.form.get("score_threshold"))

    results = []
    for file in uploaded_files:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)  # ✅ Save PDF inside the uploads folder

        text = extract_text_from_file(file)
        name = extract_name(text)  # ✅ Correct function for name
        email = extract_email(text)  # ✅ Correct function for email

        bert_score = cosine_similarity([bert_model.encode(job_description)], [bert_model.encode(text)])[0][0]

        results.append({
            "resume": file.filename,
            "name": name if name else "Unknown",
            "email": email if email else "N/A",
            "phone": extract_phone_number(text),
            "bert_score": round(float(bert_score), 2),
            "resume_link": f'=HYPERLINK("{os.path.abspath(file_path)}", "Open Resume")'  # ✅ Clickable Link
        })

    selected_candidates = [res for res in results if res["bert_score"] >= score_threshold]

    session["selected_candidates"] = selected_candidates  # ✅ Save in session

    return jsonify({"selected_candidates": selected_candidates, "scores": results})




@app.route("/download-excel", methods=["GET"])
def download_excel():
    """Generate and return an Excel file with clickable resume links."""
    if "user_email" not in session:
        return jsonify({"error": "User not logged in"}), 401

    selected_candidates = session.get("selected_candidates", [])
    if not selected_candidates:
        return jsonify({"error": "No selected candidates found!"}), 400

    df = pd.DataFrame(selected_candidates)
    file_path = "selected_candidates.xlsx"
    df.to_excel(file_path, index=False)

    return send_file(file_path, as_attachment=True)

def send_email(service,sender_name, sender_company, sender_email, recipient_email, subject, body):
    """Send an email using Gmail API from the logged-in HR's email."""
    message = MIMEText(f"Dear {body},\n\nYou have been selected for an interview at {sender_company}.\n\nBest Regards,\n{sender_name}\n{sender_company}\n{sender_email}")
    message["From"] = f"{sender_name} <{sender_email}>"
    message["To"] = recipient_email
    message["Subject"] = subject
    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

    try:
        service.users().messages().send(userId="me", body={"raw": raw_message}).execute()
        print(f"✅ Email sent from {sender_email} to {recipient_email}")
    except Exception as e:
        print(f"❌ Error sending email from {sender_email}: {e}")



@app.route("/send-email", methods=["POST"])
def send_selected_emails():
    if "user_email" not in session:
        return jsonify({"error": "User not logged in"}), 401

    data = request.get_json()
    if not data or "candidates" not in data:
        return jsonify({"error": "No candidates provided"}), 400

    for candidate in data["candidates"]:
        send_email(service, session["hr_name"], session["hr_company"], session["user_email"], 
                   candidate["email"], "Interview Selection", f" {candidate['name']}, You have been selected!")

    return jsonify({"message": "Emails sent successfully!"})


if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True)
