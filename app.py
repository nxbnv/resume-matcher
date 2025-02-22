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
#from sklearn.metrics.pairwise import cosine_similarity
import spacy
import re
import json
import psycopg2
from io import BytesIO
import requests
from dotenv import load_dotenv
#import torch
import base64
import subprocess
import platform 
from celery.result import AsyncResult
from celery_config import celery
import tasks
from datetime import datetime

print("Modules imported successfully!")

app = Flask(__name__)
#app.secret_key = os.getenv("SECRET_KEY", "your_secret_key_here")
app.secret_key="your_secret"
app.config["SESSION_TYPE"] = "filesystem"

load_dotenv()

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploaded_resumes")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Lazy loading models
nlp = None
bert_model = None

def load_spacy():
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading spaCy: {e}")
            exit(1)

def load_bert():
    global bert_model
    if bert_model is None:
        try:
            bert_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2", device="cpu")
            print("✅ BERT model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading BERT: {e}")
            exit(1)

#paraphrase-MiniLM-L3-v2

# Vectorizer Management (Download once and cache)
vectorizer_path = "models/vectorizer.pkl"
vectorizer_url = "https://huggingface.co/rohan57/mymodel/resolve/main/vectorizer.pkl"

def load_vectorizer():
    """Loads vectorizer.pkl directly from Hugging Face into memory."""
    try:
        print(f"🌍 Downloading vectorizer directly from {vectorizer_url}...")
        response = requests.get(vectorizer_url, timeout=15)
        response.raise_for_status()  # ✅ Raise error if download fails
        
        vectorizer = pickle.load(BytesIO(response.content))  # ✅ Load directly from memory
        print("✅ Vectorizer loaded successfully from Hugging Face!")
        return vectorizer
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error while downloading vectorizer: {e}")
        exit(1)
    except Exception as e:
        print(f"❌ Error loading vectorizer: {e}")
        exit(1)

vectorizer = load_vectorizer()

print("✅ All models ready for use!")

# ===================== DATABASE CONNECTION (PostgreSQL) =====================
# Replace with your actual DATABASE_URL or set it as an environment variable in Render.
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("❌ Missing DATABASE_URL environment variable!")

try:
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    print("✅ Connected to PostgreSQL database!")
except Exception as e:
    print("❌ Error connecting to PostgreSQL:", e)
    exit(1)

# ===================== CREATE TABLES IF NOT EXISTS =====================
cursor.execute("""
CREATE TABLE IF NOT EXISTS hr (
    id SERIAL PRIMARY KEY,
    full_name TEXT NOT NULL,
    company_name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
)
""")
conn.commit()
#good
# ===================== GMAIL API AUTHENTICATION =====================
SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
print("Authenticating Gmail API...")

# Helper function to decode Base64 safely
def decode_base64_env(var_name):
    encoded_value = os.getenv(var_name)
    if not encoded_value:
        raise ValueError(f"❌ Missing {var_name} environment variable!")
    try:
        return json.loads(base64.b64decode(encoded_value).decode("utf-8"))
    except Exception as e:
        raise ValueError(f"❌ Error decoding {var_name}: {e}")

# Decode client secret and token from Base64
client_config = decode_base64_env("CLIENT_SECRET_JSON_B64")

creds = None
try:
    token_data = os.getenv("TOKEN_JSON_B64")
    if token_data:
        creds = Credentials.from_authorized_user_info(json.loads(base64.b64decode(token_data).decode("utf-8")), SCOPES)
except Exception as e:
    print(f"⚠️ Error loading token: {e}")

# Authenticate if credentials are missing or invalid
if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
        creds = flow.run_local_server(port=5500)

    # Save refreshed credentials as token.json for reuse
    with open("token.json", "w") as token_file:
        token_file.write(creds.to_json())

service = build("gmail", "v1", credentials=creds)
print("✅ Gmail API authentication successful!")

# ===================== HR REGISTRATION & LOGIN =====================
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
            cursor.execute("INSERT INTO hr (full_name, company_name, email, password) VALUES (%s, %s, %s, %s)", 
                           (full_name, company_name, email, password))
            conn.commit()
            return redirect(url_for("login"))
        except psycopg2.IntegrityError:
            return jsonify({"error": "Email already registered"}), 400

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    """HR Login Page."""
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        cursor.execute("SELECT full_name, company_name, email FROM hr WHERE email = %s AND password = %s", 
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

# ===================== RESUME MATCHING & PROCESSING =====================
def extract_text_from_file(file_path, file):
    """Extract text from different file formats."""
    text = ""

    filename = os.path.basename(file_path)  # ✅ Get filename from file_path

    if filename.endswith(".txt"):
        text = file.read().decode("utf-8")

    elif filename.endswith(".pdf"):
        try:
            reader = PyPDF2.PdfReader(file)
            text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        except Exception as e:
            print(f"⚠️ PyPDF2 failed: {e}")

        # If no text is extracted, try OCR
        if not text.strip():
            print(f"🔄 Using OCR for {filename}...")
            text = pytesseract.image_to_string(Image.open(file_path))

    elif filename.endswith(".docx"):
        doc = docx.Document(file_path)  # ✅ Open with filename
        text = " ".join([para.text for para in doc.paragraphs])

    elif filename.endswith((".png", ".jpg", ".jpeg")):
        text = pytesseract.image_to_string(Image.open(file_path))

    return text


def extract_name(text):
    load_spacy()  # ✅ Ensure spaCy is loaded before using it
    """Extracts the candidate's name using spaCy's Named Entity Recognition (NER)."""
    doc = nlp(text)
    names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    
    if names:
        return names[0]  # Return the first detected name

    # Fallback: Try regex-based name extraction (Assumes "Name: XYZ" format)
    match = re.search(r"(?i)name[:\-]?\s*([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", text)
    return match.group(1) if match else "Unknown"


def get_bert_embeddings(text):
    """Generate sentence embeddings using SentenceTransformer."""
    load_bert()  # ✅ Ensure BERT model is loaded before using it

    truncated_text = text[:512]  # ✅ Process only first 512 characters to save RAM
    embeddings = bert_model.encode(truncated_text, normalize_embeddings=True)
    return embeddings.reshape(1, -1)  # ✅ Ensure 2D array for cosine similarity


def extract_email(text):
    """Extract email from text."""
    emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return emails[0] if emails else None


def extract_phone_number(text):
    """Extract phone number from text."""
    phones = re.findall(r"\+?\d[\d -]{8,15}\d", text)
    return phones[0] if phones else None

def enable_swap():
    """Enable swap memory on Render (Linux only)."""
    if platform.system() == "Windows":
        print("⚠️ Swap memory setup skipped on Windows (Not supported).")
        return

    try:
        # Check if swap is already enabled
        result = subprocess.run(["swapon", "--show"], capture_output=True, text=True)
        if "swapfile" in result.stdout:
            print("✅ Swap memory is already enabled.")
            return
        
        print("🔄 Enabling Swap (512MB)...")
        os.system("fallocate -l 512M /swapfile")  # Create swap file
        os.system("chmod 600 /swapfile")
        os.system("mkswap /swapfile")
        os.system("swapon /swapfile")
        print("✅ Swap memory enabled successfully!")

    except Exception as e:
        print(f"⚠️ Error enabling swap: {e}")
     
@app.route("/matcher", methods=["POST"])
def matcher():
    """Start the resume matching task in Celery."""
    if "user_email" not in session:
        return jsonify({"error": "User not logged in"}), 401

    print("✅ Received resume matching request...")

    job_description = request.form.get("job_description")
    score_threshold = float(request.form.get("score_threshold"))
    uploaded_files = request.files.getlist("resumes")

    if not job_description or not uploaded_files:
        return jsonify({"error": "Job description and resumes are required"}), 400

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    resumes = []
    for file in uploaded_files:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        resumes.append({
            "filename": file.filename,
            "file_path": os.path.join(UPLOAD_FOLDER, file.filename)  # ✅ Add full file path
        })

    # ✅ Send task to Celery worker
    task = tasks.match_resumes.apply_async(args=[job_description, resumes, score_threshold])

    return jsonify({"task_id": task.id, "message": "Matching task started!"}), 202

@app.route("/task-status/<task_id>", methods=["GET"])
def task_status(task_id):
    """Check the status of a Celery task."""
    task = AsyncResult(task_id,app=celery)

    if task.state == "PENDING":
        response = {"status": "Processing", "message": "Task is still running"}
    elif task.state == "SUCCESS":
        response = {"status": "Completed", "result": task.result}
    elif task.state == "FAILURE":
        response = {"status": "Failed", "error": str(task.info)}
    else:
        response = {"status": task.state, "message": "Unknown state"}

    return jsonify(response)

@app.route("/download-excel", methods=["GET"])
def download_excel():
    print("inside excel")  # Debugging

    if "user_email" not in session:
        return jsonify({"error": "User not logged in"}), 401

    selected_candidates = session.get("selected_candidates", [])
    if not selected_candidates:
        print("❌ No selected candidates in session!")  # Debugging
        return jsonify({"error": "No selected candidates found!"}), 400

    print("🔍 Selected Candidates Data:", selected_candidates)

    # ✅ Convert data to DataFrame
    df = pd.DataFrame(selected_candidates)

    print("🔍 DataFrame Columns:", df.columns)
    print("🔍 DataFrame Sample:", df.head())

    # ✅ Ensure "resume" column exists before adding hyperlinks
    if "resume" in df.columns and df["resume"].notna().any():
        #base_url = "https://huggingface.co/rohan57/mymodel/resolve/main/sample_resumes/"  # Change this to actual location
        #df["Resume Link"] = df["resume"].apply(lambda x: f'=HYPERLINK("{base_url}{x}", "Download")' if pd.notna(x) else "No File")
        # ✅ Get the absolute path of the sample_resumes folder
        #resume_folder = os.path.abspath("sample_resumes")
        base_url=os.path.abspath("uploaded_resumes")
        # ✅ Generate local file links
        df["Resume Link"] = df["resume"].apply(lambda x: f'=HYPERLINK("file:///{os.path.join(base_url, x)}", "Open")' if pd.notna(x) else "No File")
    else:
        df["Resume Link"] = "No File"  # ✅ Add empty column if "resume" is missing

    print("🔍 DataFrame Columns:", df.columns)  # Debugging

    # ✅ Generate a unique file name
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_path = f"selected_candidates_{timestamp}.xlsx"

    try:
        # ✅ Save to Excel with clickable links
        with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
            df.drop(columns=["resume"], errors="ignore").to_excel(writer, index=False, sheet_name="Candidates")

            # ✅ Apply hyperlink format safely
            workbook = writer.book
            worksheet = writer.sheets["Candidates"]
            link_format = workbook.add_format({"font_color": "blue", "underline": 1})

            if "Resume Link" in df.columns:  # ✅ Check before using
                col_idx = df.columns.get_loc("Resume Link")
                for row_num in range(1, len(df) + 1):
                    resume_file = df.at[row_num - 1, "resume"]
                    resume_url = f"{base_url}{resume_file}"
                    if isinstance(resume_file, str):
                        worksheet.write_url(row_num, col_idx, resume_url, link_format, "Download")

        return send_file(file_path, as_attachment=True)

    except Exception as e:
        print("❌ Error creating Excel file:", e)  # Debugging
        return jsonify({"error": "Failed to generate Excel file"}), 500


    
@app.route("/download-resume/<filename>")
def download_resume(filename):
    resume_folder = "uploaded_resumes"  # Change to match your storage path
    file_path = os.path.join(resume_folder, filename)
    
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    
    return send_file(file_path, as_attachment=True)

@app.route("/save-selected-candidates", methods=["POST"])
def save_selected_candidates():
    data = request.get_json()
    if not data or "candidates" not in data or not data["candidates"]:
        return jsonify({"error": "No candidates provided"}), 400

    session["selected_candidates"] = data["candidates"]
    return jsonify({"message": "Selected candidates saved!"})


def send_email(service, sender_name, sender_company, sender_email, recipient_email, subject, body):
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

@app.route("/send-email", methods=["POST"])  # ✅ Match frontend
def send_selected_emails():
    print("inside selected mails")

    # Check if user is logged in
    if "user_email" not in session:
        return jsonify({"error": "User not logged in"}), 401

    data = request.get_json()
    if not data or "candidates" not in data:
        return jsonify({"error": "No candidates provided"}), 400

    # Ensure service exists
    if "service" not in globals() or not service:
        return jsonify({"error": "Email service not initialized"}), 500

    errors = []
    for candidate in data["candidates"]:
        try:
            send_email(service, session["hr_name"], session["hr_company"], session["user_email"], 
                       candidate["email"], "Interview Selection", f"{candidate['name']}, You have been selected!")
        except Exception as e:
            errors.append(f"Failed to send to {candidate['email']}: {e}")

    if errors:
        return jsonify({"error": "Some emails failed", "details": errors}), 207  # 207 Multi-Status

    return jsonify({"message": "Emails sent successfully!"}), 200

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))  # Use Render's port
    print(f"🚀 Running Flask on port {port}...")
    app.run(host="0.0.0.0", port=port)