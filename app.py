from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware
import requests
from datetime import datetime, timedelta
import pytesseract
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import pipeline
import pytesseract
import cv2
import numpy as np
from datetime import datetime
from pydantic import BaseModel
import json
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# ================= VERIFIED NEWS FEATURE =================

# ---------------- APP SETUP ----------------
import os
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
app = FastAPI()
image_captioner = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-base"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- LOAD MODEL ----------------
classifier = pipeline(
    "text-classification",
    model="jy46604790/Fake-News-Bert-Detect"
)

# ---------------- REQUEST SCHEMA ----------------
class NewsInput(BaseModel):
    text: str

# ---------------- API ENDPOINT ----------------
@app.post("/predict")
def predict_news(data: NewsInput):

    text_lower = data.text.lower()

    # -------- OPTION 2: OFFICIAL SOURCE BYPASS --------
    official_keywords = [
        "eci.gov.in",
        "pib.gov.in",
        "gov.in",
        "election commission of india",
        "press information bureau",
        "government of india"
    ]

    for keyword in official_keywords:
        if keyword in text_lower:
            return {
                "label": "REAL (Official Source)",
                "confidence": 100
            }

    # -------- AI MODEL CHECK --------
    result = classifier(data.text)[0]

    raw_label = result["label"]
    confidence = round(result["score"] * 100, 2)

    # -------- OPTION 1: SAFE INTERPRETATION --------
    if raw_label == "LABEL_0" and confidence >= 80:
        label = "LIKELY FAKE (AI Assessment)"
    elif raw_label == "LABEL_1" and confidence >= 80:
        label = "TRUSTWORTHY STYLE (AI Assessment)"
    else:
        label = "UNCERTAIN"

    return {
        "label": label,
        "confidence": confidence
    }
# ================= VERIFIED NEWS (SERVER-SIDE) =================

@app.get("/verified-news")
def get_verified_news():
    """
    Fetch verified election-related news using NewsAPI.
    """

    url = "https://newsapi.org/v2/everything"

    params = {
        "q": "Election Commission OR voter OR voting OR Lok Sabha",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 6,
        "apiKey": NEWS_API_KEY
    }

    response = requests.get(url, params=params, timeout=10)
    data = response.json()

    verified_news = []

    if "articles" not in data:
        return []

    trusted_domains = [
        "thehindu.com",
        "indianexpress.com",
        "ndtv.com",
        "hindustantimes.com",
        "timesofindia.indiatimes.com",
        "pib.gov.in",
        "eci.gov.in"
    ]

    for article in data["articles"]:
        link = article.get("url", "")
        title = article.get("title", "")

        if not link or not title:
            continue

        if not any(domain in link for domain in trusted_domains):
            continue

        verified_news.append({
            "title": title,
            "link": link,
            "source": article["source"]["name"],
            "verified": True,
            "published": article["publishedAt"][:10]
        })

    return verified_news
@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """
    Phase 5 (FINAL):
    - Image upload
    - Image description
    - OCR text extraction
    - Election relevance detection
    - Claim verification (fake / true / uncertain)
    """

    # ---------- LOAD IMAGE ----------
    image = Image.open(file.file).convert("RGB")

    # ---------- IMAGE DESCRIPTION ----------
    caption = image_captioner(image)[0]["generated_text"]

    # ---------- OCR WITH PREPROCESSING ----------
    open_cv_image = np.array(image)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    gray = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    custom_config = r"--oem 3 --psm 6"
    extracted_text = pytesseract.image_to_string(
        gray, config=custom_config
    ).strip()

    # ---------- ELECTION RELEVANCE ----------
    election_keywords = [
        "election",
        "vote",
        "voting",
        "ballot",
        "poll",
        "polling",
        "election commission",
        "eci",
        "lok sabha",
        "assembly"
    ]

    combined_text = (caption + " " + extracted_text).lower()
    election_related = any(
        keyword in combined_text for keyword in election_keywords
    )

    # ---------- NOT ELECTION RELATED ----------
    if not election_related:
        return {
            "description": caption,
            "ocr_text": extracted_text,
            "status": "NOT_ELECTION_RELATED",
            "message": "This image does not appear to be related to elections."
        }

    # ---------- NO CLAIM FOUND ----------
    if not extracted_text:
        return {
            "description": caption,
            "ocr_text": "",
            "status": "NO_CLAIM_FOUND",
            "message": (
                "The image is election-related, but no clear textual claim "
                "was detected. Please verify the source of this image."
            )
        }

    # ---------- VERIFY CLAIM USING AI MODEL ----------
    prediction = classifier(extracted_text)[0]
    raw_label = prediction["label"]
    confidence = round(prediction["score"] * 100, 2)

    # ---------- INTERPRET RESULT ----------
    if raw_label == "LABEL_0" and confidence >= 80:
        status = "LIKELY_FAKE"
        explanation = (
            "The claim extracted from this image appears to be misleading. "
            "There is no confirmation from trusted or official sources, "
            "which is common in viral misinformation."
        )

    elif raw_label == "LABEL_1" and confidence >= 80:
        status = "LIKELY_TRUE"
        explanation = (
            "The claim extracted from this image aligns with information "
            "typically reported by trusted sources. "
            "It is likely to be accurate, but users should still "
            "verify it through official announcements."
        )

    else:
        status = "UNCERTAIN"
        explanation = (
            "There is not enough information to confidently verify this claim. "
            "The claim may be incomplete, unclear, or lack reliable context."
        )

    # ---------- FINAL RESPONSE ----------
    return {
        "description": caption,
        "ocr_text": extracted_text,
        "election_related": True,
        "verification_status": status,
        "confidence": confidence,
        "explanation": explanation
    }
class ReportInput(BaseModel):
    type: str
    content: str
    source: str | None = None

@app.post("/report-misinformation")
def report_misinformation(data: ReportInput):
    report = {
        "type": data.type,
        "content": data.content,
        "source": data.source,
        "timestamp": datetime.utcnow().isoformat()
    }

    file_path = "reported_misinformation.json"

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            reports = json.load(f)
    else:
        reports = []

    reports.append(report)

    with open(file_path, "w") as f:
        json.dump(reports, f, indent=2)

    return {"status": "success"}
