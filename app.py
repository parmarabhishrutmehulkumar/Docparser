from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import logging
import requests
import uuid
from datetime import datetime, timezone, timedelta
import json
import io
import threading
import pandas as pd
import numpy as np
import skfuzzy as fuzz
import pdfplumber

PYTESSERACT_AVAILABLE = False
try:
    import pytesseract
    from PIL import Image, ImageFilter, ImageEnhance
    pytesseract.get_tesseract_version()
    PYTESSERACT_AVAILABLE = True
except Exception:
    pass

EASYOCR_AVAILABLE = False
EASY_OCR_READER = None
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception:
    pass

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)
logger.info(
    "OCR availability — pytesseract: %s | easyocr: %s",
    PYTESSERACT_AVAILABLE, EASYOCR_AVAILABLE
)

ALLOWED_EXTENSIONS    = {"pdf", "doc", "docx", "txt"}
MAX_FILE_SIZE         = 20 * 1024 * 1024
MIN_EMBEDDED_TEXT_LEN = 50
SESSION_TTL_MINUTES   = 60

N8N_WEBHOOK_URL = os.environ.get(
    "N8N_WEBHOOK_URL",
    "http://localhost:5678/webhook-test/a6fdd077-5e86-4d4f-bebf-7178962fb86e",
)


_store: dict = {}
_store_lock = threading.Lock()


def _store_set(session_id: str, payload: dict) -> None:
    now = datetime.now(timezone.utc)
    with _store_lock:
        entry = _store.get(session_id, {})
        entry.update(payload)
        entry.setdefault("created_at", now)
        entry["updated_at"] = now
        _store[session_id] = entry


def _store_get(session_id: str) -> dict | None:
    with _store_lock:
        return _store.get(session_id)


def _purge_expired_sessions() -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=SESSION_TTL_MINUTES)
    with _store_lock:
        expired = [
            sid for sid, v in _store.items()
            if v.get("created_at", datetime.now(timezone.utc)) < cutoff
        ]
        for sid in expired:
            del _store[sid]
    if expired:
        logger.info("Purged %d expired session(s)", len(expired))



def classify_admission(text) -> str:
    """
    Reads the Academic admission requirements text for one program.
    Returns: "Strict" | "Moderate" | "Lenient"
    """
    if pd.isna(text):
        return "Lenient"
    t = str(text).lower()

    strict_keywords = [
        "excellent academic record", "outstanding academic record",
        "excellent performance", "outstanding performance",
        "above-average performance", "above-average academic",
        "top 10", "top 5", "top students",
        "highly competitive", "limited number of places",
        "limited number of seats", "limited capacity",
        "selection committee", "selection based on",
        "prior research experience", "research experience is required",
        "first-class degree", "honours degree", "honors degree",
        "restricted admission", "numerus clausus",
        "must have successfully completed"
    ]
    for kw in strict_keywords:
        if kw in t:
            return "Strict"

    moderate_keywords = [
        "bachelor's degree in", "bachelors degree in", "bachelor degree in",
        "relevant bachelor's degree", "relevant degree in",
        "related field", "related discipline", "background in",
        "minimum grade", "grade of at least", "overall grade",
        "good academic standing", "ielts", "toefl",
        "letter of motivation", "statement of purpose",
        "letter of recommendation", "applicants must hold",
        "admission requires", "admission requirement"
    ]
    for kw in moderate_keywords:
        if kw in t:
            return "Moderate"

    lenient_keywords = [
        "graduates from all disciplines",
        "open to graduates of any discipline",
        "no specific background", "no specific degree",
        "completed undergraduate degree", "first academic degree",
        "university degree in any discipline"
    ]
    for kw in lenient_keywords:
        if kw in t:
            return "Lenient"

    return "Lenient"


def load_daad_dataset(path: str = "DAAD_Dataset_Cleaned.csv") -> pd.DataFrame:
    logger.info("Loading DAAD dataset from: %s", path)
    df = pd.read_csv(path, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    df = df.drop_duplicates()

    for col in ["Tuition fees per semester in EUR", "Semester contribution",
                "Contribution per semester", "Total contribution"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                       .str.replace(".", "", regex=False)
                       .str.extract("([0-9]+)", expand=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Teaching language"] = df["Teaching language"].astype(str).str.lower().str.strip()

    keep_cols = [
        "Course ID", "University", "Programme", "Degree",
        "Teaching language", "City", "Duration_in_semesters",
        "Tuition fees per semester in EUR", "Contribution per semester",
        "Total contribution", "Academic admission requirements",
        "Description/content", "Master", "Bachelor", "PhD"
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

    df["admission_strictness"] = df["Academic admission requirements"].apply(classify_admission)
    df["admission_strictness_score"] = df["admission_strictness"].map(
        {"Lenient": 0.3, "Moderate": 0.6, "Strict": 0.9}
    )

    logger.info("Dataset loaded: %d programs", len(df))
    return df



df_daad = load_daad_dataset("DAAD_Dataset_Cleaned.csv")


def run_fuzzy_model(cgpa: float, test_score: float, admission_strictness: float) -> float:
    """
    Inputs:
      cgpa                 — 0 to 10 scale
      test_score           — IELTS band (0–9). Converted internally to 0–120.
      admission_strictness — 0.3 (Lenient) | 0.6 (Moderate) | 0.9 (Strict)

    Output:
      suitability_score — float between 0 and 1
    """


    if test_score <= 9:
        test_score = test_score * (120 / 9)

   
    x_cgpa        = np.linspace(0, 10,  100)
    x_test        = np.linspace(0, 120, 100)
    x_strict      = np.linspace(0, 1,   100)
    x_suitability = np.linspace(0, 1,   100)

   
    cgpa_low  = fuzz.trimf(x_cgpa, [0,   0,   6.5])
    cgpa_med  = fuzz.trimf(x_cgpa, [5,   7,   9  ])
    cgpa_high = fuzz.trimf(x_cgpa, [7.5, 10,  10 ])

   
    test_weak   = fuzz.trimf(x_test, [0,   0,   85 ])
    test_avg    = fuzz.trimf(x_test, [80,  100, 110])
    test_strong = fuzz.trimf(x_test, [105, 120, 120])


    strict_lenient = fuzz.trimf(x_strict, [0,   0,   0.4])
    strict_mod     = fuzz.trimf(x_strict, [0.3, 0.5, 0.7])
    strict_strict  = fuzz.trimf(x_strict, [0.6, 1,   1  ])

    
    suit_low  = fuzz.trimf(x_suitability, [0,   0,   0.4])
    suit_med  = fuzz.trimf(x_suitability, [0.3, 0.5, 0.7])
    suit_high = fuzz.trimf(x_suitability, [0.6, 1,   1  ])

   
    cgpa_l = fuzz.interp_membership(x_cgpa, cgpa_low,  cgpa)
    cgpa_m = fuzz.interp_membership(x_cgpa, cgpa_med,  cgpa)
    cgpa_h = fuzz.interp_membership(x_cgpa, cgpa_high, cgpa)

    test_w = fuzz.interp_membership(x_test, test_weak,   test_score)
    test_a = fuzz.interp_membership(x_test, test_avg,    test_score)
    test_s = fuzz.interp_membership(x_test, test_strong, test_score)

    strict_l = fuzz.interp_membership(x_strict, strict_lenient, admission_strictness)
    strict_m = fuzz.interp_membership(x_strict, strict_mod,     admission_strictness)
    strict_s = fuzz.interp_membership(x_strict, strict_strict,  admission_strictness)

  
    rule1_out = np.fmin(np.fmin(np.fmin(cgpa_h, test_s), strict_s), suit_high)  # High+Strong+Strict  → High
    rule2_out = np.fmin(np.fmin(np.fmin(cgpa_h, test_s), strict_m), suit_high)  # High+Strong+Mod     → High
    rule3_out = np.fmin(np.fmin(np.fmin(cgpa_m, test_a), strict_m), suit_med)   # Med+Avg+Mod         → Med
    rule4_out = np.fmin(np.fmin(np.fmin(cgpa_m, test_a), strict_l), suit_med)   # Med+Avg+Lenient     → Med
    rule5_out = np.fmin(np.fmin(np.fmin(cgpa_h, test_a), strict_m), suit_high)  # High+Avg+Mod        → High
    rule6_out = np.fmin(np.fmin(cgpa_l, strict_l),                  suit_low)   # Low+Lenient         → Low
    rule7_out = np.fmin(np.fmin(cgpa_l, strict_s),                  suit_low)   # Low+Strict          → Low
    rule8_out = np.fmin(np.fmin(np.fmin(cgpa_m, test_s), strict_l), suit_med)   # Med+Strong+Lenient  → Med

  
    aggregated = rule1_out
    for r in [rule2_out, rule3_out, rule4_out,
              rule5_out, rule6_out, rule7_out, rule8_out]:
        aggregated = np.fmax(aggregated, r)

   
    try:
        final_score = fuzz.defuzz(x_suitability, aggregated, "centroid")
    except Exception:
        final_score = 0.0

    return round(final_score, 3)



def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _preprocess_for_ocr(pil_img):
    pil_img = pil_img.convert("L")
    pil_img = pil_img.filter(ImageFilter.SHARPEN)
    pil_img = ImageEnhance.Contrast(pil_img).enhance(2.0)
    return pil_img


def _run_pytesseract(page, page_num: int) -> str | None:
    try:
        img = page.to_image(resolution=400).original
        img = _preprocess_for_ocr(img)
        text = pytesseract.image_to_string(img, config="--psm 6")
        if text and text.strip():
            logger.info("pytesseract OK  page=%d  chars=%d", page_num, len(text.strip()))
            return text.strip()
    except Exception as exc:
        logger.warning("pytesseract error page=%d: %s", page_num, exc)
    return None


def _run_easyocr(page, page_num: int) -> str | None:
    global EASY_OCR_READER
    try:
        if EASY_OCR_READER is None:
            logger.info("Initialising EasyOCR reader…")
            EASY_OCR_READER = easyocr.Reader(["en"], gpu=False)
        img = page.to_image(resolution=300).original.convert("RGB")
        results = EASY_OCR_READER.readtext(np.array(img))
        if results:
            text = "\n".join(r[1] for r in results)
            logger.info("EasyOCR OK  page=%d  chars=%d", page_num, len(text.strip()))
            return text.strip()
    except Exception as exc:
        logger.warning("EasyOCR error page=%d: %s", page_num, exc)
    return None


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    texts: list[str] = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            logger.info("PDF opened: %d page(s)", len(pdf.pages))
            for i, page in enumerate(pdf.pages, start=1):
                page_text = None
                try:
                    page_text = page.extract_text()
                except Exception as exc:
                    logger.warning("pdfplumber error page=%d: %s", i, exc)

                char_count = len(page_text.strip()) if page_text else 0
                if page_text and char_count >= MIN_EMBEDDED_TEXT_LEN:
                    texts.append(page_text.strip())
                    continue

                logger.info("Page %d — trying OCR…", i)
                ocr_text = None
                if PYTESSERACT_AVAILABLE:
                    ocr_text = _run_pytesseract(page, i)
                if not ocr_text and EASYOCR_AVAILABLE:
                    ocr_text = _run_easyocr(page, i)
                if ocr_text:
                    texts.append(ocr_text)
                else:
                    logger.warning("Page %d — no text extracted", i)
    except Exception as exc:
        logger.exception("Failed to process PDF: %s", exc)

    combined = "\n\n".join(texts).strip()
    logger.info("Extraction complete: %d chars", len(combined))
    return combined


def _forward_to_n8n(session_id, file_bytes, filename, mimetype, preferences, parsed_text):
    try:
        logger.info("Forwarding to n8n — session=%s", session_id)
        resp = requests.post(
            N8N_WEBHOOK_URL,
            files={"file": (filename, io.BytesIO(file_bytes), mimetype or "application/pdf")},
            data={
                "sessionId":   session_id,
                "filename":    filename,
                "preferences": json.dumps(preferences),
                "parsedText":  parsed_text,
            },
            timeout=90,
        )
        if not resp.ok:
            logger.warning("n8n returned %d: %s", resp.status_code, resp.text[:500])
            _store_set(session_id, {"status": "failed", "error": f"n8n returned HTTP {resp.status_code}"})
            return
        try:
            body = resp.json()
            processed = (body.get("processedData") or body.get("parsedData")
                         or body.get("data") or body.get("result"))
            if processed:
                _store_set(session_id, {"status": "completed", "data": processed})
                return
        except Exception:
            pass
        logger.info("n8n accepted — waiting for callback — session=%s", session_id)
    except requests.exceptions.RequestException as exc:
        logger.exception("n8n request failed — session=%s: %s", session_id, exc)
        _store_set(session_id, {"status": "failed", "error": str(exc)})


@app.route("/api/submit-preferences", methods=["POST"])
def submit_preferences():
    try:
        content_length = request.content_length
        if content_length and content_length > MAX_FILE_SIZE + 10_240:
            return jsonify({"success": False, "error": "Request body too large."}), 413

        session_id  = request.form.get("sessionId") or str(uuid.uuid4())
        preferences = {
            "fieldOfStudy":    request.form.get("fieldOfStudy"),
            "degreeLevel":     request.form.get("degreeLevel"),
            "location":        request.form.get("location"),
            "courseLanguage":  request.form.get("courseLanguage"),
            "additionalPrefs": request.form.get("additionalPrefs", ""),
        }

        resume_queued = False
        if "resume" in request.files:
            file = request.files["resume"]
            if file and file.filename:
                if not allowed_file(file.filename):
                    return jsonify({"success": False, "error": "Invalid file type."}), 400
                file.stream.seek(0)
                file_bytes = file.read()
                if len(file_bytes) > MAX_FILE_SIZE:
                    return jsonify({"success": False, "error": "File exceeds 20 MB limit."}), 400
                filename    = secure_filename(file.filename)
                mimetype    = file.mimetype or "application/pdf"
                parsed_text = extract_text_from_pdf_bytes(file_bytes)
                _store_set(session_id, {"status": "processing", "data": None, "error": None})
                threading.Thread(
                    target=_forward_to_n8n,
                    args=(session_id, file_bytes, filename, mimetype, preferences, parsed_text),
                    daemon=True,
                ).start()
                resume_queued = True

        if not resume_queued:
            _store_set(session_id, {"status": "no_resume", "data": None, "error": None})

        threading.Thread(target=_purge_expired_sessions, daemon=True).start()

        return jsonify({
            "success":      True,
            "sessionId":    session_id,
            "resumeQueued": resume_queued,
            "message":      "Resume queued. Poll /api/status/<sessionId>."
                            if resume_queued else "Preferences received. No resume uploaded.",
        }), 202

    except Exception as exc:
        logger.exception("Error in /api/submit-preferences: %s", exc)
        return jsonify({"success": False, "error": "Internal server error."}), 500


@app.route("/api/status/<session_id>", methods=["GET"])
def get_status(session_id: str):
    entry = _store_get(session_id)
    if entry is None:
        return jsonify({"success": False, "status": "not_found", "message": "Unknown session ID."}), 404

    status    = entry.get("status", "processing")
    http_code = 200 if status in ("completed", "failed", "no_resume") else 202
    return jsonify({
        "success":   status == "completed",
        "sessionId": session_id,
        "status":    status,
        "data":      entry.get("data"),
        "error":     entry.get("error"),
        "updatedAt": entry.get("updated_at", datetime.now(timezone.utc)).isoformat(),
    }), http_code


@app.route("/receive-extracted-data", methods=["POST"])
def n8n_callback():
    try:
        data = request.get_json(force=True, silent=True) or {}
        if not data:
            raw = request.data.decode("utf-8", errors="replace")
            try:
                data = json.loads(raw)
            except Exception:
                data = {"raw": raw}

        session_id = data.get("sessionId")
        if not session_id:
            return jsonify({"success": False, "error": "sessionId required"}), 400

        processed = (data.get("processedData") or data.get("parsedData")
                     or data.get("data") or data.get("result"))
        _store_set(session_id, {"status": "completed", "data": processed, "error": None})
        logger.info("n8n callback stored — session=%s", session_id)
        return jsonify({"success": True}), 200

    except Exception as exc:
        logger.exception("n8n callback error: %s", exc)
        return jsonify({"success": False, "error": "Internal server error."}), 500


@app.route("/api/fuzzy-score", methods=["POST"])
def fuzzy_score():
    """
    Called by n8n after extracting student details from resume.

    Expected JSON body:
    {
        "cgpa":          8.5,
        "ielts_score":   7.0,
        "degree_filter": "Master"
    }

    Returns top 10 recommended programs sorted by suitability score.
    """
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"success": False, "error": "No JSON body received"}), 400

        cgpa          = float(data.get("cgpa", 0))
        test_score    = float(data.get("ielts_score", 0))
        degree_filter = str(data.get("degree_filter", "Master")).strip()

        # Filter by degree type
        if degree_filter == "Master":
            df_filtered = df_daad[df_daad["Master"] == 1].copy()
        elif degree_filter == "Bachelor":
            df_filtered = df_daad[df_daad["Bachelor"] == 1].copy()
        elif degree_filter == "PhD":
            df_filtered = df_daad[df_daad["PhD"] == 1].copy()
        else:
            df_filtered = df_daad.copy()

        # Score every program in the filtered dataset
        results = []
        for _, row in df_filtered.iterrows():
            score = run_fuzzy_model(
                cgpa,
                test_score,
                float(row["admission_strictness_score"])
            )
            results.append({
                "university":           row["University"],
                "programme":            row["Programme"],
                "degree":               row["Degree"],
                "city":                 row["City"],
                "suitability_score":    score,
                "admission_strictness": row["admission_strictness"],
                "tuition_fees":         row.get("Tuition fees per semester in EUR", "N/A"),
            })

        # Sort by score descending, return top 10
        results = sorted(results, key=lambda x: x["suitability_score"], reverse=True)[:10]

        return jsonify({"success": True, "recommendations": results}), 200

    except Exception as e:
        logger.exception("Fuzzy model error: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status":            "healthy",
        "programs_loaded":   len(df_daad),
        "ocr": {
            "pytesseract": PYTESSERACT_AVAILABLE,
            "easyocr":     EASYOCR_AVAILABLE,
        },
        "sessions_in_store": len(_store),
        "n8n_webhook":       N8N_WEBHOOK_URL,
    }), 200



if __name__ == "__main__":
    logger.info("Starting backend on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)