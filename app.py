"""
app.py — Resume Parser + n8n Integration (Production-grade, async pattern)

Flow:
  Frontend  ──POST /api/submit-preferences──►  app.py
                                                  │
                                          parse PDF text
                                       (embedded or OCR fallback)
                                                  │
                                      POST to n8n webhook (async)
                                                  │
                                    return { sessionId } immediately
                                                  │
  Frontend  ──GET /api/status/<sessionId>──►  app.py  (polls until done)
                                                  │
  n8n  ──POST /receive-extracted-data──►  app.py  (stores result)
                                                  │
  Frontend  ──GET /api/status/<sessionId>──►  { status: completed, data: ... }
"""

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

# ── PDF parsing ────────────────────────────────────────────────────────────────
import pdfplumber

# ── OCR: pytesseract (primary) ─────────────────────────────────────────────────
PYTESSERACT_AVAILABLE = False
try:
    import pytesseract
    from PIL import Image, ImageFilter, ImageEnhance
    pytesseract.get_tesseract_version()          # fail fast if binary missing
    PYTESSERACT_AVAILABLE = True
except Exception:
    pass

# ── OCR: EasyOCR (fallback) ────────────────────────────────────────────────────
EASYOCR_AVAILABLE = False
EASY_OCR_READER = None   # cached — never reinitialise per page
try:
    import easyocr
    import numpy as np
    EASYOCR_AVAILABLE = True
except Exception:
    pass

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)
logger.info("OCR availability — pytesseract: %s | easyocr: %s", PYTESSERACT_AVAILABLE, EASYOCR_AVAILABLE)

# ── Config ─────────────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS      = {"pdf", "doc", "docx", "txt"}
MAX_FILE_SIZE           = 20 * 1024 * 1024          # 20 MB
MIN_EMBEDDED_TEXT_LEN   = 50                         # chars; below → treat as image-PDF
SESSION_TTL_MINUTES     = 60                         # sessions older than this are purged

N8N_WEBHOOK_URL = os.environ.get(
    "N8N_WEBHOOK_URL",
    "http://localhost:5678/webhook-test/a6fdd077-5e86-4d4f-bebf-7178962fb86e",
)

# ── In-memory session store with TTL ──────────────────────────────────────────
#
# Structure per session:
#   {
#     "status":      "processing" | "completed" | "failed",
#     "data":        <n8n response payload or None>,
#     "error":       <error message or None>,
#     "created_at":  datetime (UTC),
#     "updated_at":  datetime (UTC),
#   }
#
_store: dict[str, dict] = {}
_store_lock = threading.Lock()


def _store_set(session_id: str, payload: dict) -> None:
    """Write/update a session entry (thread-safe)."""
    now = datetime.now(timezone.utc)
    with _store_lock:
        entry = _store.get(session_id, {})
        entry.update(payload)
        entry.setdefault("created_at", now)
        entry["updated_at"] = now
        _store[session_id] = entry


def _store_get(session_id: str) -> dict | None:
    """Read a session entry (thread-safe)."""
    with _store_lock:
        return _store.get(session_id)


def _purge_expired_sessions() -> None:
    """
    Remove sessions older than SESSION_TTL_MINUTES.
    Called in a background thread after each new session is created.
    Keeps memory from growing unbounded on long-running instances.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=SESSION_TTL_MINUTES)
    with _store_lock:
        expired = [sid for sid, v in _store.items() if v.get("created_at", datetime.now(timezone.utc)) < cutoff]
        for sid in expired:
            del _store[sid]
    if expired:
        logger.info("Purged %d expired session(s) from store", len(expired))


# ── Helpers ────────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _preprocess_for_ocr(pil_img):
    """Grayscale → sharpen → contrast boost. Measurably improves OCR on scans."""
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
        logger.warning("pytesseract returned empty text on page %d", page_num)
    except Exception as exc:
        logger.warning("pytesseract error page=%d: %s", page_num, exc)
    return None


def _run_easyocr(page, page_num: int) -> str | None:
    global EASY_OCR_READER
    try:
        if EASY_OCR_READER is None:
            logger.info("Initialising EasyOCR reader (first use)…")
            EASY_OCR_READER = easyocr.Reader(["en"], gpu=False)

        img = page.to_image(resolution=300).original.convert("RGB")
        results = EASY_OCR_READER.readtext(np.array(img))
        if results:
            text = "\n".join(r[1] for r in results)
            logger.info("EasyOCR OK  page=%d  chars=%d", page_num, len(text.strip()))
            return text.strip()
        logger.warning("EasyOCR returned no results on page %d", page_num)
    except Exception as exc:
        logger.warning("EasyOCR error page=%d: %s", page_num, exc)
    return None


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Extract text from PDF bytes using a tiered strategy per page:
      1. pdfplumber embedded text  (fast, accurate for digital PDFs)
      2. pytesseract OCR            (if embedded text is sparse/absent)
      3. EasyOCR                    (fallback if pytesseract fails/unavailable)

    The `force_ocr` client flag has been intentionally removed — server decides
    the extraction strategy based on actual content, not client hints.
    """
    texts: list[str] = []

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            logger.info("PDF opened: %d page(s)", len(pdf.pages))

            for i, page in enumerate(pdf.pages, start=1):
                # Step 1 — embedded text
                page_text: str | None = None
                try:
                    page_text = page.extract_text()
                except Exception as exc:
                    logger.warning("pdfplumber error page=%d: %s", i, exc)

                char_count = len(page_text.strip()) if page_text else 0
                logger.info("Page %d — embedded chars: %d", i, char_count)

                if page_text and char_count >= MIN_EMBEDDED_TEXT_LEN:
                    texts.append(page_text.strip())
                    continue

                # Step 2 — OCR fallback
                logger.info("Page %d — insufficient embedded text, trying OCR…", i)
                ocr_text: str | None = None

                if PYTESSERACT_AVAILABLE:
                    ocr_text = _run_pytesseract(page, i)

                if not ocr_text and EASYOCR_AVAILABLE:
                    ocr_text = _run_easyocr(page, i)

                if ocr_text:
                    texts.append(ocr_text)
                else:
                    logger.warning("Page %d — no text extracted by any method", i)

    except Exception as exc:
        logger.exception("Failed to process PDF: %s", exc)

    combined = "\n\n".join(texts).strip()
    logger.info("Extraction complete: %d chars from %d page(s)", len(combined), len(texts))
    return combined


def _forward_to_n8n(
    session_id: str,
    file_bytes: bytes,
    filename: str,
    mimetype: str,
    preferences: dict,
    parsed_text: str,
) -> None:
    """
    Called in a background thread.
    POSTs file + parsed text + metadata to the n8n webhook.
    Updates session store with 'failed' status on error.
    n8n is expected to call back /receive-extracted-data when done.
    """
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
            _store_set(session_id, {
                "status": "failed",
                "error":  f"n8n returned HTTP {resp.status_code}",
            })
            return

        # n8n may respond immediately (synchronous workflow) or via callback.
        # Handle both gracefully.
        try:
            body = resp.json()
            processed = (
                body.get("processedData")
                or body.get("parsedData")
                or body.get("data")
                or body.get("result")
            )
            if processed:
                logger.info("n8n responded synchronously with data — session=%s", session_id)
                _store_set(session_id, {"status": "completed", "data": processed})
                return
        except Exception:
            pass  # response was not JSON — wait for callback

        logger.info("n8n accepted request — waiting for callback — session=%s", session_id)
        # status remains "processing"; n8n will call /receive-extracted-data

    except requests.exceptions.RequestException as exc:
        logger.exception("n8n request failed — session=%s: %s", session_id, exc)
        _store_set(session_id, {"status": "failed", "error": str(exc)})


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/api/submit-preferences", methods=["POST"])
def submit_preferences():
    """
    Accepts multipart/form-data with optional resume PDF + user preferences.
    Returns immediately with { success, sessionId } — client polls /api/status/<id>.
    """
    try:
        # ── Enforce max content-length before reading body ─────────────────────
        content_length = request.content_length
        if content_length and content_length > MAX_FILE_SIZE + 10_240:   # +10KB for form fields
            return jsonify({"success": False, "error": "Request body too large."}), 413

        session_id = request.form.get("sessionId") or str(uuid.uuid4())

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
                    return jsonify({"success": False, "error": "Invalid file type. Allowed: pdf, doc, docx, txt"}), 400

                file.stream.seek(0)
                file_bytes = file.read()

                if len(file_bytes) > MAX_FILE_SIZE:
                    return jsonify({"success": False, "error": "File exceeds 20 MB limit."}), 400

                filename = secure_filename(file.filename)
                mimetype = file.mimetype or "application/pdf"

                # ── Parse text synchronously (fast for digital PDFs) ───────────
                parsed_text = extract_text_from_pdf_bytes(file_bytes)
                logger.info("Parsed text length=%d session=%s", len(parsed_text), session_id)

                # ── Register session before spawning thread ────────────────────
                _store_set(session_id, {"status": "processing", "data": None, "error": None})

                # ── Forward to n8n in background — don't block the response ────
                thread = threading.Thread(
                    target=_forward_to_n8n,
                    args=(session_id, file_bytes, filename, mimetype, preferences, parsed_text),
                    daemon=True,
                )
                thread.start()
                resume_queued = True

        if not resume_queued:
            _store_set(session_id, {"status": "no_resume", "data": None, "error": None})

        # ── Async: return immediately, client polls ────────────────────────────
        response_body = {
            "success":      True,
            "sessionId":    session_id,
            "resumeQueued": resume_queued,
            "message":      "Resume received and queued for processing. Poll /api/status/<sessionId> for results."
            if resume_queued else "Preferences received. No resume uploaded.",
        }

        # Purge old sessions in background (housekeeping)
        threading.Thread(target=_purge_expired_sessions, daemon=True).start()

        return jsonify(response_body), 202   # 202 Accepted — processing in progress

    except Exception as exc:
        logger.exception("Error in /api/submit-preferences: %s", exc)
        return jsonify({"success": False, "error": "Internal server error."}), 500


@app.route("/api/status/<session_id>", methods=["GET"])
def get_status(session_id: str):
    """
    Frontend polls this endpoint.
    Returns current status + data once n8n has responded.

    Responses:
      202 — still processing
      200 — completed (data available)
      200 — failed (error message available)
      404 — unknown session
    """
    entry = _store_get(session_id)
    if entry is None:
        return jsonify({"success": False, "status": "not_found", "message": "Unknown session ID."}), 404

    status = entry.get("status", "processing")
    http_code = 200 if status in ("completed", "failed", "no_resume") else 202

    return jsonify({
        "success":    status == "completed",
        "sessionId":  session_id,
        "status":     status,
        "data":       entry.get("data"),
        "error":      entry.get("error"),
        "updatedAt":  entry.get("updated_at", datetime.now(timezone.utc)).isoformat(),
    }), http_code


@app.route("/receive-extracted-data", methods=["POST"])
def n8n_callback():
    """
    n8n calls this endpoint when its workflow finishes.
    Stores the result so the next /api/status poll returns it.
    """
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
            logger.warning("n8n callback missing sessionId: %s", data)
            return jsonify({"success": False, "error": "sessionId required"}), 400

        processed = (
            data.get("processedData")
            or data.get("parsedData")
            or data.get("data")
            or data.get("result")
        )

        _store_set(session_id, {"status": "completed", "data": processed, "error": None})
        logger.info("n8n callback stored — session=%s", session_id)
        return jsonify({"success": True}), 200

    except Exception as exc:
        logger.exception("n8n callback error: %s", exc)
        return jsonify({"success": False, "error": "Internal server error."}), 500


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status":  "healthy",
        "ocr": {
            "pytesseract": PYTESSERACT_AVAILABLE,
            "easyocr":     EASYOCR_AVAILABLE,
        },
        "sessions_in_store": len(_store),
        "n8n_webhook":       N8N_WEBHOOK_URL,
    }), 200


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting backend on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)