from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import logging
import requests
import uuid
from datetime import datetime, timezone
import json
import io
import time

# PDF parsing + optional OCR
import pdfplumber
PYTESSERACT_AVAILABLE = False
EASYOCR_AVAILABLE = False
try:
    import pytesseract
    from PIL import Image
    PYTESSERACT_AVAILABLE = True
except Exception:
    PYTESSERACT_AVAILABLE = False

try:
    import easyocr
    import numpy as np
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}
MAX_FILE_SIZE = 20 * 1024 * 1024  # increase if needed

# n8n webhook URL (update if different)
N8N_WEBHOOK_URL = "http://localhost:5678/webhook-test/7ff03c11-e208-46a9-8096-4e579bba78d5"

processed_results = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf_bytes(pdf_bytes):
    """
    Try to extract embedded text using pdfplumber.
    If no embedded text found, attempt OCR using pytesseract or EasyOCR.
    Returns combined text (string).
    """
    texts = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    page_text = page.extract_text()
                except Exception as e:
                    logger.debug("pdfplumber page.extract_text error on page %s: %s", i, e)
                    page_text = None

                if page_text and page_text.strip():
                    texts.append(page_text)
                    continue

                # No embedded text: attempt OCR
                ocr_text = None

                if PYTESSERACT_AVAILABLE:
                    try:
                        pil_img = page.to_image(resolution=300).original.convert("RGB")
                        ocr_text = pytesseract.image_to_string(pil_img)
                        if ocr_text and ocr_text.strip():
                            texts.append(ocr_text)
                            continue
                    except Exception as e:
                        logger.debug("pytesseract OCR error page %s: %s", i, e)

                if EASYOCR_AVAILABLE:
                    try:
                        pil_img = page.to_image(resolution=300).original.convert("RGB")
                        img_np = np.array(pil_img)
                        reader = easyocr.Reader(['en'], gpu=False)
                        results = reader.readtext(img_np)
                        if results:
                            ocr_text = "\n".join([r[1] for r in results])
                            texts.append(ocr_text)
                            continue
                    except Exception as e:
                        logger.debug("EasyOCR error page %s: %s", i, e)

                logger.debug("Page %s: no text extracted (embedded/OCR)", i)

    except Exception as e:
        logger.exception("Failed to open PDF bytes with pdfplumber: %s", e)

    combined = "\n\n".join(texts).strip()
    return combined

def send_to_n8n(file_bytes, filename, mimetype, session_id, preferences, parsed_text):
    """
    Send file + parsed text + metadata to n8n webhook.
    Returns True on 2xx response, False otherwise.
    """
    files = {
        'file': (filename, io.BytesIO(file_bytes), mimetype or 'application/pdf')
    }
    data = {
        'sessionId': session_id,
        'filename': filename,
        'preferences': json.dumps(preferences or {}),
        'parsedText': parsed_text or ''
    }
    try:
        logger.info("Sending file+parsedText to n8n at %s (session=%s)", N8N_WEBHOOK_URL, session_id)
        resp = requests.post(N8N_WEBHOOK_URL, files=files, data=data, timeout=90)
        if not resp.ok:
            logger.warning("n8n returned status %s: %s", resp.status_code, resp.text[:1000])
            return False

        try:
            result = resp.json()
            logger.info("n8n response JSON: %s", result)
            # if n8n returns immediate processed data, store it
            if isinstance(result, dict) and result.get('success') and result.get('processedData'):
                processed_results[session_id] = {
                    'status': 'completed',
                    'data': result.get('processedData'),
                    'received_at': datetime.now(timezone.utc).isoformat()
                }
                return True
        except Exception:
            # non-json response is acceptable if status OK
            logger.debug("n8n response not JSON")

        processed_results[session_id] = {
            'status': 'processing',
            'data': None,
            'sent_at': datetime.now(timezone.utc).isoformat()
        }
        return True

    except requests.exceptions.RequestException as e:
        logger.exception("Request to n8n failed: %s", e)
        return False

@app.route('/api/submit-preferences', methods=['POST'])
def submit_preferences():
    try:
        preferences = {
            'fieldOfStudy': request.form.get('fieldOfStudy'),
            'degreeLevel': request.form.get('degreeLevel'),
            'location': request.form.get('location'),
            'courseLanguage': request.form.get('courseLanguage'),
            'additionalPrefs': request.form.get('additionalPrefs', '')
        }

        session_id = request.form.get("sessionId") or str(uuid.uuid4())
        resume_uploaded = False
        parsed_text = None
        file_meta = None

        # read optional wait timeout (seconds) from form, default 30s
        wait_timeout = int(request.form.get('waitTimeout', 30))

        if 'resume' in request.files:
            file = request.files['resume']
            if file and file.filename != '':
                if not allowed_file(file.filename):
                    return jsonify({'success': False, 'error': 'Invalid file type.'}), 400

                # read bytes for parsing and forwarding
                file.stream.seek(0)
                file_bytes = file.read()
                file_size = len(file_bytes)
                if file_size > MAX_FILE_SIZE:
                    return jsonify({'success': False, 'error': 'File too large (max 20MB).'}), 400

                file_meta = {
                    'filename': secure_filename(file.filename),
                    'mimetype': file.mimetype,
                    'size': file_size
                }

                # extract text using pdfplumber (+OCR fallback)
                parsed_text = extract_text_from_pdf_bytes(file_bytes) or ''
                logger.info("Extracted parsedText length=%s for session=%s", len(parsed_text), session_id)

                # send file bytes and parsedText to n8n
                sent = send_to_n8n(file_bytes, file_meta['filename'], file_meta['mimetype'], session_id, preferences, parsed_text)
                resume_uploaded = bool(sent)
                if not sent:
                    logger.warning("Forward to n8n failed for session %s", session_id)
                    # still continue — will return combined response with processed_status indicating failure

        # ensure an entry exists
        processed_results.setdefault(session_id, {
            'status': 'processing' if resume_uploaded else 'no_resume_uploaded',
            'data': None,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

        # WAIT for n8n result up to wait_timeout seconds (polling)
        processed_item = processed_results.get(session_id)
        if resume_uploaded and wait_timeout > 0:
            start = time.time()
            while time.time() - start < wait_timeout:
                item = processed_results.get(session_id)
                if item and item.get('status') == 'completed':
                    processed_item = item
                    break
                time.sleep(1)
        else:
            processed_item = processed_results.get(session_id)

        # build single merged document for ML downstream
        merged = {
            'sessionId': session_id,
            'preferences': preferences,
            'file': file_meta,                   # null if no file uploaded
            'parsed_text': parsed_text or '',
            'resume_uploaded': resume_uploaded,
            'processed_status': processed_item.get('status') if processed_item else 'unknown',
            'processed_output': processed_item.get('data') if processed_item else None,
            'created_at': processed_item.get('timestamp') if processed_item and processed_item.get('timestamp') else datetime.now(timezone.utc).isoformat()
        }

        return jsonify({'success': True, 'result': merged}), 200

    except Exception as e:
        logger.exception("Error in submit-preferences: %s", e)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/receive-extracted-data', methods=['POST'])
def n8n_callback():
    try:
        data = request.get_json(force=True, silent=True)
        if data is None:
            raw = request.data.decode('utf-8') or ''
            try:
                data = json.loads(raw) if raw else {}
            except Exception:
                data = {"raw": raw}

        session_id = data.get('sessionId')
        parsed_data = data.get('parsedData') or data.get('processedData') or data.get('data') or data.get('result')

        if not session_id:
            logger.warning("n8n callback missing sessionId: %s", data)
            return jsonify({"success": False, "error": "no sessionId provided"}), 400

        processed_results[session_id] = {
            "status": "completed",
            "data": parsed_data,
            "received_at": datetime.now(timezone.utc).isoformat()
        }

        logger.info("Stored n8n result for session %s", session_id)
        return jsonify({"success": True}), 200

    except Exception as e:
        logger.exception("n8n callback error: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/get-processed-data/<session_id>', methods=['GET'])
def get_processed_data(session_id):
    item = processed_results.get(session_id)
    if not item:
        return jsonify({"success": False, "status": "processing"}), 202
    return jsonify({"success": True, "status": item.get("status"), "data": item.get("data")}), 200

def generate_university_recommendations(preferences):
    # simple stub recommendations
    universities = {
        'computer-science': [
            {'name': 'MIT', 'location': 'Cambridge, MA', 'rank': 1},
            {'name': 'Stanford', 'location': 'Stanford, CA', 'rank': 2},
            {'name': 'Carnegie Mellon', 'location': 'Pittsburgh, PA', 'rank': 3}
        ],
        'engineering': [
            {'name': 'Caltech', 'location': 'Pasadena, CA', 'rank': 1},
            {'name': 'UC Berkeley', 'location': 'Berkeley, CA', 'rank': 2},
            {'name': 'Georgia Tech', 'location': 'Atlanta, GA', 'rank': 3}
        ],
        'business': [
            {'name': 'Harvard Business School', 'location': 'Boston, MA', 'rank': 1},
            {'name': 'Wharton', 'location': 'Philadelphia, PA', 'rank': 2},
            {'name': 'Stanford GSB', 'location': 'Stanford, CA', 'rank': 3}
        ]
    }
    field = (preferences or {}).get('fieldOfStudy', 'computer-science')
    return universities.get(field, universities['computer-science'])[:3]

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'University Finder API is running'})

if __name__ == '__main__':
    logger.info("Starting backend on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)