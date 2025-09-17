from flask import Flask, request, jsonify
from flask_cors import CORS
from docling.document_converter import DocumentConverter
from werkzeug.utils import secure_filename
import os
import logging
import requests
import uuid
from datetime import datetime

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# n8n webhook (keep your existing webhook URL)
N8N_WEBHOOK_URL = "http://localhost:5678/webhook-test/a6fdd077-5e86-4d4f-bebf-7178962fb86e"

processed_results = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_resume_content(file_path):
    converter = DocumentConverter()
    result = converter.convert(file_path)
    return result.document.export_to_markdown()

def send_to_n8n(filename, parsed_text, session_id):
    payload = {
        "filename": filename,
        "parsedText": parsed_text,
        "sessionId": session_id
    }
    try:
        resp = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=20)
        resp.raise_for_status()
        
        # Get the extracted data directly from the response
        result = resp.json()
        if result.get('success'):
            extracted_data = result.get('extractedData', {})
            
            # Store the extracted data in processed_results
            processed_results[session_id] = {
                'status': 'completed',
                'data': extracted_data,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Extracted data for session {session_id}: {extracted_data}")
            return True
        else:
            logger.error(f"n8n returned error for session {session_id}")
            return False
            
    except Exception as e:
        logger.error("Failed sending to n8n: %s", e)
        return False

def analyze_resume_content(content, preferences):
    field_keywords = {
        'computer-science': ['python', 'java', 'programming', 'software', 'developer', 'coding', 'algorithm'],
        'engineering': ['engineering', 'technical', 'design', 'project', 'CAD', 'matlab'],
        'business': ['management', 'business', 'marketing', 'sales', 'finance', 'consulting'],
        'medicine': ['medical', 'healthcare', 'patient', 'clinical', 'hospital', 'nursing'],
        'law': ['legal', 'law', 'court', 'attorney', 'litigation', 'contract'],
        'arts': ['creative', 'design', 'art', 'music', 'writing', 'literature'],
        'sciences': ['research', 'laboratory', 'analysis', 'data', 'experiment', 'scientific'],
        'social-sciences': ['psychology', 'sociology', 'research', 'social', 'community'],
        'education': ['teaching', 'education', 'curriculum', 'student', 'classroom'],
        'psychology': ['psychology', 'mental health', 'counseling', 'therapy', 'behavioral']
    }

    content_lower = (content or "").lower()
    field_of_study = preferences.get('fieldOfStudy', '')
    relevant_keywords = field_keywords.get(field_of_study, [])
    keyword_matches = sum(1 for keyword in relevant_keywords if keyword in content_lower)
    match_score = min(100, (keyword_matches / max(len(relevant_keywords), 1)) * 100)

    analysis = {
        'match_score': round(match_score, 2),
        'relevant_keywords_found': keyword_matches,
        'total_keywords_checked': len(relevant_keywords),
        'recommendations': []
    }

    if match_score >= 70:
        analysis['recommendations'].append("Your background strongly aligns with your chosen field!")
    elif match_score >= 40:
        analysis['recommendations'].append("Good alignment with some areas for skill development.")
    else:
        analysis['recommendations'].append("Consider highlighting relevant skills or exploring related fields.")

    return analysis

def generate_university_recommendations(preferences, resume_analysis=None):
    universities = {
        'computer-science': [
            {'name': 'MIT', 'location': 'Cambridge, MA', 'rank': 1, 'match_score': 95},
            {'name': 'Stanford University', 'location': 'Stanford, CA', 'rank': 2, 'match_score': 93},
            {'name': 'Carnegie Mellon', 'location': 'Pittsburgh, PA', 'rank': 3, 'match_score': 90}
        ],
        'engineering': [
            {'name': 'Caltech', 'location': 'Pasadena, CA', 'rank': 1, 'match_score': 94},
            {'name': 'UC Berkeley', 'location': 'Berkeley, CA', 'rank': 2, 'match_score': 92},
            {'name': 'Georgia Tech', 'location': 'Atlanta, GA', 'rank': 3, 'match_score': 88}
        ],
        'business': [
            {'name': 'Harvard Business School', 'location': 'Boston, MA', 'rank': 1, 'match_score': 96},
            {'name': 'Wharton School', 'location': 'Philadelphia, PA', 'rank': 2, 'match_score': 94},
            {'name': 'Stanford GSB', 'location': 'Stanford, CA', 'rank': 3, 'match_score': 92}
        ]
    }
    field = preferences.get('fieldOfStudy', 'computer-science')
    base_recommendations = universities.get(field, universities['computer-science'])
    if resume_analysis:
        resume_score = resume_analysis.get('match_score', 50)
        for uni in base_recommendations:
            adjustment = (resume_score - 50) * 0.1
            uni['match_score'] = min(100, max(0, uni['match_score'] + adjustment))
    return base_recommendations[:3]

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
        resume_analysis = None
        resume_content = None

        if 'resume' in request.files:
            file = request.files['resume']
            if file and file.filename != '':
                if not allowed_file(file.filename):
                    return jsonify({'success': False, 'error': 'Invalid file type.'}), 400

                file.seek(0, os.SEEK_END)
                file_size = file.tell()
                file.seek(0)
                if file_size > MAX_FILE_SIZE:
                    return jsonify({'success': False, 'error': 'File too large (max 5MB).'}), 400

                filename = secure_filename(file.filename)
                temp_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_{filename}")
                file.save(temp_path)
                logger.info(f"Saved resume file: {temp_path}")

                resume_content = extract_resume_content(temp_path)
                resume_analysis = analyze_resume_content(resume_content, preferences)

                # send parsed text to n8n
                send_to_n8n(filename, resume_content, session_id)

                try:
                    os.remove(temp_path)
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file: {e}")

        recommendations = generate_university_recommendations(preferences, resume_analysis)

        processed_results[session_id] = {
            'status': 'sent_to_n8n',
            'preview': resume_content[:1000] if resume_content else None,
            'timestamp': datetime.utcnow().isoformat()
        }

        response = {
            'success': True,
            'sessionId': session_id,
            'data': {
                'preferences': preferences,
                'resume_uploaded': resume_content is not None,
                'resume_analysis': resume_analysis,
                'resume_content_preview': resume_content[:200] + "..." if resume_content and len(resume_content) > 200 else resume_content,
                'university_recommendations': recommendations
            }
        }
        return jsonify(response), 200

    except Exception as e:
        logger.exception("Error processing request")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/receive-extracted-data', methods=['POST'])
def n8n_callback():
    """
    Receive processed data from n8n.
    Accepts JSON like:
      { "sessionId": "...", "parsedData": {...} }
    or
      { "sessionId": "...", "processedData": {...} }
    or various agent outputs (string or JSON).
    """
    try:
        data = request.get_json(force=True, silent=True)
        # fallback: if body was sent as raw string
        if data is None:
            raw = request.data.decode('utf-8') or ''
            try:
                import json
                data = json.loads(raw) if raw else {}
            except Exception:
                data = {"raw": raw}

        session_id = data.get('sessionId') or data.get('sessionId'.lower())
        # try common fields where processed content may be
        processed_data = data.get('parsedData') or data.get('processedData') or data.get('output') or data.get('data') or data.get('result') or data.get('body') or data.get('raw')

        if not session_id:
            logger.warning("n8n callback received without sessionId: %s", data)
            return jsonify({"success": False, "error": "no sessionId provided"}), 400

        # store result (overwrite previous)
        processed_results[session_id] = {
            "status": "completed",
            "data": processed_data,
            "received_at": datetime.utcnow().isoformat()
        }

        logger.info("Stored n8n result for session %s", session_id)
        return jsonify({"success": True}), 200

    except Exception as e:
        logger.exception("n8n callback error")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/get-processed-data/<session_id>', methods=['GET'])
def get_processed_data(session_id):
    item = processed_results.get(session_id)
    if not item:
        return jsonify({"success": False, "status": "processing"}), 202
    return jsonify({"success": True, "status": item.get("status"), "data": item.get("data"), "preview": item.get("preview")}), 200

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'University Finder API is running'})

if __name__ == '__main__':
    logger.info("Starting backend on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
