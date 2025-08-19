from flask import Flask, request, jsonify
from flask_cors import CORS
from docling.document_converter import DocumentConverter
import os
import tempfile
from werkzeug.utils import secure_filename
import logging
import requests
import uuid   # ✅ added for session id generation

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_resume_content(file_path, session_id=None):
    """Extract content from resume using docling"""
    try:
        converter = DocumentConverter()
        result = converter.convert(file_path)
        markdown_content = result.document.export_to_markdown()

        logger.info(f"Successfully extracted content from {file_path}")
        logger.info("=" * 50)
        logger.info("DOCLING EXTRACTED CONTENT:")
        logger.info("=" * 50)
        logger.info(markdown_content[:500] + "..." if len(markdown_content) > 500 else markdown_content)
        logger.info("=" * 50)

        # Save content preview to file
        output_file = f"extracted_content_{os.path.basename(file_path)}.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        # ✅ Send to n8n with session id
        send_to_n8n(markdown_content, session_id)

        return {
            'success': True,
            'content': markdown_content,
            'error': None
        }
    except Exception as e:
        logger.error(f"Error extracting content from {file_path}: {str(e)}")
        return {
            'success': False,
            'content': None,
            'error': str(e)
        }


def send_to_n8n(parsed_content, session_id=None):
    """Send parsed content to n8n webhook"""
    n8n_webhook_url = "http://localhost:5678/webhook-test/a6fdd077-5e86-4d4f-bebf-7178962fb86e"

    # ✅ Generate random session if none provided
    if not session_id:
        session_id = str(uuid.uuid4())

    try:
        response = requests.post(
            n8n_webhook_url,
            json={
                "sessionId": session_id,
                "content": parsed_content
            }
        )
        response.raise_for_status()
        logger.info(f"Successfully sent data to n8n with sessionId={session_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to send data to n8n: {str(e)}")
        return False


def analyze_resume_content(content, preferences):
    # (your analysis function unchanged)
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

    content_lower = content.lower()
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

        # ✅ Get sessionId from frontend if available, else None
        session_id = request.form.get("sessionId")

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
                temp_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(temp_path)
                logger.info(f"Saved resume file: {temp_path}")

                # ✅ Pass session_id so n8n memory works
                extraction_result = extract_resume_content(temp_path, session_id)

                if extraction_result['success']:
                    resume_content = extraction_result['content']
                    resume_analysis = analyze_resume_content(resume_content, preferences)

                try:
                    os.remove(temp_path)
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file: {e}")

        recommendations = generate_university_recommendations(preferences, resume_analysis)

        response = {
            'success': True,
            'message': 'Preferences submitted successfully!',
            'data': {
                'preferences': preferences,
                'resume_uploaded': resume_content is not None,
                'resume_analysis': resume_analysis,
                'resume_content_preview': resume_content[:200] + "..." if resume_content and len(resume_content) > 200 else resume_content,
                'resume_content_length': len(resume_content) if resume_content else 0,
                'university_recommendations': recommendations
            }
        }
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


def generate_university_recommendations(preferences, resume_analysis=None):
    # (unchanged mock data generator)
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


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'University Finder API is running'})


if __name__ == '__main__':
    print("Starting University Finder Backend...")
    app.run(debug=True, host='0.0.0.0', port=5000)
