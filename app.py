"""
Flask Web Application for Waste Classification
Allows users to upload images and get classification results
"""
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import os
import json
from datetime import datetime
from predict import WasteClassifier
import config

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(config.BASE_DIR, 'static', 'uploads')
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}

# Initialize classifier (will be loaded on first request)
classifier = None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_classifier():
    """Get or initialize the classifier"""
    global classifier
    if classifier is None:
        try:
            classifier = WasteClassifier()
        except Exception as e:
            print(f"Error loading classifier: {e}")
            return None
    return classifier


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check if file type is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed types: ' + ', '.join(ALLOWED_EXTENSIONS)}), 400
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get classifier
        clf = get_classifier()
        if clf is None:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
        
        # Make prediction
        result = clf.predict(filepath, return_probabilities=True)
        
        # Add file info to result
        result['filename'] = filename
        result['filepath'] = f'/static/uploads/{filename}'
        result['timestamp'] = timestamp
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    clf = get_classifier()
    
    status = {
        'status': 'ok' if clf is not None else 'error',
        'model_loaded': clf is not None,
        'timestamp': datetime.now().isoformat()
    }
    
    if clf is not None and hasattr(clf, 'metadata'):
        status['model_info'] = clf.metadata
    
    return jsonify(status)


@app.route('/api/classes')
def get_classes():
    """Get available waste classes"""
    clf = get_classifier()
    if clf is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    classes = clf.label_encoder.classes_.tolist()
    return jsonify({'classes': classes})


if __name__ == '__main__':
    print("=" * 60)
    print("WASTE CLASSIFICATION WEB APPLICATION")
    print("=" * 60)
    print("\nStarting server...")
    print("Make sure you have trained the model first!")
    print("\nAccess the application at: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
