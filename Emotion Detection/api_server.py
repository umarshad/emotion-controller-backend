"""
Flask API Server for Emotion Detection
Wraps existing emotion detection logic without modifying detection.py or src/emotions.py
"""
import os
import time
import base64
import json
import uuid
import logging
import sys
import cv2
import numpy as np
from functools import wraps
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import model_from_json, Sequential
from tensorflow.keras.utils import get_custom_objects
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Configure structured logging
log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Register the Sequential class with TensorFlow Keras
get_custom_objects().update({'Sequential': Sequential})

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter app

# Global variables for model and cascade (loaded once at startup)
_model = None
_face_cascade = None

# Emotion labels mapping (from detection.py)
EMOTION_LABELS = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# API Statistics
_api_stats = {
    'total_requests': 0,
    'successful_requests': 0,
    'failed_requests': 0,
    'total_processing_time_ms': 0,
    'requests_by_emotion': {emotion: 0 for emotion in EMOTION_LABELS.values()},
    'requests_by_error': {}
}

# Rate limiting (simple in-memory implementation)
_rate_limits = {}  # {api_key: {'count': int, 'reset_time': timestamp}}
RATE_LIMIT_WINDOW = 60  # 60 seconds
RATE_LIMIT_MAX_REQUESTS = 100  # 100 requests per window

# Load API keys
def load_api_keys():
    """Load API keys from environment variable or file"""
    # Try environment variable first
    env_keys = os.environ.get('EMOTION_API_KEYS')
    if env_keys:
        return [key.strip() for key in env_keys.split(',')]
    
    # Try loading from file
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        keys_file = os.path.join(script_dir, 'api_keys.json')
        if os.path.exists(keys_file):
            with open(keys_file, 'r') as f:
                data = json.load(f)
                return data.get('api_keys', [])
    except Exception as e:
        print(f"Warning: Could not load API keys from file: {e}")
    
    # Default test keys
    return [
        'emotion-api-test-1234567890abcdef',
        'emotion-api-postman-9876543210fedcba',
        'emotion-api-flutter-mobile-abcdef1234567890'
    ]

_valid_api_keys = load_api_keys()

def require_api_key(f):
    """Decorator to require API key authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check for API key in headers
        api_key = None
        
        # Try X-API-Key header
        if 'X-API-Key' in request.headers:
            api_key = request.headers.get('X-API-Key')
        # Try Authorization: Bearer <key>
        elif 'Authorization' in request.headers:
            auth_header = request.headers.get('Authorization', '')
            if auth_header.startswith('Bearer '):
                api_key = auth_header[7:]
        
        if not api_key:
            return jsonify({
                "success": False,
                "error": {
                    "code": "UNAUTHORIZED",
                    "message": "API key required",
                    "details": "Please provide an API key in the 'X-API-Key' header or 'Authorization: Bearer <key>' header"
                },
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
            }), 401
        
        if api_key not in _valid_api_keys:
            return jsonify({
                "success": False,
                "error": {
                    "code": "UNAUTHORIZED",
                    "message": "Invalid API key",
                    "details": "The provided API key is not valid"
                },
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
            }), 401
        
        # Check rate limit
        current_time = time.time()
        if api_key in _rate_limits:
            rate_data = _rate_limits[api_key]
            if current_time > rate_data['reset_time']:
                # Reset window
                rate_data['count'] = 1
                rate_data['reset_time'] = current_time + RATE_LIMIT_WINDOW
            else:
                rate_data['count'] += 1
                if rate_data['count'] > RATE_LIMIT_MAX_REQUESTS:
                    logger.warning(f"Rate limit exceeded for API key: {api_key[:8]}...")
                    return jsonify({
                        "success": False,
                        "error": {
                            "code": "TOO_MANY_REQUESTS",
                            "message": "Rate limit exceeded",
                            "details": f"Maximum {RATE_LIMIT_MAX_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds"
                        },
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
                    }), 429
        else:
            # Initialize rate limit for this key
            _rate_limits[api_key] = {
                'count': 1,
                'reset_time': current_time + RATE_LIMIT_WINDOW
            }
        
        return f(*args, **kwargs)
    return decorated_function

def create_error_response(error_code, message, details, status_code=400, request_id=None):
    """Create standardized error response"""
    if request_id is None:
        request_id = str(uuid.uuid4())
    
    return jsonify({
        "success": False,
        "error": {
            "code": error_code,
            "message": message,
            "details": details
        },
        "request_id": request_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    }), status_code


def load_model():
    """
    Load the emotion detection model from JSON and weights
    Extracted from detection.py without modification
    """
    global _model
    if _model is not None:
        return _model
    
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(script_dir, "facialemotionmodel.json")
        weights_path = os.path.join(script_dir, "facialemotionmodel.h5")
        
        # Open and read the model JSON file
        with open(json_path, "r") as json_file:
            model_json = json_file.read()
        
        # Load the model from JSON
        model = model_from_json(model_json)
        
        # Load weights into the model
        model.load_weights(weights_path)
        _model = model
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        return None


def load_face_cascade():
    """
    Load the face cascade classifier
    Extracted from detection.py without modification
    """
    global _face_cascade
    if _face_cascade is not None:
        return _face_cascade
    
    try:
        haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        cascade = cv2.CascadeClassifier(haar_file)
        _face_cascade = cascade
        logger.info("Face cascade loaded successfully")
        return cascade
    except Exception as e:
        logger.error(f"Error loading face cascade: {e}", exc_info=True)
        return None


def extract_features(image):
    """
    Extract features from the image for model input
    Extracted from detection.py without modification
    """
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0


def detect_emotion_in_image(image_array, model, face_cascade):
    """
    Detect emotion in a single image frame
    Modified from detect_emotions_in_frame to return structured data instead of drawing on frame
    
    Args:
        image_array: numpy array of the image (BGR format)
        model: loaded emotion detection model
        face_cascade: loaded face cascade classifier
    
    Returns:
        dict with emotion detection results or None if no face detected
    """
    if model is None or face_cascade is None:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces) == 0:
        return None
    
    # Process first face only (for real-time performance)
    (x, y, w, h) = faces[0]
    
    # Extract face region
    face_image = gray[y:y + h, x:x + w]
    
    # Resize to 48x48 (model input size)
    face_image = cv2.resize(face_image, (48, 48))
    
    # Extract features
    img_features = extract_features(face_image)
    
    # Predict emotion
    predictions = model.predict(img_features, verbose=0)
    
    # Get all emotion probabilities
    emotion_probs = {}
    for label_id, emotion_name in EMOTION_LABELS.items():
        emotion_probs[emotion_name] = float(predictions[0][label_id])
    
    # Get primary emotion (highest confidence)
    primary_emotion_idx = int(np.argmax(predictions))
    primary_emotion = EMOTION_LABELS[primary_emotion_idx]
    primary_confidence = float(predictions[0][primary_emotion_idx])
    
    # Get top 3 emotions
    top_indices = np.argsort(predictions[0])[-3:][::-1]
    top_emotions = [
        {
            "emotion": EMOTION_LABELS[int(idx)],
            "confidence": float(predictions[0][int(idx)])
        }
        for idx in top_indices
    ]
    
    return {
        "primary_emotion": primary_emotion,
        "primary_confidence": primary_confidence,
        "all_emotions": emotion_probs,
        "top_emotions": top_emotions,
        "face_detected": True,
        "face_bounds": {
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h)
        },
        "num_faces": len(faces)
    }


def decode_base64_image(base64_string, image_format='jpeg'):
    """
    Decode base64 encoded image to numpy array
    
    Args:
        base64_string: base64 encoded image string
        image_format: format of the image ('jpeg' or 'png')
    
    Returns:
        numpy array of the image (BGR format for OpenCV)
    """
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(base64_string)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        if image_format.lower() == 'png':
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:  # default to jpeg
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        return image
    except Exception as e:
        raise ValueError(f"Error decoding image: {str(e)}")


def validate_request(data, request_id=None):
    """Validate request data and return error response if invalid"""
    # Check content type
    if request.content_type != 'application/json':
        return create_error_response(
            "UNSUPPORTED_MEDIA_TYPE",
            "Invalid content type",
            "Content-Type must be 'application/json'",
            415,
            request_id
        )
    
    # Check if data exists
    if not data:
        return create_error_response(
            "BAD_REQUEST",
            "Invalid request format",
            "Request body must be valid JSON",
            400,
            request_id
        )
    
    # Check if image field exists
    if 'image' not in data:
        return create_error_response(
            "BAD_REQUEST",
            "Missing required field",
            "Request must include 'image' field with base64-encoded image",
            400,
            request_id
        )
    
    # Validate image size (max 10MB when base64 decoded)
    image_str = data.get('image', '')
    if len(image_str) > 15 * 1024 * 1024:  # ~10MB base64 encoded
        return create_error_response(
            "PAYLOAD_TOO_LARGE",
            "Image too large",
            "Image size must be less than 10MB",
            413,
            request_id
        )
    
    # Validate image format
    image_format = data.get('format', 'jpeg').lower()
    if image_format not in ['jpeg', 'jpg', 'png']:
        return create_error_response(
            "UNSUPPORTED_MEDIA_TYPE",
            "Unsupported image format",
            f"Image format must be 'jpeg' or 'png', got '{image_format}'",
            415,
            request_id
        )
    
    return None  # Validation passed


@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """Health check endpoint (public, no auth required)"""
    logger.debug("Health check endpoint accessed")
    """Health check endpoint (public, no auth required)"""
    return jsonify({
        "status": "healthy",
        "model_loaded": _model is not None,
        "cascade_loaded": _face_cascade is not None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    })


@app.route('/api/v1/info', methods=['GET'])
def api_info():
    """API information endpoint (public, no auth required)"""
    logger.info("API info endpoint accessed")
    return jsonify({
        "api_version": "1.0.0",
        "name": "Emotion Detection API",
        "description": "Real-time emotion detection from facial images",
        "supported_emotions": list(EMOTION_LABELS.values()),
        "model_status": {
            "loaded": _model is not None,
            "cascade_loaded": _face_cascade is not None
        },
        "endpoints": {
            "health": "/api/v1/health",
            "info": "/api/v1/info",
            "model_info": "/api/v1/model-info",
            "detect_emotion": "/api/v1/detect-emotion",
            "stats": "/api/v1/stats"
        },
        "authentication": {
            "required": True,
            "methods": ["X-API-Key header", "Authorization: Bearer <key>"]
        },
        "rate_limit": {
            "max_requests": RATE_LIMIT_MAX_REQUESTS,
            "window_seconds": RATE_LIMIT_WINDOW
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    })


@app.route('/api/v1/model-info', methods=['GET'])
def model_info():
    """Model information endpoint (public, no auth required)"""
    logger.info("Model info endpoint accessed")
    
    if _model is None:
        return jsonify({
            "success": False,
            "error": {
                "code": "MODEL_NOT_LOADED",
                "message": "Model is not loaded",
                "details": "The emotion detection model has not been loaded"
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        }), 503
    
    try:
        # Get model input shape
        input_shape = _model.input_shape if hasattr(_model, 'input_shape') else None
        output_shape = _model.output_shape if hasattr(_model, 'output_shape') else None
        
        return jsonify({
            "success": True,
            "model": {
                "loaded": True,
                "input_shape": input_shape,
                "output_shape": output_shape,
                "emotion_labels": list(EMOTION_LABELS.values()),
                "num_emotions": len(EMOTION_LABELS),
                "cascade_loaded": _face_cascade is not None
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        })
    except Exception as e:
        logger.error(f"Error getting model info: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "Error retrieving model information",
                "details": str(e)
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        }), 500


@app.route('/api/v1/stats', methods=['GET'])
@require_api_key
def api_stats():
    """API statistics endpoint (requires API key)"""
    avg_processing_time = 0
    if _api_stats['total_requests'] > 0:
        avg_processing_time = _api_stats['total_processing_time_ms'] / _api_stats['total_requests']
    
    success_rate = 0
    if _api_stats['total_requests'] > 0:
        success_rate = (_api_stats['successful_requests'] / _api_stats['total_requests']) * 100
    
    return jsonify({
        "total_requests": _api_stats['total_requests'],
        "successful_requests": _api_stats['successful_requests'],
        "failed_requests": _api_stats['failed_requests'],
        "success_rate_percent": round(success_rate, 2),
        "average_processing_time_ms": round(avg_processing_time, 2),
        "requests_by_emotion": _api_stats['requests_by_emotion'],
        "requests_by_error": _api_stats['requests_by_error'],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    })


@app.route('/api/v1/detect-emotion', methods=['POST'])
@require_api_key
def detect_emotion():
    """
    Main endpoint for emotion detection
    
    Expected request body:
    {
        "image": "base64_encoded_image_string",
        "format": "jpeg|png" (optional, defaults to jpeg),
        "timestamp": "ISO8601_timestamp" (optional)
    }
    
    Returns:
    {
        "success": true/false,
        "emotion": {...} or null,
        "error": {...} or null,
        "timestamp": "ISO8601_timestamp",
        "processing_time_ms": float
    }
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    logger.info(f"[{request_id}] Emotion detection request received")
    
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'image' not in data:
            logger.warning(f"[{request_id}] Invalid request: missing image field")
            return jsonify({
                "success": False,
                "error": {
                    "code": "INVALID_REQUEST",
                    "message": "Missing 'image' field in request",
                    "details": "Please provide a base64 encoded image in the 'image' field"
                },
                "request_id": request_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
                "processing_time_ms": (time.time() - start_time) * 1000
            }), 400
        
        # Get image format (default to jpeg)
        image_format = data.get('format', 'jpeg')
        
        # Decode image
        try:
            image_array = decode_base64_image(data['image'], image_format)
        except ValueError as e:
            _api_stats['failed_requests'] += 1
            _api_stats['requests_by_error']['INVALID_IMAGE'] = _api_stats['requests_by_error'].get('INVALID_IMAGE', 0) + 1
            return create_error_response(
                "INVALID_IMAGE",
                "Failed to decode image",
                str(e),
                400,
                request_id
            )
        
        # Ensure model and cascade are loaded
        model = load_model()
        face_cascade = load_face_cascade()
        
        if model is None:
            _api_stats['failed_requests'] += 1
            _api_stats['requests_by_error']['MODEL_ERROR'] = _api_stats['requests_by_error'].get('MODEL_ERROR', 0) + 1
            return create_error_response(
                "SERVICE_UNAVAILABLE",
                "Emotion detection model not loaded",
                "The model failed to load. Please check server logs.",
                503,
                request_id
            )
        
        if face_cascade is None:
            _api_stats['failed_requests'] += 1
            _api_stats['requests_by_error']['CASCADE_ERROR'] = _api_stats['requests_by_error'].get('CASCADE_ERROR', 0) + 1
            return create_error_response(
                "SERVICE_UNAVAILABLE",
                "Face detection cascade not loaded",
                "The face detection cascade failed to load. Please check server logs.",
                503,
                request_id
            )
        
        # Detect emotion with timeout handling
        try:
            detection_result = detect_emotion_in_image(image_array, model, face_cascade)
        except Exception as e:
            _api_stats['failed_requests'] += 1
            _api_stats['requests_by_error']['INTERNAL_ERROR'] = _api_stats['requests_by_error'].get('INTERNAL_ERROR', 0) + 1
            print(f"Error in emotion detection: {e}")
            return create_error_response(
                "INTERNAL_ERROR",
                "Error during emotion detection",
                "An error occurred while processing the image. Please try again.",
                500,
                request_id
            )
        
        processing_time = (time.time() - start_time) * 1000
        _api_stats['total_processing_time_ms'] += processing_time
        
        if detection_result is None:
            _api_stats['failed_requests'] += 1
            _api_stats['requests_by_error']['NO_FACE_DETECTED'] = _api_stats['requests_by_error'].get('NO_FACE_DETECTED', 0) + 1
            logger.info(f"[{request_id}] No face detected in image")
            return jsonify({
                "success": False,
                "emotion": None,
                "error": {
                    "code": "NO_FACE_DETECTED",
                    "message": "No face detected in the image",
                    "details": "Please ensure your face is clearly visible in the frame"
                },
                "request_id": request_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
                "processing_time_ms": round(processing_time, 2)
            }), 200  # Return 200 but with success: false
        
        # Update statistics
        _api_stats['successful_requests'] += 1
        emotion_type = detection_result["primary_emotion"]
        _api_stats['requests_by_emotion'][emotion_type] = _api_stats['requests_by_emotion'].get(emotion_type, 0) + 1
        
        # Format response
        response = {
            "success": True,
            "emotion": {
                "type": detection_result["primary_emotion"],
                "confidence": detection_result["primary_confidence"],
                "intensity": int(detection_result["primary_confidence"] * 100),
                "detected_emotions": detection_result["top_emotions"],
                "face_detected": True,
                "face_bounds": detection_result["face_bounds"],
                "num_faces": detection_result["num_faces"]
            },
            "request_id": request_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "processing_time_ms": round(processing_time, 2)
        }
        
        logger.info(f"[{request_id}] Emotion detected: {emotion_type} (confidence: {detection_result['primary_confidence']:.2f}, processing: {processing_time:.2f}ms)")
        return jsonify(response), 200
        
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        _api_stats['failed_requests'] += 1
        _api_stats['requests_by_error']['INTERNAL_ERROR'] = _api_stats['requests_by_error'].get('INTERNAL_ERROR', 0) + 1
        logger.error(f"[{request_id}] Error in detect_emotion endpoint: {e}", exc_info=True)
        return create_error_response(
            "INTERNAL_ERROR",
            "Internal server error",
            "An unexpected error occurred. Please try again later.",
            500,
            request_id
        )


if __name__ == '__main__':
    # Load environment variables for production configuration
    api_host = os.environ.get('API_HOST', '0.0.0.0')
    api_port = int(os.environ.get('API_PORT', 5000))
    log_level = os.environ.get('LOG_LEVEL', 'INFO')
    max_request_size = int(os.environ.get('MAX_REQUEST_SIZE', 10 * 1024 * 1024))  # 10MB default
    cors_origins = os.environ.get('CORS_ORIGINS', '*')
    
    # Update CORS configuration
    if cors_origins != '*':
        origins = [origin.strip() for origin in cors_origins.split(',')]
        CORS(app, origins=origins)
    else:
        CORS(app)  # Allow all origins (development)
    
    # Configure Flask app
    app.config['MAX_CONTENT_LENGTH'] = max_request_size
    
    # Load model and cascade at startup
    logger.info("=" * 60)
    logger.info("Emotion Detection API Server")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  Host: {api_host}")
    logger.info(f"  Port: {api_port}")
    logger.info(f"  Log Level: {log_level}")
    logger.info(f"  Max Request Size: {max_request_size / (1024*1024):.1f}MB")
    logger.info(f"  CORS Origins: {cors_origins}")
    logger.info("=" * 60)
    logger.info("Loading emotion detection model...")
    model = load_model()
    if model is None:
        logger.error("Failed to load emotion detection model!")
        logger.error("Make sure facialemotionmodel.json and facialemotionmodel.h5 are in the same directory.")
        sys.exit(1)
    logger.info("Model loaded successfully")
    
    logger.info("Loading face cascade...")
    cascade = load_face_cascade()
    if cascade is None:
        logger.error("Failed to load face cascade!")
        sys.exit(1)
    logger.info("Face cascade loaded successfully")
    
    # Reload API keys from environment if set
    env_keys = os.environ.get('EMOTION_API_KEYS')
    if env_keys:
        _valid_api_keys.clear()
        _valid_api_keys.extend([key.strip() for key in env_keys.split(',')])
        logger.info(f"Loaded {len(_valid_api_keys)} API key(s) from environment")
    else:
        logger.warning("Using default API keys (set EMOTION_API_KEYS for production)")
    
    logger.info("-" * 60)
    logger.info("API Endpoints:")
    logger.info(f"  Health Check: http://{api_host}:{api_port}/api/v1/health")
    logger.info(f"  API Info: http://{api_host}:{api_port}/api/v1/info")
    logger.info(f"  Model Info: http://{api_host}:{api_port}/api/v1/model-info")
    logger.info(f"  Detect Emotion: http://{api_host}:{api_port}/api/v1/detect-emotion")
    logger.info(f"  Statistics: http://{api_host}:{api_port}/api/v1/stats")
    logger.info("-" * 60)
    
    # Only run Flask dev server if not using Gunicorn (production)
    if __name__ == '__main__':
        logger.info("Starting Flask development server...")
        logger.info("Server will be accessible at:")
        logger.info(f"  - Local: http://localhost:{api_port}")
        logger.info(f"  - Network: http://{api_host}:{api_port}")
        
        # Get local IP address for physical device connection
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            logger.info(f"  - Physical Devices: http://{local_ip}:{api_port}")
        except Exception as e:
            logger.warning(f"Could not determine local IP: {e}")
            logger.info("  - Physical Devices: Use your computer's IP address")
        
        logger.info("=" * 60)
        
        # Run Flask app (only in development)
        try:
            app.run(host=api_host, port=api_port, debug=(log_level == 'DEBUG'))
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            logger.error(f"Make sure port {api_port} is not already in use.")
            sys.exit(1)

