from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from io import BytesIO
from PIL import Image
import json
import base64
import threading

app = Flask(__name__)
CORS(app)

# Load model and face detector
model = load_model("emotion_model.h5", compile=False)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# Global camera stream
camera = cv2.VideoCapture(0)
camera_lock = threading.Lock()
current_frame = None
is_streaming = False

@app.route('/detect-emotion', methods=['POST'])
def detect_emotion():
    try:
        # Check if image is in form data
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read image from file
        image = Image.open(file.stream)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return jsonify({'emotion': 'No face detected', 'confidence': 0}), 200
        
        emotions_detected = []
        
        for (x,y,w,h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (64,64))
            face = face/255.0
            face = np.reshape(face, (1,64,64,1))
            
            prediction = model.predict(face, verbose=0)
            emotion_idx = np.argmax(prediction)
            emotion = emotion_labels[emotion_idx]
            confidence = float(prediction[0][emotion_idx])
            
            emotions_detected.append({
                'emotion': emotion,
                'confidence': confidence
            })
        
        # Return dominant emotion
        dominant_emotion = emotions_detected[0]
        
        return jsonify({
            'emotion': dominant_emotion['emotion'],
            'confidence': dominant_emotion['confidence'],
            'faces_detected': len(faces),
            'all_emotions': emotions_detected
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

def generate_frames():
    global current_frame, is_streaming
    is_streaming = True
    
    while is_streaming:
        success, frame = camera.read()
        if not success:
            continue
        
        # Flip for selfie view
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (64, 64))
            face = face / 255.0
            face = np.reshape(face, (1, 64, 64, 1))
            
            prediction = model.predict(face, verbose=0)
            emotion_idx = np.argmax(prediction)
            emotion = emotion_labels[emotion_idx]
            confidence = prediction[0][emotion_idx]
            
            # Draw rectangle and text on frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{emotion} ({confidence*100:.1f}%)"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

if __name__ == "__main__":
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

