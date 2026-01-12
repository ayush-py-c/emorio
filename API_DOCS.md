# Emotion Detection API

Real-time emotion detection API server using TensorFlow and Flask.

## Endpoints

### 1. Health Check
- **URL:** `/health`
- **Method:** `GET`
- **Response:**
```json
{
  "status": "ok"
}
```

### 2. Detect Emotion
- **URL:** `/detect-emotion`
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`
- **Parameters:**
  - `image` (required): Image file containing a face

- **Response (Success):**
```json
{
  "emotion": "Happy",
  "confidence": 0.95,
  "faces_detected": 1,
  "all_emotions": [
    {
      "emotion": "Happy",
      "confidence": 0.95
    }
  ]
}
```

- **Response (No Face):**
```json
{
  "emotion": "No face detected",
  "confidence": 0
}
```

## Emotions Detected
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

## Testing with cURL

```bash
# Health check
curl http://127.0.0.1:5000/health

# Detect emotion from an image
curl -X POST -F "image=@/path/to/image.jpg" http://127.0.0.1:5000/detect-emotion
```

## Testing with Python

```python
import requests

# Health check
response = requests.get('http://127.0.0.1:5000/health')
print(response.json())

# Detect emotion
with open('image.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post('http://127.0.0.1:5000/detect-emotion', files=files)
    print(response.json())
```

## Running the Server

```bash
cd emotion_app
set TF_ENABLE_ONEDNN_OPTS=0
python app.py
```

Server will start at `http://127.0.0.1:5000`
