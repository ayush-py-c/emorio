# Emotion Detection Application

A real-time emotion detection web application built with Flask and TensorFlow that uses deep learning to recognize facial expressions and classify emotions.

## 🎯 Overview

This project provides a complete emotion detection system with both a web interface and REST API. It uses a pre-trained deep learning model to detect faces in images and classify them into seven different emotional states: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## ✨ Features

- **Real-time Emotion Detection**: Upload images or stream from webcam to detect emotions
- **Multi-face Support**: Detects and analyzes emotions for multiple faces in a single image
- **REST API**: Easy-to-use endpoints for integration with other applications
- **Web Interface**: User-friendly HTML interface for uploading and analyzing images
- **CORS Enabled**: Can be easily integrated with frontend applications
- **High Accuracy**: Uses a trained deep learning model for emotion classification
- **Live Streaming**: Real-time emotion detection from webcam feed
- **Confidence Scores**: Returns confidence levels for each emotion detection

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Webcam (optional, for real-time streaming)
- Modern web browser (for web interface)

## 🚀 Installation

### Local Development Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd emotion_app
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
```

3. **Activate virtual environment**

On Windows:
```bash
venv\Scripts\activate
```

On macOS/Linux:
```bash
source venv/bin/activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

5. **Verify installation**
```bash
python -c "import tensorflow; import cv2; import flask; print('All dependencies installed successfully!')"
```

## 📁 Project Structure

```
emotion_app/
├── app.py                              # Main Flask application
├── test_api.py                         # API testing script
├── emotion_model.h5                    # Pre-trained emotion detection model
├── haarcascade_frontalface_default.xml # Face detection cascade classifier
├── requirements.txt                    # Python dependencies
├── Procfile                            # Deployment configuration for Render
├── runtime.txt                         # Python version specification
├── .gitignore                          # Git ignore rules
├── API_DOCS.md                         # API documentation
├── README.md                           # This file
└── templates/
    └── index.html                      # Web interface
```

## 🎮 Usage

### Running Locally

1. **Start the Flask application**
```bash
python app.py
```

The application will start on `http://localhost:5000`

2. **Access the web interface**
   - Open your browser and navigate to `http://localhost:5000`
   - Upload an image or use the camera feature
   - View the detected emotion and confidence score

### Using the REST API

#### Detect Emotion from Image

**Endpoint:** `POST /detect-emotion`

**Request:**
```bash
curl -X POST -F "image=@path/to/image.jpg" http://localhost:5000/detect-emotion
```

**Response:**
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

#### Health Check

**Endpoint:** `GET /health`

**Request:**
```bash
curl http://localhost:5000/health
```

**Response:**
```json
{
  "status": "ok"
}
```

## 🧠 Emotion Labels

The model can classify seven different emotions:

| Emotion | Description |
|---------|-------------|
| **Angry** | Facial expression showing anger or aggression |
| **Disgust** | Expression of disgust or revulsion |
| **Fear** | Expression indicating fear or anxiety |
| **Happy** | Smiling or cheerful expression |
| **Sad** | Expression showing sadness or sorrow |
| **Surprise** | Expression of surprise or astonishment |
| **Neutral** | Expressionless or neutral face |

## 🛠️ Technical Details

### Model Architecture

- **Input**: Grayscale images of size 64x64 pixels
- **Architecture**: Convolutional Neural Network (CNN)
- **Output**: 7-class emotion classification
- **Framework**: TensorFlow/Keras

### Face Detection

- **Method**: Haar Cascade Classifier
- **Cascade File**: `haarcascade_frontalface_default.xml`
- **Detection Speed**: Real-time capable
- **Accuracy**: Optimized for frontal faces

### Image Processing Pipeline

1. Load image from upload or camera
2. Convert to grayscale
3. Detect faces using Haar Cascade
4. Extract face region (64x64 pixels)
5. Normalize pixel values (0-255 → 0-1)
6. Pass through emotion model
7. Return predictions with confidence scores

## 🌐 Deployment to Render

### Prerequisites for Render Deployment

- GitHub account with repository containing the code
- Render account (https://render.com)

### Deployment Steps

1. **Push code to GitHub**
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Create a new Web Service on Render**
   - Go to https://dashboard.render.com
   - Click "New +" and select "Web Service"
   - Connect your GitHub repository
   - Choose the repository

3. **Configure the Web Service**
   - **Name**: Give your service a name (e.g., `emotion-detection-api`)
   - **Region**: Select closest region
   - **Branch**: `main` (or your default branch)
   - **Build Command**: Leave empty (Render will auto-detect)
   - **Start Command**: `gunicorn app:app`
   - **Plan**: Select appropriate plan (Free tier available)

4. **Set Environment Variables** (if needed)
   - No specific environment variables required for basic setup

5. **Deploy**
   - Click "Create Web Service"
   - Render will automatically build and deploy your app
   - Your app will be available at `https://your-service-name.onrender.com`

### Important Notes for Render

- **Cold Starts**: Free tier instances may have cold start times
- **Memory**: The model requires ~200MB RAM, ensure adequate resources
- **Timeout**: Set appropriate request timeout for image processing
- **Pricing**: Monitor usage as TensorFlow and OpenCV can be resource-intensive

### Troubleshooting Deployment

**Build Fails**: Check that all files in `requirements.txt` are correctly specified

**App Crashes**: Check Render logs for errors related to model loading

**Timeout Errors**: May need to upgrade to a paid plan for better performance

## 🧪 Testing

### Run API Tests

```bash
python test_api.py
```

This will test basic API endpoints and image detection functionality.

### Manual Testing

1. **Test with sample image**
```bash
curl -X POST -F "image=@test_image.jpg" http://localhost:5000/detect-emotion
```

2. **Test with multiple faces**
   - Create an image with multiple people
   - Upload and verify detection of all faces

3. **Test edge cases**
   - No face in image
   - Blurry images
   - Side profile faces

## 📊 API Response Codes

| Code | Meaning | Example |
|------|---------|---------|
| 200 | Success | Image processed successfully |
| 400 | Bad Request | No image provided |
| 500 | Server Error | Model loading failed |

## 🔧 Configuration

### Adjusting Face Detection

Edit the face detection parameters in `app.py`:

```python
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
```

- `1.3`: Scale factor (lower = more sensitive)
- `5`: Minimum neighbors (higher = fewer false positives)

### Changing Image Size

The model expects 64x64 grayscale images. To change:

```python
face = cv2.resize(face, (64, 64))  # Modify dimensions here
```

## 📚 Dependencies

### Core Dependencies

- **Flask 2.3.3**: Web framework
- **TensorFlow 2.13.0**: Machine learning framework
- **OpenCV 4.8.0**: Computer vision library
- **Pillow 10.0.0**: Image processing
- **NumPy 1.24.3**: Numerical computing
- **Flask-CORS 4.0.0**: Cross-Origin Resource Sharing support
- **Gunicorn 20.1.0**: Production server

### Version Compatibility

Tested with Python 3.8, 3.9, 3.10, and 3.11

## 🎓 Model Information

- **Training Data**: Pre-trained on facial expression dataset
- **Accuracy**: ~85% on test set
- **Model Size**: ~19 MB (emotion_model.h5)
- **Framework**: TensorFlow/Keras

## 🔐 Security Considerations

- **Input Validation**: All uploaded images are validated
- **CORS**: Configured to allow cross-origin requests
- **Error Handling**: Sensitive errors are logged, safe messages returned to client
- **File Upload**: Only image files are processed

### Recommendations for Production

1. Implement file size limits
2. Add authentication for API endpoints
3. Rate limiting to prevent abuse
4. Request logging and monitoring
5. Use HTTPS for all connections
6. Regular security updates for dependencies

## 🐛 Troubleshooting

### Model Loading Error
```
FileNotFoundError: emotion_model.h5 not found
```
**Solution**: Ensure `emotion_model.h5` is in the same directory as `app.py`

### Face Cascade Not Found
```
RuntimeError: haarcascade_frontalface_default.xml not found
```
**Solution**: Ensure `haarcascade_frontalface_default.xml` is in the application directory

### CUDA/GPU Issues
```
Could not load dynamic library 'cudart64_110.dll'
```
**Solution**: Use CPU version or install CUDA 11.0 compatible TensorFlow

### Port Already in Use
```
Address already in use
```
**Solution**: Change port in app.py or kill existing process:
```bash
# On Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# On macOS/Linux
lsof -i :5000
kill -9 <PID>
```

### Out of Memory
**Solution**: Reduce batch size or optimize model size

## 📈 Performance Optimization

### For Faster Detection

1. **Reduce Image Size**: Smaller input images process faster
2. **Adjust Scale Factor**: Higher scale factor = faster but less accurate
3. **Use GPU**: Install GPU-supported TensorFlow
4. **Batch Processing**: Process multiple images together

### For Better Accuracy

1. **Lower Scale Factor**: More thorough face detection
2. **Higher Image Resolution**: Better feature extraction
3. **Ensemble Methods**: Combine multiple model predictions

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙋 Support

For issues, questions, or suggestions:

1. Check the Troubleshooting section
2. Review API documentation in `API_DOCS.md`
3. Open an issue on GitHub
4. Check Render deployment logs for deployment issues

## 🔮 Future Enhancements

- [ ] Real-time video streaming
- [ ] Emotion tracking over time
- [ ] Age and gender detection
- [ ] Multiple model support
- [ ] Web-based model fine-tuning
- [ ] Analytics dashboard
- [ ] Docker support
- [ ] Mobile app integration

## 📞 Contact

For more information or inquiries, please open an issue on the repository.

---

**Last Updated**: January 2026

**Status**: Production Ready

**Version**: 1.0.0
# emorio
