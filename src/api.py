import os
import io
import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Fruit Classification API",
    description="API untuk mendeteksi kesegaran buah (Apel, Pisang, Jeruk) menggunakan MobileNetV2",
    version="1.0.0"
)

# Enable CORS for fullstack web and mobile apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve model path dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_mobilenet_model.keras')

print(f"Loading Keras model from {MODEL_PATH}...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
    model_loaded = True
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    # Fallback to local execution path
    try:
        MODEL_PATH = 'src/best_mobilenet_model.keras'
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully from fallback path!")
        model_loaded = True
    except Exception as fallback_err:
        print(f"Fallback model loading failed: {fallback_err}")
        model = None
        model_loaded = False

CLASS_NAMES = [
    'freshapples', 
    'freshbanana', 
    'freshoranges', 
    'rottenapples', 
    'rottenbanana', 
    'rottenoranges'
]

MAP_INDONESIAN = {
    'freshapples': {'class': 'apel', 'condition': 'segar'},
    'freshbanana': {'class': 'pisang', 'condition': 'segar'},
    'freshoranges': {'class': 'jeruk', 'condition': 'segar'},
    'rottenapples': {'class': 'apel', 'condition': 'busuk'},
    'rottenbanana': {'class': 'pisang', 'condition': 'busuk'},
    'rottenoranges': {'class': 'jeruk', 'condition': 'busuk'},
}

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "model_path": MODEL_PATH
    }

@app.post("/predict")
async def predict_fruit(file: UploadFile = File(...)):
    if not model_loaded or model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model is not loaded on server. Please check server configuration and path."
        )
    
    # Read file bytes
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read upload file: {str(e)}")
    
    # Decode image using OpenCV
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file format. Ensure the uploaded file is an image.")
    
    # Convert BGR (OpenCV default) to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to 224x224 (Input shape expected by the MobileNetV2 architecture)
    img_resized = cv2.resize(img_rgb, (224, 224))
    
    # Expand dimensions for batching: (1, 224, 224, 3)
    img_tensor = np.expand_dims(img_resized, axis=0).astype(np.float32)
    
    # Run prediction
    try:
        # Using model(x, training=False) is faster and more memory efficient than model.predict()
        preds = model(img_tensor, training=False).numpy()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running inference: {str(e)}")
    
    # Process output probabilities
    probabilities = preds[0]
    predicted_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_idx])
    original_class = CLASS_NAMES[predicted_idx]
    
    # Mapping to Indonesian output
    mapping = MAP_INDONESIAN.get(original_class, {'class': 'unknown', 'condition': 'unknown'})
    
    # Prepare class probability breakdown
    prob_breakdown = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}
    
    return {
        "status": "success",
        "data": {
            "class": mapping['class'],
            "condition": mapping['condition'],
            "confidence": confidence,
            "original_class": original_class,
            "probabilities": prob_breakdown
        }
    }
