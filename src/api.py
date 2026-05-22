import os
import io
import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from ultralytics import YOLO

app = FastAPI(title="Advanced Fruit Detection & Classification API")

# 1. Load Models on startup
CLASSIFIER_PATH = os.path.join(os.path.dirname(__file__), 'best_mobilenet_model.keras')
classifier = tf.keras.models.load_model(CLASSIFIER_PATH)

# Loading pre-trained YOLOv8 (6MB, auto-downloads on first load)
detector = YOLO('yolov8n.pt') 

# Mappings
COCO_FRUIT_MAP = {
    46: 'pisang',
    47: 'apel',
    49: 'jeruk'
}

CLASS_NAMES = [
    'freshapples', 'freshbanana', 'freshoranges', 
    'rottenapples', 'rottenbanana', 'rottenoranges'
]

@app.post("/predict")
async def predict_multi_fruits(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    # Run YOLOv8 detection
    results = detector(img)[0]
    
    fruits_found = []
    
    for idx, box in enumerate(results.boxes):
        cls_id = int(box.cls[0].item())
        
        # Task 2: Filter out non-fruits
        if cls_id not in COCO_FRUIT_MAP:
            continue
            
        fruit_name = COCO_FRUIT_MAP[cls_id]
        
        # Get coordinates
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        
        # Task 1: Crop individual fruits
        crop = img[ymin:ymax, xmin:xmax]
        if crop.size == 0:
            continue
            
        # Preprocess crop for MobileNetV2
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_resized = cv2.resize(crop_rgb, (224, 224))
        crop_tensor = np.expand_dims(crop_resized, axis=0).astype(np.float32)
        
        # Predict freshness
        preds = classifier(crop_tensor, training=False).numpy()[0]
        
        # Constrain Keras classification to only the detected fruit category
        if fruit_name == 'apel':
            fresh_prob, rotten_prob = preds[0], preds[3]
        elif fruit_name == 'pisang':
            fresh_prob, rotten_prob = preds[1], preds[4]
        else: # jeruk
            fresh_prob, rotten_prob = preds[2], preds[5]
            
        # Normalize probabilities between segar & busuk
        total_prob = fresh_prob + rotten_prob
        segar_conf = fresh_prob / total_prob if total_prob > 0 else 0
        busuk_conf = rotten_prob / total_prob if total_prob > 0 else 0
        
        condition = "segar"
        confidence = segar_conf
        notes = "Classification normal."
        
        if segar_conf >= busuk_conf:
            # Task 3: If segar confidence is below 60%, mark as busuk
            if segar_conf < 0.60:
                condition = "busuk"
                confidence = busuk_conf # Or keep segar_conf and note override
                notes = "Otomatis diubah menjadi busuk karena confidence segar di bawah 60%."
        else:
            condition = "busuk"
            confidence = busuk_conf

        fruits_found.append({
            "id": idx + 1,
            "class": fruit_name,
            "condition": condition,
            "confidence": float(confidence),
            "box": [int(ymin), int(xmin), int(ymax), int(xmax)],
            "notes": notes
        })
        
    return {
        "status": "success",
        "fruits_detected": fruits_found,
        "summary": {
            "total_detected": len(fruits_found),
            "segar": sum(1 for f in fruits_found if f["condition"] == "segar"),
            "busuk": sum(1 for f in fruits_found if f["condition"] == "busuk")
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}
