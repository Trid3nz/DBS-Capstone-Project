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
CLASSIFIER_PATH = os.path.join(os.path.dirname(__file__), 'Model/best_mobilenet_model.keras')
classifier = tf.keras.models.load_model(CLASSIFIER_PATH)

# Loading pre-trained YOLOv8 (6MB, auto-downloads on first load)
detector = YOLO(os.path.join(os.path.dirname(__file__), 'Model/yolo26n.pt')) 

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
async def predict_multi_fruits(file: UploadFile = File(...), return_image: bool = False):
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
        segar_conf = fresh_prob / total_prob if total_prob > 0 else 0.0
        busuk_conf = rotten_prob / total_prob if total_prob > 0 else 0.0
        
        condition = "segar" if segar_conf >= busuk_conf else "busuk"

        fruits_found.append({
            "id": idx + 1,
            "class": fruit_name,
            "condition": condition,
            "segar_confidence": float(segar_conf),
            "busuk_confidence": float(busuk_conf),
            "box": [int(ymin), int(xmin), int(ymax), int(xmax)]
        })
        
        # Draw bounding box and label if return_image is requested
        if return_image:
            # Segar is Green, Busuk is Red
            color = (0, 255, 0) if condition == "segar" else (0, 0, 255)
            # Draw rectangle
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
            # Draw label text
            conf = segar_conf if condition == "segar" else busuk_conf
            label = f"{fruit_name} ({condition} {conf:.0%})"
            cv2.putText(img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
    # If YOLO didn't detect any fruits, fallback to MobileNetV2 on the entire image
    if len(fruits_found) == 0:
        # Preprocess entire image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_tensor = np.expand_dims(img_resized, axis=0).astype(np.float32)
        
        # Predict freshness
        preds = classifier(img_tensor, training=False).numpy()[0]
        
        # Sum fresh and rotten probabilities for each fruit category
        apple_total = preds[0] + preds[3]
        banana_total = preds[1] + preds[4]
        orange_total = preds[2] + preds[5]
        
        # Find which fruit is predicted
        totals = [apple_total, banana_total, orange_total]
        fruit_idx = int(np.argmax(totals))
        
        fruit_names = ['apel', 'pisang', 'jeruk']
        fruit_name = fruit_names[fruit_idx]
        
        if fruit_idx == 0:
            fresh_prob, rotten_prob = preds[0], preds[3]
        elif fruit_idx == 1:
            fresh_prob, rotten_prob = preds[1], preds[4]
        else:
            fresh_prob, rotten_prob = preds[2], preds[5]
            
        total_prob = fresh_prob + rotten_prob
        segar_conf = fresh_prob / total_prob if total_prob > 0 else 0.0
        busuk_conf = rotten_prob / total_prob if total_prob > 0 else 0.0
        
        condition = "segar" if segar_conf >= busuk_conf else "busuk"
            
        fruits_found.append({
            "id": 1,
            "class": fruit_name,
            "condition": condition,
            "segar_confidence": float(segar_conf),
            "busuk_confidence": float(busuk_conf),
            "box": [0, 0, int(img.shape[0]), int(img.shape[1])]
        })
        
        if return_image:
            color = (0, 255, 0) if condition == "segar" else (0, 0, 255)
            # Draw border around the image
            cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), color, 4)
            conf = segar_conf if condition == "segar" else busuk_conf
            label = f"Fallback: {fruit_name} ({condition} {conf:.0%})"
            cv2.putText(img, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if return_image:
        _, encoded_img = cv2.imencode('.jpg', img)
        return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")
        
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
