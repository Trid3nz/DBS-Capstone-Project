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
        
        # Draw bounding box and label if return_image is requested
        if return_image:
            # Segar is Green, Busuk is Red
            color = (0, 255, 0) if condition == "segar" else (0, 0, 255)
            # Draw rectangle
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
            # Draw label text
            label = f"{fruit_name} ({condition} {confidence:.0%})"
            cv2.putText(img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
    # If YOLO didn't detect any fruits, fallback to MobileNetV2 on the entire image
    if len(fruits_found) == 0:
        # Preprocess entire image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_tensor = np.expand_dims(img_resized, axis=0).astype(np.float32)
        
        # Predict freshness
        preds = classifier(img_tensor, training=False).numpy()[0]
        predicted_idx = int(np.argmax(preds))
        confidence = float(preds[predicted_idx])
        original_class = CLASS_NAMES[predicted_idx]
        
        # Mapping
        MAP_INDONESIAN = {
            'freshapples': {'class': 'apel', 'condition': 'segar'},
            'freshbanana': {'class': 'pisang', 'condition': 'segar'},
            'freshoranges': {'class': 'jeruk', 'condition': 'segar'},
            'rottenapples': {'class': 'apel', 'condition': 'busuk'},
            'rottenbanana': {'class': 'pisang', 'condition': 'busuk'},
            'rottenoranges': {'class': 'jeruk', 'condition': 'busuk'},
        }
        
        mapping = MAP_INDONESIAN.get(original_class, {'class': 'unknown', 'condition': 'unknown'})
        fruit_name = mapping['class']
        condition = mapping['condition']
        
        # If the model is not confident (spewing nonsense), return a message
        if confidence < 0.40:
            if return_image:
                h, w, _ = img.shape
                cv2.putText(img, "Cannot determine fruit type", (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                _, encoded_img = cv2.imencode('.jpg', img)
                return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")
            else:
                return {
                    "status": "cannot_determine",
                    "message": "The model cannot confidently determine the fruit type or freshness from this image.",
                    "confidence": confidence,
                    "raw_prediction": f"{fruit_name} ({condition})"
                }
        
        # Apply Task 3 threshold override to fallback prediction
        notes = "Classification normal (Fallback Mode)."
        if condition == "segar" and confidence < 0.60:
            condition = "busuk"
            # Get probability of the corresponding rotten class (fresh class index + 3)
            rotten_idx = predicted_idx + 3
            confidence = float(preds[rotten_idx])
            notes = "Otomatis diubah menjadi busuk karena confidence segar di bawah 60% (Fallback Mode)."
            
        fruits_found.append({
            "id": 1,
            "class": fruit_name,
            "condition": condition,
            "confidence": confidence,
            "box": [0, 0, int(img.shape[0]), int(img.shape[1])],
            "notes": notes
        })
        
        if return_image:
            color = (0, 255, 0) if condition == "segar" else (0, 0, 255)
            # Draw border around the image
            cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), color, 4)
            label = f"Fallback: {fruit_name} ({condition} {confidence:.0%})"
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
