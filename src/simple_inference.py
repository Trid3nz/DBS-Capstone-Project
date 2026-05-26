import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

def main():
    if len(sys.argv) < 2:
        print("Usage: python simple_inference.py <path_to_image>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        sys.exit(1)

    print(f"Running inference on: {image_path}\n")

    # Load YOLOv8 for detection
    print("Loading YOLOv8 detector...")
    detector = YOLO('yolov8n.pt')
    
    # Load MobileNetV2 for classification
    print("Loading MobileNetV2 classifier...")
    model_path = os.path.join(os.path.dirname(__file__), 'best_mobilenet_model.keras')
    classifier = tf.keras.models.load_model(model_path)
    
    # Mappings
    COCO_FRUIT_MAP = {46: 'pisang', 47: 'apel', 49: 'jeruk'}
    
    # Read Image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image.")
        sys.exit(1)
        
    # YOLOv8 Inference
    print("\nDetecting fruits...")
    results = detector(img)[0]
    
    fruits_found = 0
    
    for idx, box in enumerate(results.boxes):
        cls_id = int(box.cls[0].item())
        
        # Filter non-fruits
        if cls_id not in COCO_FRUIT_MAP:
            continue
            
        fruits_found += 1
        fruit_name = COCO_FRUIT_MAP[cls_id]
        
        # Coordinates
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        
        # Crop & Preprocess
        crop = img[ymin:ymax, xmin:xmax]
        if crop.size == 0: continue
            
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_resized = cv2.resize(crop_rgb, (224, 224))
        crop_tensor = np.expand_dims(crop_resized, axis=0).astype(np.float32)
        
        # Predict Freshness
        preds = classifier(crop_tensor, training=False).numpy()[0]
        
        if fruit_name == 'apel':
            fresh_prob, rotten_prob = preds[0], preds[3]
        elif fruit_name == 'pisang':
            fresh_prob, rotten_prob = preds[1], preds[4]
        else: # jeruk
            fresh_prob, rotten_prob = preds[2], preds[5]
            
        total_prob = fresh_prob + rotten_prob
        segar_conf = fresh_prob / total_prob if total_prob > 0 else 0
        busuk_conf = rotten_prob / total_prob if total_prob > 0 else 0
        
        condition = "segar"
        confidence = segar_conf
        
        if segar_conf >= busuk_conf:
            if segar_conf < 0.60:
                condition = "busuk (Threshold Override)"
                confidence = busuk_conf
        else:
            condition = "busuk"
            confidence = busuk_conf
            
        print(f"--- Fruit {fruits_found} ---")
        print(f"Type      : {fruit_name}")
        print(f"Condition : {condition}")
        print(f"Confidence: {confidence:.2%}")
        print(f"BBox      : [{xmin}, {ymin}, {xmax}, {ymax}]\n")

    if fruits_found == 0:
        print("No target fruits (apel, pisang, jeruk) detected in the image.")

if __name__ == "__main__":
    main()
