import os
import tensorflow as tf
import numpy as np

def evaluate():
    # 1. Setup paths
    model_path = os.path.join(os.path.dirname(__file__), 'Model/best_mobilenet_model.keras')
    test_dir = r"D:\Coding Camp\New folder\Dataset\test"
    
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found at {test_dir}")
        return

    # 2. Load the Model
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    
    # 3. Load the Test Dataset
    BATCH_SIZE = 32
    IMG_SIZE = (224, 224)
    print("Loading test dataset...")
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        shuffle=False,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE
    )
    
    class_names = test_dataset.class_names
    print(f"Classes: {class_names}")

    # Prefetch and Preprocess
    AUTOTUNE = tf.data.AUTOTUNE
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
    
    # 4. Evaluate Model
    print("\nEvaluating model...")
    
    # Get standard loss and accuracy
    loss, accuracy = model.evaluate(test_dataset, verbose=1)
    
    # Calculate MAE (Mean Absolute Error)
    # Since this is classification, we'll calculate MAE on the predicted probabilities vs one-hot encoded true labels
    print("\nCalculating MAE...")
    y_true = []
    y_pred = []
    
    for images, labels in test_dataset:
        preds = model.predict(images, verbose=0)
        y_pred.extend(preds)
        y_true.extend(labels.numpy())
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # One-hot encode y_true to match y_pred shape
    y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=len(class_names))
    
    # Calculate MAE: mean(|y_true - y_pred|)
    mae = np.mean(np.abs(y_true_onehot - y_pred))
    
    print("\n" + "="*40)
    print("MODEL EVALUATION RESULTS")
    print("="*40)
    print(f"Accuracy : {accuracy:.4f} (Target: >= 0.85)")
    print(f"MAE      : {mae:.4f} (Target: <= 0.02)")
    print("="*40)
    
    # Generate simple classification report
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    from sklearn.metrics import classification_report, confusion_matrix
    report_str = classification_report(y_true, y_pred_classes, target_names=class_names)
    report_dict = classification_report(y_true, y_pred_classes, target_names=class_names, output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    
    print("\nClassification Report:")
    print(report_str)
    
    # 5. Export Results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save text report
    report_txt_path = os.path.join(results_dir, 'evaluation_report.txt')
    with open(report_txt_path, 'w') as f:
        f.write("="*40 + "\n")
        f.write("MODEL EVALUATION RESULTS\n")
        f.write("="*40 + "\n")
        f.write(f"Accuracy : {accuracy:.4f} (Target: >= 0.85)\n")
        f.write(f"MAE      : {mae:.4f} (Target: <= 0.02)\n")
        f.write("="*40 + "\n\n")
        f.write("Classification Report:\n")
        f.write(report_str + "\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(conf_matrix) + "\n")
    
    # Save structured JSON
    import json
    results_json_path = os.path.join(results_dir, 'evaluation_results.json')
    json_data = {
        "accuracy": float(accuracy),
        "mae": float(mae),
        "classification_report": report_dict,
        "confusion_matrix": conf_matrix.tolist(),
        "classes": class_names
    }
    with open(results_json_path, 'w') as f:
        json.dump(json_data, f, indent=4)
        
    print("\n" + "-"*40)
    print(f"Results exported successfully to:")
    print(f" - {report_txt_path}")
    print(f" - {results_json_path}")
    print("-"*40)

if __name__ == "__main__":
    evaluate()
