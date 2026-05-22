import numpy as np
import cv2
from fastapi.testclient import TestClient
import sys
import os

# Set path to import from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api import app

client = TestClient(app)

def test_health():
    print("=== Testing /health endpoint ===")
    response = client.get("/health")
    assert response.status_code == 200
    print("Response JSON:")
    print(response.json())
    print("Health check passed!\n")

def test_predict():
    print("=== Testing /predict endpoint ===")
    # Generate a dummy image (224x224x3) to simulate an image file
    dummy_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    
    # Encode dummy image to JPEG format bytes
    success, encoded_img = cv2.imencode('.jpg', dummy_img)
    assert success, "Failed to encode dummy image to JPEG format"
    img_bytes = encoded_img.tobytes()
    
    # Send image to the endpoint using a mock multipart form upload
    response = client.post(
        "/predict",
        files={"file": ("dummy.jpg", img_bytes, "image/jpeg")}
    )
    
    assert response.status_code == 200, f"Predict failed: {response.text}"
    result = response.json()
    print("Response JSON:")
    import json
    print(json.dumps(result, indent=2))
    
    # Validate structure
    assert result["status"] == "success", "Response status is not success"
    data = result["data"]
    assert "class" in data, "class is missing from response data"
    assert "condition" in data, "condition is missing from response data"
    assert "confidence" in data, "confidence is missing from response data"
    assert "original_class" in data, "original_class is missing from response data"
    assert "probabilities" in data, "probabilities are missing from response data"
    
    print("\nDummy prediction test passed successfully!")

if __name__ == "__main__":
    try:
        test_health()
        test_predict()
        print("\nAll programmatic tests passed successfully!")
    except AssertionError as ae:
        print(f"\nAssertion Error during test: {ae}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during test: {e}")
        sys.exit(1)
