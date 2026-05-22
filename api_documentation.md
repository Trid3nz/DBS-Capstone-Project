# Fruit Classification API Documentation

This API provides endpoints for detecting the type and freshness of fruits (Apples, Bananas, Oranges) using a MobileNetV2 machine learning model.

## Base URL
When running locally: `http://127.0.0.1:8000` (or the port specified by your Uvicorn server).

---

## Endpoints

### 1. Health Check
Checks the operational status of the API and verifies if the machine learning model is loaded successfully.

- **URL:** `/health`
- **Method:** `GET`
- **Authentication:** None required

#### Success Response (200 OK)
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "d:\\Coding Camp\\New folder\\DBS-Capstone-Project\\src\\best_mobilenet_model.keras"
}
```

#### Error Response (If model failed to load)
```json
{
  "status": "unhealthy",
  "model_loaded": false,
  "model_path": "src/best_mobilenet_model.keras"
}
```

---

### 2. Predict Fruit
Analyzes an uploaded image and predicts the fruit type and condition (fresh or rotten).

- **URL:** `/predict`
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`
- **Authentication:** None required

#### Request Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | `file` | Yes | The image file to analyze (JPEG, PNG, etc.). |

#### Success Response (200 OK)
Returns the classification of the fruit in Indonesian (`apel`, `pisang`, `jeruk`) and its condition (`segar`, `busuk`), along with confidence scores.

```json
{
  "status": "success",
  "data": {
    "class": "apel",
    "condition": "segar",
    "confidence": 0.98543,
    "original_class": "freshapples",
    "probabilities": {
      "freshapples": 0.98543,
      "freshbanana": 0.00123,
      "freshoranges": 0.00054,
      "rottenapples": 0.01011,
      "rottenbanana": 0.00012,
      "rottenoranges": 0.00257
    }
  }
}
```

#### Error Responses

**400 Bad Request** (Invalid File or Failed to Read)
```json
{
  "detail": "Invalid image file format. Ensure the uploaded file is an image."
}
```

**503 Service Unavailable** (Model Not Loaded)
```json
{
  "detail": "Model is not loaded on server. Please check server configuration and path."
}
```

**500 Internal Server Error** (Inference Error)
```json
{
  "detail": "Error running inference: <error_message>"
}
```

---

## Example Usage

### Using Python (`requests` library)
```python
import requests

url = "http://127.0.0.1:8000/predict"
file_path = "path/to/your/image.jpg"

with open(file_path, "rb") as image_file:
    files = {"file": image_file}
    response = requests.post(url, files=files)

print(response.json())
```

### Using cURL
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/image.jpg"
```
