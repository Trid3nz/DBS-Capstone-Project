# Fruit Classification API Documentation

This API provides endpoints for detecting the type and freshness of fruits (Apples, Bananas, Oranges) using a MobileNetV2 machine learning model.

## Base URL
When running locally: `http://127.0.0.1:8000` (or the port specified by your Uvicorn server).

---

## Running Locally

To run the API in a local development environment, follow these steps:

1. **Install Dependencies:**
   Ensure you have Python installed, then install the required packages from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Server:**
   Run the FastAPI server using Uvicorn. From the root of the project directory (where `src` is located), execute:
   ```bash
   uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
   ```
   The `--reload` flag enables auto-reloading during development.

3. **Access the API:**
   The API will be available at `http://127.0.0.1:8000`. You can also view the interactive Swagger UI documentation at `http://127.0.0.1:8000/docs`.

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
| `return_image` | `boolean` | No (query) | If `true`, returns the JPEG image with bounding boxes and freshness labels drawn on it instead of JSON. Defaults to `false`. |

#### Success Response (200 OK)
Returns a list of all fruits detected in the image, along with their condition, confidence, and bounding boxes.

```json
{
  "status": "success",
  "fruits_detected": [
    {
      "id": 1,
      "class": "apel",
      "condition": "segar",
      "confidence": 0.98543,
      "box": [100, 150, 300, 400],
      "notes": "Classification normal."
    }
  ],
  "summary": {
    "total_detected": 1,
    "segar": 1,
    "busuk": 0
  }
}
```

#### Fallback Behavior (No Fruits Detected by YOLO)
If the YOLO detector fails to find any apples, bananas, or oranges, the API automatically falls back to classifying the **entire image** using the MobileNetV2 model.

- If the fallback classifier is confident (**confidence >= 40%**), it returns a single fruit prediction with the bounding box set to the entire image coordinates `[0, 0, height, width]` and note `Classification normal (Fallback Mode).`.
- If the fallback classifier is not confident (**confidence < 40%**), the API returns a `cannot_determine` status indicating the image content is unclear:
  ```json
  {
    "status": "cannot_determine",
    "message": "The model cannot confidently determine the fruit type or freshness from this image.",
    "confidence": 0.2354,
    "raw_prediction": "pisang (busuk)"
  }
  ```

---

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
