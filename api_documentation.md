# Fruit Classification & Detection API Documentation

This API provides highly-optimized endpoints for detecting the presence, type, and freshness condition of fruits (Apples, Bananas, Oranges) using a robust hybrid approach combining **YOLOv8** (for real-time object detection/localization) and a custom-trained **MobileNetV2** (for high-fidelity freshness classification).

---

## Architecture Overview

The API implements a dual-model hybrid pipeline:
1. **Detection (YOLOv8):** Checks the image for target fruits (`apel`, `pisang`, `jeruk`) and extracts their precise bounding boxes, discarding non-fruit objects automatically.
2. **Classification (MobileNetV2):** Crops each detected fruit and evaluates its condition (fresh vs. rotten).
3. **Fallback Mechanism:** If YOLOv8 detects absolutely zero target fruits, the pipeline falls back to classifying the **entire image** using the MobileNetV2 model.

---

## Base URL

When running locally, the default endpoint is:
* **Base URL:** `http://127.0.0.1:8000`

---

## Running Locally

Follow these steps to run the development server on your machine:

### 1. Install Dependencies
Ensure you have Python installed, then install all the backend packages:
```bash
pip install -r requirements.txt
```

### 2. Start the Server
Run the FastAPI application with Uvicorn from the root folder:
```bash
uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```
* The `--reload` flag enables auto-reloading whenever you make changes to your codebase.

### 3. Interactive Documentation
Once the server is running, you can access the interactive Swagger UI at:
* **Interactive UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **ReDoc Alternative:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## Endpoints

### 1. Health Check
Checks if the API server is up and running.

- **URL:** `/health`
- **Method:** `GET`
- **Authentication:** None required

#### Success Response (200 OK)
```json
{
  "status": "healthy"
}
```

---

### 2. Predict Multi Fruits
Processes an uploaded image, runs YOLOv8 fruit detection, crops the detected fruits, runs MobileNetV2 to classify freshness, and applies custom business threshold overrides.

- **URL:** `/predict`
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`
- **Authentication:** None required

#### Request Parameters

| Parameter | Type | Required | Position | Description |
|-----------|------|----------|----------|-------------|
| `file` | `file` (Binary) | **Yes** | Body | The target image file to analyze (supported: JPEG, PNG, BMP, etc.). |
| `return_image` | `boolean` | No | Query | If `true`, the API bypasses the JSON payload response and instead returns a processed **JPEG image** containing annotated bounding boxes, class labels, condition, and confidence values. Defaults to `false`. |

---

### Business Rules & Threshold Logic (Backend Responsibility)

> [!NOTE]
> The API focuses exclusively on object detection (YOLOv8) and raw feature classification (MobileNetV2). All business rules and custom threshold overrides—such as the **60% Freshness Confidence Rule** (auto-converting low-confidence `segar` items to `busuk`) or the **40% Fallback safety threshold**—are deferred to the main backend application. The API provides raw normalized confidence scores for both states to facilitate this.

> [!IMPORTANT]
> **YOLOv8 Fallback Classification:**
> If YOLOv8 does not find any fruits, the API automatically classifies the **entire image** using the MobileNetV2 model. It returns a single fruit prediction spanning the entire image size `[0, 0, height, width]` with normalized `segar_confidence` and `busuk_confidence`, letting the backend handle any confidence limits.

---

#### Response Examples

##### A. Successful Bounded Detection (200 OK - `return_image=false`)
When YOLOv8 successfully detects fruits and classifies their freshness:
```json
{
  "status": "success",
  "fruits_detected": [
    {
      "id": 1,
      "class": "apel",
      "condition": "segar",
      "segar_confidence": 0.98543,
      "busuk_confidence": 0.01457,
      "box": [100, 150, 300, 400]
    },
    {
      "id": 2,
      "class": "pisang",
      "condition": "segar",
      "segar_confidence": 0.5521,
      "busuk_confidence": 0.4479,
      "box": [220, 50, 450, 380]
    }
  ],
  "summary": {
    "total_detected": 2,
    "segar": 2,
    "busuk": 0
  }
}
```
* *Note on Box Coordinates format:* Bounding boxes are formatted as `[ymin, xmin, ymax, xmax]` matching standard cropping pixel index offsets.

##### B. Successful Fallback Detection (200 OK - `return_image=false`)
When no individual objects are identified, and the model classifies the whole image:
```json
{
  "status": "success",
  "fruits_detected": [
    {
      "id": 1,
      "class": "jeruk",
      "condition": "segar",
      "segar_confidence": 0.7429,
      "busuk_confidence": 0.2571,
      "box": [0, 0, 1080, 1920]
    }
  ],
  "summary": {
    "total_detected": 1,
    "segar": 1,
    "busuk": 0
  }
}
```

##### C. Image Visualization Mode (200 OK - `return_image=true`)
* **Content-Type:** `image/jpeg`
* **Response Body:** Binary JPEG stream. The returned image has green boxes for `segar` fruits and red boxes for `busuk` fruits, alongside label text showing the class, simple condition, and its confidence (e.g., `apel (segar 98%)`).

---

#### Error Responses

##### **400 Bad Request**
Returned when the uploaded file is not a valid image format.
```json
{
  "detail": "Invalid image"
}
```

##### **500 Internal Server Error**
Returned when an unexpected inference error occurs during model evaluation.
```json
{
  "detail": "Error running inference: <error_message>"
}
```

---

## Integration Examples

### 1. Python (`requests`)
```python
import requests

url = "http://127.0.0.1:8000/predict"
image_path = "sample_fruit.jpg"

with open(image_path, "rb") as file_bytes:
    files = {"file": ("image.jpg", file_bytes, "image/jpeg")}
    response = requests.post(url, files=files)

print("Status Code:", response.status_code)
print("Response JSON:\n", response.json())
```

### 2. cURL
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@sample_fruit.jpg;type=image/jpeg' \
  -F 'return_image=false'
```

### 3. JavaScript (`fetch` - JSON Mode)
```javascript
const formData = new FormData();
formData.append("file", imageBlobOrFile);

fetch("http://127.0.0.1:8000/predict?return_image=false", {
  method: "POST",
  body: formData
})
.then(response => response.json())
.then(data => console.log(data))
.catch(error => console.error("Error:", error));
```

### 4. JavaScript (`fetch` - Image Visualization Mode)
```javascript
const formData = new FormData();
formData.append("file", imageBlobOrFile);

fetch("http://127.0.0.1:8000/predict?return_image=true", {
  method: "POST",
  body: formData
})
.then(response => response.blob())
.then(imageBlob => {
  const imageObjectURL = URL.createObjectURL(imageBlob);
  document.getElementById("myResultImage").src = imageObjectURL;
})
.catch(error => console.error("Error visualising image:", error));
```
