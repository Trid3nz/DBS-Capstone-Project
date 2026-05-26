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

### Business Rules & Threshold Logic

The `/predict` endpoint implements strict quality control rules defined by domain requirements:

> [!IMPORTANT]
> **1. The 60% Freshness Confidence Rule:**
> If MobileNetV2 predicts that a fruit is `segar` (fresh) but the confidence score is **below 60%** (`< 0.60`), the system automatically overrides the condition classification to `busuk` (rotten). The notes field will explicitly mention: `"Otomatis diubah menjadi busuk karena confidence segar di bawah 60%."` (or similar for fallback mode).

> [!NOTE]
> **2. YOLOv8 Fallback Classification:**
> If YOLOv8 does not find any fruits, the API processes the **entire image** using the MobileNetV2 model.
> * **Confidence >= 40%:** Returns a single fruit prediction with coordinates spanning the entire image size `[0, 0, height, width]` and notes marked as `"Classification normal (Fallback Mode)."`.
> * **Confidence < 40%:** Returns a `"cannot_determine"` response, indicating the model is not confident enough to verify the image content.

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
      "confidence": 0.98543,
      "box": [100, 150, 300, 400],
      "notes": "Classification normal."
    },
    {
      "id": 2,
      "class": "pisang",
      "condition": "busuk",
      "confidence": 0.8921,
      "box": [220, 50, 450, 380],
      "notes": "Otomatis diubah menjadi busuk karena confidence segar di bawah 60%."
    }
  ],
  "summary": {
    "total_detected": 2,
    "segar": 1,
    "busuk": 1
  }
}
```
* *Note on Box Coordinates format:* Bounding boxes are formatted as `[ymin, xmin, ymax, xmax]` matching standard cropping pixel index offsets.

##### B. Successful Fallback Detection (200 OK - `return_image=false`)
When no individual objects are identified, and the model classifies the whole image above the `40%` confidence threshold:
```json
{
  "status": "success",
  "fruits_detected": [
    {
      "id": 1,
      "class": "jeruk",
      "condition": "segar",
      "confidence": 0.7429,
      "box": [0, 0, 1080, 1920],
      "notes": "Classification normal (Fallback Mode)."
    }
  ],
  "summary": {
    "total_detected": 1,
    "segar": 1,
    "busuk": 0
  }
}
```

##### C. Fallback Cannot Determine (200 OK - `return_image=false`)
When YOLOv8 finds zero fruits and the full-image classifier confidence is below `40%` (e.g. invalid object/kucing):
```json
{
  "status": "cannot_determine",
  "message": "The model cannot confidently determine the fruit type or freshness from this image.",
  "confidence": 0.2354,
  "raw_prediction": "pisang (busuk)"
}
```

##### D. Image Visualization Mode (200 OK - `return_image=true`)
* **Content-Type:** `image/jpeg`
* **Response Body:** Binary JPEG stream. The returned image has green boxes for `segar` fruits and red boxes for `busuk` fruits, alongside label text (e.g., `apel (segar 98%)`). If in fallback cannot determine state, the image is overlaid with the error string `"Cannot determine fruit type"`.

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
