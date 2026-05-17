# TensorFlow.js Model Integration Guide

This guide explains how to integrate the converted Keras model (`mobilenet_fruit_model.keras` -> TFJS) into a web application using JavaScript. 

## 1. Hosting the Model

TensorFlow.js requires the model files to be served via HTTP/HTTPS. You cannot load them directly from the local file system using `file://` protocols.

Make sure you copy the entire `src/tfjs_model` directory (which contains `model.json` and the `.bin` files) to the public or static assets folder of your web application (e.g., `public/tfjs_model/`).

## 2. Including TensorFlow.js

You can install TensorFlow.js via NPM or include it via a script tag in your HTML.

**Option A: HTML Script Tag**
```html
<!-- Include the latest version of TensorFlow.js -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
```

**Option B: NPM**
```bash
npm install @tensorflow/tfjs
```
```javascript
import * as tf from '@tensorflow/tfjs';
```

## 3. Loading the Model

In your JavaScript code, use `tf.loadLayersModel()` to load the model asynchronously. 

```javascript
let model;

async function loadModel() {
    console.log("Loading model...");
    // Replace with the actual URL/path to your model.json
    model = await tf.loadLayersModel('/tfjs_model/model.json'); 
    console.log("Model loaded successfully!");
}

// Call this function when the app initializes
loadModel();
```

## 4. Class Labels

The model was trained to output probabilities for the following 6 classes. You should keep this array in your JavaScript to map the model's output index to the readable class name.

```javascript
const CLASS_NAMES = [
    'freshapples', 
    'freshbanana', 
    'freshoranges', 
    'rottenapples', 
    'rottenbanana', 
    'rottenoranges'
];
```

## 5. Image Preprocessing & Prediction

The model expects an input image of shape `[1, 224, 224, 3]`. 

> [!TIP]
> **Built-in Preprocessing:** Because the original Keras model included the `tf.keras.applications.mobilenet_v2.preprocess_input` equivalent in its architecture (using TrueDivide and Subtract layers), you **do not** need to normalize the pixel values to `[-1, 1]` or `[0, 1]` manually. You only need to pass the raw pixel values (0-255) as floats!

Here is the complete function to predict the class of an image element:

```javascript
async function predictFruit(imageElement) {
    if (!model) {
        console.error("Model is not loaded yet!");
        return;
    }

    // Start tracking the inference time
    const startTime = performance.now();

    // Wrap the inference in tf.tidy to automatically clean up memory (tensors)
    const prediction = tf.tidy(() => {
        // 1. Convert image element (<img>, <video>, or <canvas>) to a tensor
        let tensor = tf.browser.fromPixels(imageElement);
        
        // 2. Resize the image to 224x224 (the shape expected by MobileNetV2)
        tensor = tf.image.resizeBilinear(tensor, [224, 224]);
        
        // 3. Convert to float32 (values remain 0-255, the model does the rest)
        tensor = tensor.toFloat();
        
        // 4. Expand dimensions from [224, 224, 3] to [1, 224, 224, 3] for batching
        tensor = tensor.expandDims(0);
        
        // 5. Run the prediction
        return model.predict(tensor);
    });

    // 6. Extract the result
    // .dataSync() downloads the tensor values synchronously to a regular Float32Array
    const probabilities = prediction.dataSync(); 
    
    // Find the index of the highest probability
    const predictedClassIndex = prediction.argMax(1).dataSync()[0];
    const predictedClassName = CLASS_NAMES[predictedClassIndex];
    const confidence = probabilities[predictedClassIndex]; // As a decimal (e.g. 0.9)

    // Calculate processing time
    const endTime = performance.now();
    const processingTimeMs = endTime - startTime;

    // Parse the model's output into the requested JSON structure
    const isRotten = predictedClassName.startsWith("rotten");
    const label = isRotten ? "busuk" : "segar";

    let fruitClass = "";
    if (predictedClassName.includes("apples")) fruitClass = "apel";
    else if (predictedClassName.includes("banana")) fruitClass = "pisang";
    else if (predictedClassName.includes("oranges")) fruitClass = "jeruk";
    
    // Clean up the prediction tensor
    prediction.dispose();

    // Return the formatted JSON object
    return {
        label: label,
        class: fruitClass,
        confidence: confidence,
        processing_time_ms: processingTimeMs
    };
}
```

## 6. Example Usage

Assuming you have an HTML image element like this:
```html
<img id="my-fruit-image" src="test_apple.jpg" crossorigin="anonymous" />
<button onclick="runInference()">Predict</button>
<div id="result"></div>
```

You can trigger the prediction:
```javascript
async function runInference() {
    const image = document.getElementById('my-fruit-image');
    
    // Make sure the image is fully loaded before predicting!
    const result = await predictFruit(image);
    
    if (result) {
        // Output the JSON format
        console.log("Prediction Result:", JSON.stringify(result, null, 2));

        // Display in the DOM
        document.getElementById('result').innerHTML = `
            <pre>${JSON.stringify(result, null, 2)}</pre>
        `;
    }
}
```

## Summary Checklist

- [ ] Ensure `tfjs_model/model.json` and `tfjs_model/group1-shard...bin` files are publicly accessible via HTTP.
- [ ] Add `@tensorflow/tfjs` to your project.
- [ ] Initialize `tf.loadLayersModel(...)` properly and wait for it.
- [ ] Use `tf.tidy()` to prevent WebGL memory leaks when processing image tensors.
- [ ] Provide 224x224 float tensors (raw 0-255 RGB values) for inferences.
