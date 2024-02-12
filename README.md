
# Image Processing API for Table Extraction

## Introduction
This API is designed to extract tabular data from images, leveraging advanced OCR and table detection techniques. It's ideal for digitizing documents and automating data extraction from various formats.

## Installation
### Using Conda
If you prefer using Conda for managing your Python environments, you can create a new environment and install the necessary packages as follows:

1. Create a new Conda environment:
```bash
conda create --name table_extraction python=3.8
```
2. Activate the Conda environment:
```bash
conda activate table_extraction
```
3. Install the required packages:
```bash
pip install -r requirements.txt
```

## API Usage
Start the Flask server by running:
```bash
python main.py
```
To process an image, send a POST request with a base64-encoded image string. Use the following curl command as an example:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"image_base64": "<base64_string>"}' http://localhost:5000/process-image
```

## API Reference
### Endpoint: `/process-image`
- **Method:** POST
- **Body:** keys: 'image_base64' .JSON object containing a base64-encoded image string.
- **Response:** keys:'json_result' and 'detected_tables_base64' . JSON object with the extracted table data and base64 table image cropped.


### IOS CONVERSION COMPATIBILITY

Deploying Huggingface models on iOS devices is possible through model optimization and conversion techniques such as quantization, pruning, and using intermediary formats like ONNX for conversion to CoreML. Apple's CoreML framework supports deploying machine learning models on iOS devices, but careful model optimization and testing are crucial to ensure performance and feasibility on mobile devices.

Converting Transformer models for iOS involves several steps, typically requiring model optimization and translation into a format compatible with Core ML, Apple's machine learning framework for iOS devices. Here's a high-level overview:

1. Export to ONNX: First, export the PyTorch models to ONNX format, a popular open model format compatible across different ML frameworks.
2. Optimize the ONNX Model: Use the ONNX Runtime to optimize the model for inference efficiency.
3. Convert to Core ML: Use the coremltools library to convert the optimized ONNX model to Core ML format.
4. Integrate into iOS App: Finally, integrate the Core ML model into your iOS app using Xcode and the Core ML framework.