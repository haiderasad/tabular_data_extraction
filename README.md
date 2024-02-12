
# Image Processing API for Table Extraction

## Introduction
This API is designed to extract tabular data from images, leveraging advanced OCR and table detection techniques. It's ideal for digitizing documents and automating data extraction from various formats.

## Requirements
- Python 3.8 or newer
- Flask
- Pillow for image handling
- Any specific libraries for OCR and table detection (e.g., Tesseract, PyTesseract, or a custom model)

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
conda install flask pillow
# Use conda or pip to install other required libraries, such as OCR tools.
```

### Using pip
1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Install the required Python packages:
```bash
pip install -r requirements.txt
```

## API Usage
Start the Flask server by running:
```bash
python app.py
```
To process an image, send a POST request with a base64-encoded image string. Use the following curl command as an example:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"image_base64": "<base64_string>"}' http://localhost:5000/process-image
```

## API Reference
### Endpoint: `/process-image`
- **Method:** POST
- **Body:** JSON object containing a base64-encoded image string.
- **Response:** JSON object with the extracted table data.

## Contributing
We welcome contributions! Please fork the repository and submit pull requests with your suggested changes.

## License
This project is open-sourced under the MIT License. See the LICENSE file for more details.
