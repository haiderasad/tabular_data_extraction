from flask import Flask, request, jsonify
import base64
import io
from PIL import Image
from table_extractor import process_pdf
from TableExtractor import OCRProcessor
from io import BytesIO

app = Flask(__name__)

ocr_processor = None

@app.before_first_request
def initialize_ocr_processor():
    global ocr_processor
    ocr_processor = OCRProcessor()
    
@app.route('/process-image', methods=['POST'])
def process_image():
    data = request.get_json()
    if not data or 'image_base64' not in data:
        return jsonify({'error': 'Invalid request'}), 400

    image_data = data['image_base64']
    image_data = bytes(image_data, 'utf-8')
    image_data = base64.b64decode(image_data)
    ip_image = Image.open(io.BytesIO(image_data))
    
    img_det,json_table=ocr_processor.process_pdf(ip_image)
    
    print(json_table)
    
    # Convert the PIL Image to a bytes object
    buffered = BytesIO()
    img_det.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    results = {
        'json_result': json_table,
        'detected_tables_base64':img_str
    }
    
    return jsonify(results), 200

if __name__ == '__main__':
    app.run(debug=True)
