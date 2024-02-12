import base64
import requests
from io import BytesIO
from PIL import Image

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def send_image_to_api(image_base64, api_url):
    headers = {'Content-Type': 'application/json'}
    data = {"image_base64": image_base64}
    response = requests.post(api_url, json=data, headers=headers)
    return response.json()

# Replace 'path/to/your/image.jpg' with the actual path to your image
image_path = '/home/haider/Downloads/yoo.png'

api_url = 'http://localhost:5000/process-image'

# Encode the image
image_base64 = encode_image_to_base64(image_path)

# Send the encoded image to the API and get the response
response = send_image_to_api(image_base64, api_url)

image_data = base64.b64decode(response['detected_tables_base64'])
    
# Convert the bytes to a PIL Image
image = Image.open(BytesIO(image_data))

# Display the image
image.show()

print(response['json_result'])
