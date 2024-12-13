from flask import Flask, request, jsonify
from vision_model import VisionModel  # Import model
import torch
from PIL import Image
from torchvision import transforms
from flask_cors import CORS

# Initialize Flask application
app = Flask(__name__)
CORS(app)

# Load model
try:
    model = VisionModel("densenet", num_classes=4)  # Adjust `num_classes` as needed
    model.load_state_dict(torch.load('model_final.pth', map_location=torch.device('cpu')))
    model.eval()
except Exception as e:
    model = None
    model_load_error = str(e)

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model is not available. Check server setup.'}), 500

    try:
        # Check if file is in the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file found in the request'}), 400

        file = request.files['file']
        image = Image.open(file.stream).convert('RGB')

        # Transform image to tensor
        img_tensor = transform(image).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted_class = torch.max(output, 1)

        return jsonify({'prediction': int(predicted_class.item())})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
