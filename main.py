from flask import Flask, request, jsonify
from vision_model import VisionModel  # Import model
import torch
from PIL import Image
from torchvision import transforms
import io
from flask_cors import CORS

# Inisialisasi aplikasi Flask
app = Flask(__name__)
CORS(app)

# Muat model
model = VisionModel("densenet", num_classes=10)  # Ganti `num_classes` sesuai kebutuhan Anda
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()

# Transformasi untuk gambar
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Baca file gambar dari request
        if 'file' not in request.files:
            return jsonify({'error': 'File tidak ditemukan dalam permintaan'}), 400

        file = request.files['file']
        image = Image.open(file.stream)

        # Transformasi gambar menjadi tensor
        img_tensor = transform(image).unsqueeze(0)

        # Prediksi
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted_class = torch.max(output, 1)

        return jsonify({'prediction': int(predicted_class.item())})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
