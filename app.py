from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from ultralytics import YOLO
import os
import uuid

app = Flask(__name__)
CORS(app)

# Load YOLOv8 model
model = YOLO("yolov8/best.pt")

# Upload folder setup
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    filename = f"{uuid.uuid4()}.jpg"
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(image_path)

    # Run prediction
    results = model.predict(source=image_path, save=True, conf=0.3)

    # Get path to annotated image
    result_dir = results[0].save_dir
    result_file = os.path.join(result_dir, filename)

    # Return annotated image
    return send_file(result_file, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

