from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from ultralytics import YOLO
import os
import uuid
import shutil # For cleaning up directories

app = Flask(__name__)
CORS(app)

# Load the YOLO model
try:
    model = YOLO("yolov8/best.pt")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    # Exit or handle gracefully if the model can't be loaded
    # For now, we'll let the app run, but predict calls will fail.
    model = None # Set model to None if loading fails

# Define upload and prediction folders
UPLOAD_FOLDER = 'static/uploads'
# YOLO will create its own 'runs/detect/predict' directories
# We don't need a separate 'RESULT_FOLDER' as YOLO handles saving.

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'YOLO model not loaded. Please check server logs.'}), 503 # Service Unavailable

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']

    # Basic file type validation (can be more robust)
    if not image.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        return jsonify({'error': 'Invalid file type. Only images (png, jpg, jpeg, gif) are allowed.'}), 400

    filename = f"{uuid.uuid4()}_{image.filename}" # Keep original extension, add UUID
    image_path = os.path.join(UPLOAD_FOLDER, filename)

    try:
        image.save(image_path)
    except Exception as e:
        return jsonify({'error': f'Failed to save image: {str(e)}'}), 500

    results_dir = None # Initialize to None for cleanup in finally block
    try:
        # Perform prediction. YOLO's save=True creates a new 'predictX' folder.
        # We also pass a project and name to have more control over the output path,
        # making cleanup easier.
        yolo_results = model.predict(
            source=image_path,
            save=True,
            conf=0.3,
            project='runs', # Default project folder
            name='detect_results', # A consistent name for prediction runs
            exist_ok=True # Allows overwriting if 'detect_results' exists, or creates 'detect_results2' etc.
        )

        if not yolo_results:
            return jsonify({'error': 'Prediction failed or no results returned by YOLO.'}), 500

        # YOLO typically saves results in a directory like 'runs/detect_results' or 'runs/detect_results2'
        # The first result object contains the path to its save directory.
        results_dir = yolo_results[0].save_dir
        predicted_image_name = os.path.basename(yolo_results[0].path) # Get the filename YOLO used
        result_file_path = os.path.join(results_dir, predicted_image_name)


        if not os.path.exists(result_file_path):
            return jsonify({'error': 'Predicted image file not found on server after prediction.'}), 500

        # Send the predicted image back
        return send_file(result_file_path, mimetype='image/jpeg')

    except Exception as e:
        print(f"Prediction error: {e}") # Log the error on the server
        return jsonify({'error': f'Prediction failed: {str(e)}. Check server logs for details.'}), 500
    finally:
        # Clean up the original uploaded image
        if os.path.exists(image_path):
            os.remove(image_path)

        # IMPORTANT: Cleaning up YOLO's output directory immediately after send_file()
        # can be problematic if send_file() hasn't finished sending the file.
        # For production, consider:
        # 1. Returning the image bytes directly without saving to disk on the server.
        # 2. A separate background task to clean up old prediction folders.
        # For this example, we'll keep the cleanup here for simplicity, but be aware of the caveat.
        if results_dir and os.path.exists(results_dir):
             try:
                 # Be careful with shutil.rmtree as it deletes everything recursively
                 # Consider a more granular cleanup if other files are in results_dir that you want to keep
                 # For YOLO, it usually creates unique folders, so rmtree is often fine.
                 shutil.rmtree(results_dir)
             except Exception as e:
                 print(f"Error cleaning up results directory {results_dir}: {e}")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)