from flask import Flask, request, jsonify, url_for
from flask_cors import CORS
from ultralytics import YOLO
import os
import uuid
import shutil
import logging
import subprocess
from collections import defaultdict # Import defaultdict for easy counting

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024 # 200 MB

try:
    model = YOLO("yolov8/best.pt")
    logging.info("YOLOv8 model loaded successfully from yolov8/best.pt.")
except Exception as e:
    logging.error(f"Model Load Error: Could not load yolov8/best.pt - {e}")
    model = None

# --- NEW: Dedicated folder for serving predicted results publicly ---
# Make sure this is accessible as a static endpoint.
RESULT_FOLDER = 'static/results'
os.makedirs(RESULT_FOLDER, exist_ok=True)
logging.info(f"Result folder created/verified at: {RESULT_FOLDER}")

# Your existing upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
logging.info(f"Upload folder created/verified at: {UPLOAD_FOLDER}")

# --- Helper function to save and get public URL ---
def save_result_and_get_url(source_path, filename):
    unique_id = uuid.uuid4().hex
    ext = os.path.splitext(filename)[1]
    new_filename = f"{unique_id}{ext}"
    destination_path = os.path.join(RESULT_FOLDER, new_filename)
    shutil.copy(source_path, destination_path)
    # Use url_for to generate a public URL for the file
    # 'static' is the default endpoint for the 'static' folder in Flask
    public_url = url_for('static', filename=f'results/{new_filename}', _external=True)
    return public_url, destination_path # Return both URL and server path for potential cleanup

@app.route('/predict_image', methods=['POST'])
def predict_image():
    if model is None:
        logging.error("predict_image: Model not loaded. Returning 503.")
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 503

    if 'image' not in request.files:
        logging.warning("predict_image: No 'image' file part in request.")
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    allowed_image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
    if not image.filename or not image.filename.lower().endswith(allowed_image_extensions):
        logging.warning(f"predict_image: Invalid image format for {image.filename}. Supported: {", ".join(allowed_image_extensions)}")
        return jsonify({'error': f'Invalid image format. Supported formats: {", ".join(allowed_image_extensions)}'}), 400

    filename = f"{uuid.uuid4()}_{image.filename}"
    image_path = os.path.join(UPLOAD_FOLDER, filename)

    results_dir = None
    predicted_media_url = None
    predicted_media_server_path = None # To track the file in RESULT_FOLDER for cleanup
    try:
        image.save(image_path)
        logging.info(f"predict_image: Image saved to: {image_path}")

        results = model.predict(source=image_path, save=True, conf=0.3,
                                project='runs', name='detect_results', exist_ok=True)

        results_dir = results[0].save_dir
        result_file_path = os.path.join(results_dir, os.path.basename(image_path))

        if not os.path.exists(result_file_path):
            logging.error(f"predict_image: Prediction output file not found at expected path: {result_file_path}")
            found_output_files = [f for f in os.listdir(results_dir) if f.lower().endswith(allowed_image_extensions)]
            if found_output_files:
                result_file_path = os.path.join(results_dir, found_output_files[0])
                logging.info(f"predict_image: Found output image at fallback path: {result_file_path}")
            else:
                return jsonify({'error': 'Prediction output file not found. Internal server error.'}), 500

        # --- NEW: Save result to RESULT_FOLDER and get public URL ---
        predicted_media_url, predicted_media_server_path = save_result_and_get_url(result_file_path, os.path.basename(result_file_path))
        logging.info(f"predict_image: Public URL for predicted image: {predicted_media_url}")

        # --- NEW: Extract detection details and format description ---
        detections = []
        object_counts = defaultdict(int)

        if results and results[0].boxes:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]
                detections.append({'class': label, 'confidence': round(conf, 2)})
                object_counts[label] += 1
        
        description_parts = []
        if object_counts:
            for obj_class, count in object_counts.items():
                description_parts.append(f"{count} {obj_class}{'s' if count > 1 else ''}")
            description = ", ".join(description_parts)
        else:
            description = "No objects detected."


        # Return JSON with URL and description
        return jsonify({
            'status': 'success',
            'predicted_image_url': predicted_media_url,
            'description': description,
            'detections': detections,
            'total_objects': len(detections)
        }), 200

    except Exception as e:
        logging.error(f"predict_image: Prediction failed for {image.filename}: {str(e)}", exc_info=True)
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)
            logging.info(f"predict_image: Removed uploaded image: {image_path}")
        if results_dir and os.path.exists(results_dir):
            try:
                shutil.rmtree(results_dir)
                logging.info(f"predict_image: Removed YOLO results directory: {results_dir}")
            except OSError as e:
                logging.warning(f"predict_image: Error removing results directory {results_dir}: {e}")
        # IMPORTANT: For RESULT_FOLDER, you might need a separate cleanup strategy
        # (e.g., a background job to delete old files, or a client-side deletion request).
        # We are NOT deleting from RESULT_FOLDER here, as the client needs to fetch it.
        # if predicted_media_server_path and os.path.exists(predicted_media_server_path):
        #     os.remove(predicted_media_server_path)
        #     logging.info(f"predict_image: Removed predicted media from results folder: {predicted_media_server_path}")


@app.route('/predict_video', methods=['POST'])
def predict_video():
    if model is None:
        logging.error("predict_video: Model not loaded. Returning 503.")
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 503

    if 'video' not in request.files:
        logging.warning("predict_video: No 'video' file part in request.")
        return jsonify({'error': 'No video uploaded'}), 400

    video = request.files['video']
    allowed_video_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
    if not video.filename or not video.filename.lower().endswith(allowed_video_extensions):
        logging.warning(f"predict_video: Invalid video format for {video.filename}. Supported: {", ".join(allowed_video_extensions)}")
        return jsonify({'error': f'Invalid video format. Supported formats: {", ".join(allowed_video_extensions)}'}), 400

    original_filename = video.filename
    filename = f"{uuid.uuid4()}_{original_filename}"
    video_path = os.path.join(UPLOAD_FOLDER, filename)

    results_dir = None
    converted_video_path = None
    predicted_media_url = None
    predicted_media_server_path = None
    try:
        video.save(video_path)
        logging.info(f"predict_video: Video saved to: {video_path}")

        logging.info(f"predict_video: Starting YOLO prediction for video: {video_path}")
        results = model.predict(source=video_path, save=True, conf=0.3,
                                project='runs', name='video_results', exist_ok=True)

        results_dir = results[0].save_dir

        predicted_output_files = [f for f in os.listdir(results_dir) if f.lower().endswith('.avi')]

        if not predicted_output_files:
            logging.error(f"predict_video: No .avi video file found from YOLO prediction in: {results_dir}")
            return jsonify({'error': 'YOLO prediction did not produce an AVI file. Check YOLO output settings.'}), 500

        yolo_output_avi_path = os.path.join(results_dir, predicted_output_files[0])
        logging.info(f"predict_video: YOLO output AVI found: {yolo_output_avi_path}")

        converted_video_filename = f"{os.path.splitext(predicted_output_files[0])[0]}.mp4"
        converted_video_path = os.path.join(results_dir, converted_video_filename)

        logging.info(f"predict_video: Converting {yolo_output_avi_path} to {converted_video_path} (MP4) using FFmpeg.")

        command = [
            'ffmpeg',
            '-i', yolo_output_avi_path,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', 'faststart',
            '-y',
            converted_video_path
        ]

        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            logging.info(f"predict_video: FFmpeg conversion successful. Output: {result.stdout}, Errors: {result.stderr}")
        except subprocess.CalledProcessError as e:
            logging.error(f"predict_video: FFmpeg conversion failed (exit code {e.returncode}): {e.stderr}", exc_info=True)
            return jsonify({'error': f'Video conversion failed on server: {e.stderr}'}), 500
        except FileNotFoundError:
            logging.error("predict_video: FFmpeg command not found. Is FFmpeg installed and in your server's PATH?")
            return jsonify({'error': 'FFmpeg is not installed on the server for video conversion. Please install it.'}), 500
        except Exception as e:
            logging.error(f"predict_video: Unexpected error during FFmpeg execution: {str(e)}", exc_info=True)
            return jsonify({'error': f'An unexpected error occurred during video conversion: {str(e)}'}), 500

        if not os.path.exists(converted_video_path):
            logging.error(f"predict_video: Converted MP4 video file not found at: {converted_video_path}. Conversion may have failed silently.")
            return jsonify({'error': 'Converted video not found after conversion. Conversion might have failed.'}), 500

        # --- NEW: Save result to RESULT_FOLDER and get public URL ---
        predicted_media_url, predicted_media_server_path = save_result_and_get_url(converted_video_path, os.path.basename(converted_video_path))
        logging.info(f"predict_video: Public URL for predicted video: {predicted_media_url}")

        # --- NEW: Extract detection details from results (e.g., number of objects per frame) ---
        # Video results are more complex. YOLO provides a list of Results objects, one per frame.
        # You might want to summarize detections, e.g., total objects detected, or common objects.
        # For simplicity, let's just count total boxes detected across all frames.
        total_frames_processed = len(results)
        total_objects_detected = 0
        for r in results:
            if r.boxes:
                total_objects_detected += len(r.boxes)

        # You could also process 'results' to get per-frame object counts/labels
        # and return a more detailed 'detections_summary'.

        # Return JSON with URL and description
        return jsonify({
            'status': 'success',
            'predicted_video_url': predicted_media_url,
            'description': f'Video processed successfully with YOLOv8. {total_frames_processed} frames analyzed.',
            'total_objects_detected_across_video': total_objects_detected
        }), 200

    except Exception as e:
        logging.error(f"predict_video: Prediction/conversion failed for {original_filename}: {str(e)}", exc_info=True)
        if "MemoryError" in str(e):
            return jsonify({'error': 'Server ran out of memory during video processing. Try a smaller video or more RAM.'}), 507
        elif "No space left on device" in str(e):
            return jsonify({'error': 'Server disk space is full. Please free up space.'}), 507
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
            logging.info(f"predict_video: Removed uploaded video: {video_path}")
        if results_dir and os.path.exists(results_dir):
            try:
                shutil.rmtree(results_dir)
                logging.info(f"predict_video: Removed YOLO results directory: {results_dir}")
            except OSError as e:
                logging.warning(f"predict_video: Error removing results directory {results_dir}: {e}")
        # Again, we are NOT deleting from RESULT_FOLDER here.
        # if predicted_media_server_path and os.path.exists(predicted_media_server_path):
        #     os.remove(predicted_media_server_path)
        #     logging.info(f"predict_video: Removed predicted media from results folder: {predicted_media_server_path}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)