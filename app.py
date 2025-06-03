from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from ultralytics import YOLO
import os
import uuid
import shutil
import logging
import subprocess # Required for running ffmpeg

app = Flask(__name__)
CORS(app)

# Configure logging for better insights
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configure Max Request Body Size for large files ---
# Default is often 16MB. Set this to a value that accommodates your largest expected video file.
# 200 * 1024 * 1024 bytes = 200 MB. Adjust as needed.
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024 # 200 MB

# Load YOLO model
try:
    # Ensure 'yolov8/best.pt' path is correct relative to where your Flask app runs,
    # or provide an absolute path.
    model = YOLO("yolov8/best.pt")
    logging.info("YOLOv8 model loaded successfully from yolov8/best.pt.")
except Exception as e:
    logging.error(f"Model Load Error: Could not load yolov8/best.pt - {e}")
    model = None

# Create upload folder if it doesn't exist
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
logging.info(f"Upload folder created/verified at: {UPLOAD_FOLDER}")

@app.route('/predict_image', methods=['POST'])
def predict_image():
    """
    Endpoint for image prediction.
    Receives an image, runs YOLO inference, and returns the predicted image.
    """
    if model is None:
        logging.error("predict_image: Model not loaded. Returning 503.")
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 503

    if 'image' not in request.files:
        logging.warning("predict_image: No 'image' file part in request.")
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    # Validate image file format
    allowed_image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
    if not image.filename or not image.filename.lower().endswith(allowed_image_extensions):
        logging.warning(f"predict_image: Invalid image format for {image.filename}. Supported: {allowed_image_extensions}")
        return jsonify({'error': f'Invalid image format. Supported formats: {", ".join(allowed_image_extensions)}'}), 400

    # Generate a unique filename to prevent overwrites and security issues
    filename = f"{uuid.uuid4()}_{image.filename}"
    image_path = os.path.join(UPLOAD_FOLDER, filename)

    results_dir = None # Initialize to None for cleanup in finally block
    try:
        # Save the uploaded image temporarily
        image.save(image_path)
        logging.info(f"predict_image: Image saved to: {image_path}")

        # Run YOLO prediction
        # `save=True` will save the predicted image to a 'runs' directory
        results = model.predict(source=image_path, save=True, conf=0.3,
                                project='runs', name='detect_results', exist_ok=True)
        
        # The save directory created by YOLO (e.g., runs/detect_results/expX)
        results_dir = results[0].save_dir
        # The predicted image file path within the results directory
        # YOLO typically saves with the same basename as the input.
        result_file_path = os.path.join(results_dir, os.path.basename(image_path))

        if not os.path.exists(result_file_path):
            logging.error(f"predict_image: Prediction output file not found at expected path: {result_file_path}")
            # Fallback: Check if any image file exists in the results_dir
            found_output_files = [f for f in os.listdir(results_dir) if f.lower().endswith(allowed_image_extensions)]
            if found_output_files:
                result_file_path = os.path.join(results_dir, found_output_files[0])
                logging.info(f"predict_image: Found output image at fallback path: {result_file_path}")
            else:
                return jsonify({'error': 'Prediction output file not found. Internal server error.'}), 500

        logging.info(f"predict_image: Sending predicted image: {result_file_path}")
        # Send the predicted image back to the client
        return send_file(result_file_path, mimetype='image/jpeg')

    except Exception as e:
        logging.error(f"predict_image: Prediction failed for {image.filename}: {str(e)}", exc_info=True)
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    finally:
        # Clean up temporary files and directories
        if os.path.exists(image_path):
            os.remove(image_path)
            logging.info(f"predict_image: Removed uploaded image: {image_path}")
        if results_dir and os.path.exists(results_dir):
            try:
                shutil.rmtree(results_dir)
                logging.info(f"predict_image: Removed results directory: {results_dir}")
            except OSError as e:
                logging.warning(f"predict_image: Error removing results directory {results_dir}: {e}")

@app.route('/predict_video', methods=['POST'])
def predict_video():
    """
    Endpoint for video prediction.
    Receives a video, runs YOLO inference, converts the output to MP4, and returns the MP4 video.
    """
    if model is None:
        logging.error("predict_video: Model not loaded. Returning 503.")
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 503

    if 'video' not in request.files:
        logging.warning("predict_video: No 'video' file part in request.")
        return jsonify({'error': 'No video uploaded'}), 400

    video = request.files['video']
    # Validate video file format
    allowed_video_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
    if not video.filename or not video.filename.lower().endswith(allowed_video_extensions):
        logging.warning(f"predict_video: Invalid video format for {video.filename}. Supported: {allowed_video_extensions}")
        return jsonify({'error': f'Invalid video format. Supported formats: {", ".join(allowed_video_extensions)}'}), 400

    # Generate a unique filename for the uploaded video
    original_filename = video.filename
    filename = f"{uuid.uuid4()}_{original_filename}"
    video_path = os.path.join(UPLOAD_FOLDER, filename)

    results_dir = None # Initialize for cleanup
    converted_video_path = None # Initialize for cleanup
    try:
        # Save the uploaded video temporarily
        video.save(video_path)
        logging.info(f"predict_video: Video saved to: {video_path}")

        # Run YOLO prediction on the video
        # YOLOv8 often outputs .avi files for video by default with `save=True`
        logging.info(f"predict_video: Starting YOLO prediction for video: {video_path}")
        results = model.predict(source=video_path, save=True, conf=0.3,
                                project='runs', name='video_results', exist_ok=True)
        
        results_dir = results[0].save_dir # Directory where YOLO saved its output

        # Find the .avi file (or whatever YOLO output) from the prediction
        # We explicitly look for .avi because that's what YOLO often generates
        predicted_output_files = [f for f in os.listdir(results_dir) if f.lower().endswith('.avi')]
        
        if not predicted_output_files:
            logging.error(f"predict_video: No .avi video file found from YOLO prediction in: {results_dir}")
            return jsonify({'error': 'YOLO prediction did not produce an AVI file. Check YOLO output settings.'}), 500

        yolo_output_avi_path = os.path.join(results_dir, predicted_output_files[0])
        logging.info(f"predict_video: YOLO output AVI found: {yolo_output_avi_path}")

        # --- Convert the YOLO output (AVI) to MP4 using FFmpeg ---
        # Generate a new filename for the MP4 output
        converted_video_filename = f"{os.path.splitext(predicted_output_files[0])[0]}.mp4"
        converted_video_path = os.path.join(results_dir, converted_video_filename)

        logging.info(f"predict_video: Converting {yolo_output_avi_path} to {converted_video_path} (MP4) using FFmpeg.")
        
        # FFmpeg command for conversion
        # -i: input file
        # -c:v libx264: video codec (H.264) - highly compatible
        # -preset fast: encoding speed vs. compression efficiency tradeoff
        # -crf 23: Constant Rate Factor (quality setting, 0-51, lower is higher quality)
        # -c:a aac: audio codec (AAC) - highly compatible
        # -b:a 128k: audio bitrate
        # -movflags faststart: optimizes MP4 for web streaming by moving metadata to start
        # -y: overwrite output file without asking
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
            # Run the ffmpeg command. `check=True` raises an error if ffmpeg fails.
            # `capture_output=True` captures stdout/stderr, `text=True` decodes it.
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

        logging.info(f"predict_video: Sending converted MP4 video: {converted_video_path} with mimetype video/mp4")
        # Send the converted MP4 video back to the client
        return send_file(converted_video_path, mimetype='video/mp4')

    except Exception as e:
        # Catch any other exceptions that might occur before or after FFmpeg call
        logging.error(f"predict_video: Prediction/conversion failed for {original_filename}: {str(e)}", exc_info=True)
        if "MemoryError" in str(e):
             return jsonify({'error': 'Server ran out of memory during video processing. Try a smaller video or more RAM.'}), 507
        elif "No space left on device" in str(e):
             return jsonify({'error': 'Server disk space is full. Please free up space.'}), 507
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    finally:
        # Clean up temporary uploaded video and the entire YOLO results directory
        if os.path.exists(video_path):
            os.remove(video_path)
            logging.info(f"predict_video: Removed uploaded video: {video_path}")
        if results_dir and os.path.exists(results_dir):
            try:
                shutil.rmtree(results_dir)
                logging.info(f"predict_video: Removed YOLO results directory: {results_dir}")
            except OSError as e:
                logging.warning(f"predict_video: Error removing results directory {results_dir}: {e}")

if __name__ == '__main__':
    # Run the Flask app. For production, use a WSGI server like Gunicorn.
    # debug=True is good for development, but set to False in production.
    app.run(host='0.0.0.0', port=5000)