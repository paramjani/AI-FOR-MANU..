from flask import Flask, render_template, request, redirect, send_from_directory
import os
from werkzeug.utils import secure_filename
from detect_video import process_uploaded_image, process_live_camera

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No image uploaded.", 400

    image = request.files['image']
    filename = secure_filename(image.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(path)

    result_path = process_uploaded_image(path)
    return redirect(f"/{result_path}")

@app.route('/live')
def live():
    process_live_camera()
    return "✅ Live detection completed. Check the violations folder."

@app.route('/<path:path>')
def serve_file(path):
    return send_from_directory('.', path)

if __name__ == "__main__":
    app.run(debug=True)
