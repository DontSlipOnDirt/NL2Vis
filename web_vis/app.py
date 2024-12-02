from flask import Flask, request, jsonify, send_file, render_template, Response
from werkzeug.utils import secure_filename
import os
from core.chat_manager import ChatManager, ChartExecutionResult
import tempfile
import json
import time

app = Flask(__name__)

# 配置上传文件夹
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    uploaded_files = request.files.getlist("files[]")
    file_paths = []
    for file in uploaded_files:
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file_paths.append(file_path)
    return jsonify({'message': 'Files uploaded successfully', 'file_paths': file_paths})

def generate_progress():
    yield "data: " + json.dumps({"status": "Starting process", "progress": 0}) + "\n\n"
    time.sleep(1.5)
    yield "data: " + json.dumps({"status": "Reading data", "progress": 25}) + "\n\n"
    time.sleep(1.5)
    yield "data: " + json.dumps({"status": "Generating visualization", "progress": 50}) + "\n\n"
    time.sleep(1.5)
    yield "data: " + json.dumps({"status": "Rendering", "progress": 75}) + "\n\n"
    time.sleep(1.5)


@app.route('/progress')
def progress():
    return Response(generate_progress(), content_type='text/event-stream')


@app.route('/generate', methods=['POST'])
def generate_visualization():
    data = request.json
    query = data.get('query')
    file_paths = data.get('file_paths')

    if not query or not file_paths:
        return jsonify({'error': 'Missing query or file paths'}), 400

    chat_manager = ChatManager(csv_files=file_paths, log_path="./logs.txt")
    code = chat_manager.start(query)
    result = chat_manager.execute_to_svg(code)

    if result.status:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.svg', mode='w+') as temp_file:
            temp_file.write(result.svg_string)
            temp_file_path = temp_file.name
        return jsonify({'success': True, 'svg_path': temp_file_path})
    else:
        return jsonify({'success': False, 'error': result.error_msg})

@app.route('/download/<path:filename>', methods=['GET'])
def download_file(filename):
    return send_file(filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
