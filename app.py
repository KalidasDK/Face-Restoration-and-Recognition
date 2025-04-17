import os
import shutil
import subprocess
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

app = Flask(__name__)

# Configuration
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'GFPGAN', 'inputs', 'upload')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'GFPGAN', 'results', 'restored_imgs')
KNOWN_PEOPLE = os.path.join(BASE_DIR, 'known_people')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(KNOWN_PEOPLE, exist_ok=True)

def clear_folder(folder):
    """Clear all contents of a directory"""
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Error clearing {file_path}: {e}')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Clear previous files
    clear_folder(UPLOAD_FOLDER)
    clear_folder(OUTPUT_FOLDER)

    # Handle file upload
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    
    filename = file.filename
    file.save(os.path.join(UPLOAD_FOLDER, filename))

    # Process image
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    command = [
        'python3', 'face_recognizer.py',
        '--image', input_path
        #'--known', KNOWN_PEOPLE,
        #'--tolerance', '0.52',
        #'--cpus', '-1'
    ]
    # Process image and capture output
    result = subprocess.run(command, capture_output=True, text=True)
    output = result.stdout

    # Extract person's name from output
    identified_names = []
    for line in output.split('\n'):
        if "Person identified:" in line:
            name = line.split(":")[1].strip()
            identified_names.append(name)
    if not identified_names:
        identified_names = ["Unknown"]

    return redirect(url_for('show_results', name=','.join(identified_names)))


@app.route('/results/<name>')
def show_results(name):
    try:
        print(name)
        input_img = os.listdir(UPLOAD_FOLDER)[0]
        output_img = os.listdir(OUTPUT_FOLDER)[0]
    except IndexError:
        return redirect(url_for('index'))
    
    # Split names back into list
    names_list = name.split(',')
    
    return render_template('results.html',
                         input_img=input_img,
                         output_img=output_img,
                         person_names=names_list)

@app.route('/uploads/<filename>')
def send_uploaded(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/result_images/<filename>')
def send_result(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)