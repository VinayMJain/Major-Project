from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from mesonet import load_meso_model, preprocess_image_for_meso

app = Flask(__name__)

UPLOAD_FOLDER = 'C:/Users/vinay/OneDrive/Desktop/Major/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

meso_model = load_meso_model('C:/Users/vinay/OneDrive/Desktop/Major/weights/Meso4_DF.weights.h5')

@app.route('/')
def upload_form():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
   
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    image = cv2.imread(filepath)

    if image is None:
        return jsonify({'error': 'Failed to read the image. Please upload a valid image file.'})

    processed_image = preprocess_image_for_meso(filepath)

    result = meso_model.predict(processed_image)
    
    label = 'Fake' if result >= .468 else 'Real'
    print(result)
    print(label)
    return jsonify({'result': label})

if __name__ == "__main__":
    app.run(debug=True)
