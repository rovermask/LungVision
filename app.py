from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Set upload folder path inside 'static'
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model
# model = load_model('saved_models/TB_Detection_Model_CNN.keras')

# Insight images (also inside static)
insight_images = [
    'insights/accuracy_curve.png',
    'insights/loss_curve.png',
    'insights/dataset_split.png',
    'insights/metrics_scores.png',
    'insights/confusion_matrix.png',
    'insights/dataset_distribution.png',
]

# Class labels
categories = ["Normal", "Tuberculosis"]

# Preprocess function
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")           
    img = img.resize((224, 224))                       
    img_array = np.array(img, dtype=np.float32)         
    img_array = img_array / 255.0                       
    img_array = np.expand_dims(img_array, axis=0)       
    return img_array


def load_tb_model():
    return load_model('saved_models/TB_Detection_Model_CNN.keras')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    result = None
    confidence = None
    image_path = None

    if request.method == 'POST':
        file = request.files['xray']
        if file and file.filename != '':
            filename = file.filename
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            img = preprocess_image(save_path)
            model =load_tb_model()
            pred = model.predict(img)[0][0]

            if pred >= 0.5:
                result = "Tuberculosis"
                confidence = round(pred * 100, 2)
            else:
                result = "Normal"
                confidence = round((1 - pred) * 100, 2)

            prediction = f"{result} ({confidence:.2f}% confidence)"
            image_path = url_for('static', filename=f'uploads/{filename}')

    return render_template(
        'index.html',
        prediction=prediction,
        result=result,
        confidence=confidence,
        image_path=image_path,
        insight_images=insight_images
    )

if __name__ == '__main__':
    app.run(debug=True)
