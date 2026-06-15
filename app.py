from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)   # ✅ MUST be before routes

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = tf.keras.models.load_model("model/skin_model.h5")

classes = ['acne', 'eczema', 'melanoma', 'psoriasis']

def predict_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.reshape(img, (1,224,224,3))

    prediction = model.predict(img)
    return classes[np.argmax(prediction)]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    img = cv2.imread(filepath)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.reshape(img, (1,224,224,3))

    prediction = model.predict(img)
    index = np.argmax(prediction)
    result = classes[index]
    confidence = round(float(np.max(prediction)) * 100, 2)

    return render_template("result.html",
                           prediction=result,
                           confidence=confidence,
                           image_path=filepath)

if __name__ == "__main__":
    app.run(debug=True)