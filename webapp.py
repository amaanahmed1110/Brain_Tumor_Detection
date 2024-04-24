from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load your pre-trained Keras model
model = load_model('BrainTumor10EpochsCategorical.h5')

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for predicting brain tumor
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file
    file = request.files['file']

    # Check if file is uploaded
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    # Read the image file
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Preprocess the image
    img = preprocess_image(img)

    # Make predictions
    predictions = model.predict(np.expand_dims(img, axis=0))

    # Get the predicted class
    predicted_class = np.argmax(predictions)

    # Determine result
    result = "Brain tumor not detected" if predicted_class == 0 else "Brain tumor detected"

    # Return the result
    return jsonify({'result': result})

# Define a function to preprocess the image
def preprocess_image(img):
    # Resize the image to match the input shape expected by your model
    img = cv2.resize(img, (64, 64))
    # Perform any additional preprocessing required (e.g., normalization)
    return img

if __name__ == '__main__':
    app.run(debug=True)
