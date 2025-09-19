from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image

# Load the trained model
model = load_model("mnist_cnn.h5")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file:
        # Preprocess the uploaded image
        img = Image.open(file).convert("L").resize((28, 28))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # Predict
        prediction = np.argmax(model.predict(img_array))
        return f"Predicted Digit: {prediction}"

    return "No file uploaded!"

if __name__ == '__main__':
    app.run(debug=True)
