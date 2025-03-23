import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained CNN model
model = load_model('app/cnn_image_classification.h5')


# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')


# Route to handle image classification
@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image was sent with the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    # If no file is selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read the image file
        img = Image.open(file.stream)

        # Preprocess the image to match the model input (28x28 grayscale MNIST images)
        img = img.convert('L')  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to 28x28 pixels

        # Convert the image to numpy array and normalize it
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]

        # Reshape the image to match the model input shape (1, 28, 28, 1)
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension (1 for grayscale)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1 image in batch)

        # Predict the digit
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction, axis=1)[0]  # Get the class with the highest probability

        return jsonify({'prediction': int(predicted_digit)}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
