from flask import Flask, request, jsonify, render_template
import base64
import numpy as np
from PIL import Image
import io
import tensorflow as tf

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('MNIST.h5')

def preprocess_image(image):
    # Resize to 28x28 pixels
    image = image.resize((28, 28))
    # Convert to grayscale
    image = image.convert('L')
    # Convert to numpy array
    image = np.array(image)
    # Normalize the image
    image = image / 255.0
    # Reshape to add batch dimension
    image = image.reshape(1, 28, 28, 1)
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Decode the base64 image
    image_data = base64.b64decode(data['image'].split(',')[1])
    image = Image.open(io.BytesIO(image_data))
    
    # Preprocess the image
    image = preprocess_image(image)
    
    # Debug: print the preprocessed image
    print("Preprocessed image shape:", image.shape)
    print("Preprocessed image array:", image)
    
    # Make prediction
    prediction = model.predict(image)
    
    # Debug: print the prediction array
    print("Prediction array:", prediction)
    
    # Get the predicted digit
    digit = np.argmax(prediction[0])
    
    return jsonify({'digit': int(digit)})

if __name__ == '__main__':
    app.run(debug=True)
