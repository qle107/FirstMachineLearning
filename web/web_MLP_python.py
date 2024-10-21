from flask import Flask, render_template, request, jsonify
import os
import tempfile
from src.MLP_Web import MLP, load_and_train_model, predict_single  # Assuming these are your utilities

app = Flask(__name__)

# Placeholder for the MLP model
mlp = None

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/train_model", methods=["POST"])
def train_model():
    global mlp
    mlp = load_and_train_model()  # Assuming this trains and returns your model
    return jsonify({'status': 'Model trained successfully'})

@app.route("/predict", methods=["POST"])
def predict():
    if mlp is None:
        return jsonify({'error': 'Model is not trained yet'}), 400

    if 'image' not in request.files:
        return jsonify({'error': 'No image in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save the uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            file_path = temp_file.name
            file.save(file_path)

        # Predict using the model
        prediction = predict_single(file_path, mlp)

        # Remove the temporary file
        os.remove(file_path)

        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
