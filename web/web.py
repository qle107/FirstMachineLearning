from flask import Flask, render_template, request, jsonify
import os
import tempfile
from src.MLP_RUST import load_and_train_model as load_and_train_mlp, mlp_predict_single  # Assuming these are your utilities
from src.LM_Rust_Web import load_and_train_lm, lm_predict_single

app = Flask(__name__)

# Placeholder for the models
mlp = None
lm = None

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/train_model", methods=["POST"])
def train_model():
    global mlp, lm
    try:
        mlp = load_and_train_mlp()
        lm = load_and_train_lm()
        return jsonify({'status': 'Model trained successfully'})
    except Exception as e:
        return jsonify({'status': f'Model training failed: {str(e)}'}), 500

@app.route("/predict", methods=["POST"])
def predict():
    global mlp, lm
    if mlp is None and lm is None:
        return jsonify({'error': 'Models are not trained yet'}), 400

    if 'image' not in request.files or 'model' not in request.form:
        return jsonify({'error': 'No image or model selected in the request'}), 400

    file = request.files['image']
    selected_model = request.form['model']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            file_path = temp_file.name
            file.save(file_path)

        # Predict using the selected model
        if selected_model == 'mlp':
            if mlp is None:
                return jsonify({'error': 'MLP model is not trained yet'}), 400
            prediction = mlp_predict_single(file_path, mlp)
        elif selected_model == 'lm_rosen':
            if lm is None:
                return jsonify({'error': 'LM Rosen model is not trained yet'}), 400
            prediction = lm_predict_single(file_path, lm)
        else:
            return jsonify({'error': 'Invalid model selected'}), 400

        # Remove the temporary file
        os.remove(file_path)

        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
