# Vehicle Detection Web Application

This project is a web application that allows users to upload an image of a vehicle and get a classification prediction. The model is implemented using a Multi-Layer Perceptron (MLP) in Rust, and the application is built with Flask.


## Installation
1. **Install the required Python packages**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Flask application**:
    ```sh
    python \web\web.py
    ```

2. **Open your web browser**:
    Navigate to `http://127.0.0.1:8080` to access the web application.

3. **Upload an image**:
    - Click on "Upload an Image" and select an image file (JPEG or PNG).
    - Click the "Predict" button to get the classification result.
    - The uploaded image and its classification result will be displayed on the page.

## Project Structure

- `src\`: Directory to store the source code of MLP in Rust and MLP implemented in python for this dataset.
- `dll\`: Directory to store the compiled Rust shared library.
- `web\web.py`: The Flask web server that handles image upload and prediction using .dll.
- `web\web_MLP_python.py`: The Flask web server for MLP in python.
- `src\MLP_RUST.py`: The Python script that interfaces with the Rust MLP model for training and prediction.
- `web\templates\index.html`: The HTML file for the web application's front-end.