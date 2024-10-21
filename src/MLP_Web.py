import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.datasets import mnist
import time
import re
from io import BytesIO
from numba import vectorize, jit, prange, cuda
import base64


@vectorize(['float32(float32)'], target='cuda')
def relu(x):
    return max(0, x)


@vectorize(['float32(float32)'], target='cuda')
def relu_derivative(x):
    return np.float32(1.0) if x > 0 else np.float32(0.0)


@jit(nopython=True, parallel=True, target_backend='cuda')
def softmax(x):
    result = np.empty_like(x, dtype=np.float32)
    for i in prange(x.shape[0]):
        exps = np.exp(x[i] - np.max(x[i]))
        result[i] = exps / np.sum(exps)
    return result


class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size).astype(np.float32) * 0.01
        self.weights_hidden_output = np.random.randn(hidden_size, output_size).astype(np.float32) * 0.01
        self.bias_hidden = np.zeros((1, hidden_size), dtype=np.float32)
        self.bias_output = np.zeros((1, output_size), dtype=np.float32)
        self.learning_rate = np.float32(learning_rate)

    def forward(self, x):
        self.hidden_input = np.dot(x, self.weights_input_hidden).astype(np.float32) + self.bias_hidden
        self.hidden_output = relu(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output).astype(np.float32) + self.bias_output
        self.final_output = softmax(self.final_input)
        return self.final_output

    def backward(self, x, y, output):
        output_error = output - y
        hidden_error = np.dot(output_error, self.weights_hidden_output.T)
        hidden_delta = hidden_error * relu_derivative(self.hidden_input)

        # Gradient clipping
        grad_clip = 1.0
        grad_hidden_output = np.dot(self.hidden_output.T, output_error)
        grad_input_hidden = np.dot(x.T, hidden_delta)

        if np.max(np.abs(grad_hidden_output)) > grad_clip:
            grad_hidden_output = grad_clip * np.sign(grad_hidden_output)
        if np.max(np.abs(grad_input_hidden)) > grad_clip:
            grad_input_hidden = grad_clip * np.sign(grad_input_hidden)

        self.weights_hidden_output -= (self.learning_rate * grad_hidden_output).astype(np.float32)
        self.bias_output -= (self.learning_rate * np.sum(output_error, axis=0, keepdims=True)).astype(np.float32)
        self.weights_input_hidden -= (self.learning_rate * grad_input_hidden).astype(np.float32)
        self.bias_hidden -= (self.learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)).astype(np.float32)

        # L2 regularization
        l2_reg = np.float32(0.01)
        self.weights_hidden_output -= l2_reg * self.weights_hidden_output
        self.weights_input_hidden -= l2_reg * self.weights_input_hidden

    def train(self, x, y, epochs, batch_size):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(x), batch_size):
                batch_x = x[i:i + batch_size]
                batch_y = y[i:i + batch_size]
                output = self.forward(batch_x)
                self.backward(batch_x, batch_y, output)
                batch_loss = -np.sum(batch_y * np.log(output + 1e-8))  # Avoid log(0)
                total_loss += batch_loss
            print(f"Epoch {epoch}, Loss: {total_loss / len(x)}")

    def predict(self, x):
        output = self.forward(x)
        return np.argmax(output, axis=1)


def one_hot_encode(labels, num_classes):
    one_hot = np.zeros((labels.size, num_classes), dtype=np.float32)
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot


def preprocess_image(image_data,target_size =(28,28)):
    image = Image.open(image_data).convert('L')
    resized_image = image.resize(target_size)
    image_array = np.asarray(resized_image).astype(np.float64) / 255.0
    return image_array.flatten()


def load_and_train_model():
    # Load dataset
    data = pd.read_csv('../src/another_image_dataset.csv')

    # Split dataset into features and labels
    X = data.drop(columns=['label']).values
    y = data['label'].values

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert labels to one-hot encoding
    y_train_onehot = np.eye(3)[y_train]
    y_test_onehot = np.eye(3)[y_test]

    # Parameters
    input_size = X_train.shape[1]
    hidden_size = 512  # You can adjust this value
    output_size = 3
    learning_rate = 0.001
    epochs = 15
    batch_size = 128

    mlp = MLP(input_size, hidden_size, output_size, learning_rate)
    start_time = time.time()
    mlp.train(X_train, y_train_onehot, epochs, batch_size)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")

    return mlp


def predict_single(img_path, mlp, target_size=(28, 28)):
    image_array = preprocess_image(img_path, target_size)
    # image_array = image_array.reshape(1, -1)
    y_pred = mlp.predict(image_array)[0]
    str_result = ["Plane", "Car", "Bike"]
    return str_result[y_pred]