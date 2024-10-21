import numpy as np
from ctypes import *

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PIL import Image


class MLP(Structure):
    _fields_ = [
        ("weight_input_hidden", POINTER(c_double)),
        ("weight_output_hidden", POINTER(c_double)),
        ("bias_hidden", POINTER(c_double)),
        ("bias_output", POINTER(c_double)),
        ("learning_rate", c_double),
        ("hidden_input", POINTER(c_double)),
        ("hidden_output", POINTER(c_double)),
        ("final_input", POINTER(c_double)),
        ("final_output", POINTER(c_double)),
        ("input_size", c_size_t),
        ("hidden_size", c_size_t),
        ("output_size", c_size_t),
    ]


lib = CDLL('../dll/yetnewmlp.dll')

lib.mlp_new.restype = POINTER(MLP)
lib.mlp_new.argtypes = [c_size_t, c_size_t, c_size_t, c_double]
lib.mlp_forward.restype = POINTER(c_double)
lib.mlp_forward.argtypes = [POINTER(MLP), POINTER(c_double), c_size_t, c_size_t]
lib.mlp_train.restype = None
lib.mlp_train.argtypes = [POINTER(MLP), POINTER(c_double), POINTER(c_double), c_size_t, c_size_t, c_size_t, c_size_t,
                          c_size_t, c_size_t]
lib.mlp_free.restype = None
lib.mlp_free.argtypes = [POINTER(MLP)]


def numpy_to_ctypes(array):
    return array.ctypes.data_as(POINTER(c_double))


def ctypes_to_numpy(ptr, shape):
    size = np.prod(shape)
    return np.ctypeslib.as_array(ptr, shape=(size,)).reshape(shape)


def one_hot_encode(labels, num_classes=10):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot


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
    epochs = 25
    batch_size = 40

    # Create an MLP
    mlp = lib.mlp_new(input_size, hidden_size, output_size, c_double(learning_rate))

    # Prepare input data
    X_train_ptr = numpy_to_ctypes(X_train)
    y_train_ptr = numpy_to_ctypes(y_train_onehot)

    # Train the MLP
    lib.mlp_train(mlp, X_train_ptr, y_train_ptr, X_train.shape[0], X_train.shape[1], y_train_onehot.shape[0],
                  y_train_onehot.shape[1], epochs, batch_size)

    return mlp


def preprocess_image(img_path, target_size=(28, 28)):
    image = Image.open(img_path).convert('L')
    resized_image = image.resize(target_size)
    image_array = np.asarray(resized_image).astype(np.float64) / 255.0
    return image_array.flatten()


def mlp_predict_single(img_path, mlp, target_size=(28, 28)):
    image_array = preprocess_image(img_path, target_size)
    single_test_ptr = numpy_to_ctypes(image_array)
    single_output_ptr = lib.mlp_forward(mlp, single_test_ptr, 1, image_array.shape[0])
    single_output = ctypes_to_numpy(single_output_ptr, (1, 10))
    y_pred = np.argmax(single_output, axis=1)[0]
    str_result = ["plane", "car", "bike"]
    return str(str_result[y_pred])
