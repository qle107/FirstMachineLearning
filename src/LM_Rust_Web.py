import ctypes
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the shared library
lib = ctypes.CDLL('../dll/linear_classifier.dll')

# Define argument and return types
lib.linear_model_new.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
lib.linear_model_new.restype = ctypes.POINTER(ctypes.c_void_p)

lib.linear_model_free.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
lib.linear_model_free.restype = None

lib.linear_model_predict.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_double), ctypes.c_int]
lib.linear_model_predict.restype = ctypes.POINTER(ctypes.c_double)

lib.linear_model_train.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_double),
                                   ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_double,
                                   ctypes.c_int]
lib.linear_model_train.restype = None


# Helper function to predict
def predict(model, input_data, shape):
    input_array = np.array(input_data, dtype=np.float64)
    prediction_ptr = lib.linear_model_predict(model, input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                              input_array.size)
    pred_array = np.ctypeslib.as_array(prediction_ptr, shape=(shape,))
    return pred_array


# Load the dataset
dataset = pd.read_csv('../src/another_image_dataset.csv')

# Assuming the last column is the label and the rest are features
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Convert labels to one-hot encoding
encoder = OneHotEncoder(sparse_output=False)
Y_one_hot = encoder.fit_transform(Y.reshape(-1, 1))

# Split the dataset into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_one_hot, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def load_and_train_lm():
    # Create a new model
    model = lib.linear_model_new(Y_train.shape[1], X_train.shape[1])
    print("training")
    # Train the model
    lib.linear_model_train(model, X_train.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                           Y_train.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), X_train.shape[0], X_train.shape[1],
                           0.01, 1000)
    print("Finished")
    return model


def preprocess_image(img_path, target_size=(28, 28)):
    image = Image.open(img_path).convert('L')
    resized_image = image.resize(target_size)
    image_array = np.asarray(resized_image).astype(np.float64) / 255.0
    return image_array.flatten()


def lm_predict_single(img_path, model, dimension=3, target_size=(28, 28)):
    image_array = preprocess_image(img_path, target_size)
    y_pred = predict(model, image_array, dimension)
    str_result = ["plane", "car", "bike"]
    pred = int(np.argmax(y_pred))
    return str(str_result[pred])
