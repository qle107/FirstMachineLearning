import ctypes
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load the shared library
class LinearModel(ctypes.Structure):
    _fields_ = [("nb_class", ctypes.c_size_t),
                ("weights", ctypes.POINTER(ctypes.c_double))]


# Load the shared library into ctypes
lib = ctypes.CDLL('../dll/linear_classifier.dll')

# Define the argument and return types
lib.linear_model_new.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
lib.linear_model_new.restype = ctypes.POINTER(LinearModel)

lib.linear_model_free.argtypes = [ctypes.POINTER(LinearModel)]

lib.linear_model_predict.argtypes = [ctypes.POINTER(LinearModel), ctypes.POINTER(ctypes.c_double), ctypes.c_int]
lib.linear_model_predict.restype = ctypes.POINTER(ctypes.c_double)

lib.linear_model_train.argtypes = [ctypes.POINTER(LinearModel), ctypes.POINTER(ctypes.c_double),
                                   ctypes.POINTER(ctypes.c_double), ctypes.c_int64, ctypes.c_int64, ctypes.c_double,
                                   ctypes.c_int]


# Helper function to predict
def predict(model, input_data, shape):
    input_array = np.array(input_data, dtype=np.float64)
    prediction_ptr = lib.linear_model_predict(model, input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                              input_array.size)
    pred_array = np.ctypeslib.as_array(prediction_ptr, shape=(shape,))
    return pred_array


# Load the dataset
dataset = pd.read_csv('./another_image_dataset.csv')

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
    begin = time.time()
    # Train the model
    lib.linear_model_train(model, X_train.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                           Y_train.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), X_train.shape[0], X_train.shape[1],
                           0.01, 1000)
    print(time.time()-begin)
    return model


new_model = load_and_train_lm()
correct = 0
for i in range(X_test.shape[0]):
    result = predict(new_model, X_test[i],3 )
    answer = np.argmax(result)
    if answer == np.argmax(Y_test[i]):
        correct += 1
print(correct / X_test.shape[0]*100)
