import os
from PIL import Image
import numpy as np
import pandas as pd

# Paths to the folders
image_folder = './DATA/train/'
folders = ['plane', 'car', 'bike']

# Image size
image_size = (28, 28)

# Init lists to hold image data and labels
data = []
labels = []

# Loop through each folder and process the images
for label, folder in enumerate(folders):
    folder_path = os.path.join(image_folder, folder)
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
            image_path = os.path.join(folder_path, filename)
            with Image.open(image_path) as img:
                # Resize the image
                img_resized = img.convert('L').resize(image_size)
                # Convert image to numpy array and flatten it
                img_array = np.array(img_resized).flatten()
                # Append the flattened image and label to the lists
                data.append(img_array)
                labels.append(label)

data_array = np.array(data)
labels_array = np.array(labels)

# Create a DataFrame
dataset = pd.DataFrame(data_array)
dataset['label'] = labels_array

# Save the dataset to a CSV file
file_name = '128_dataset.csv'
dataset.to_csv(file_name, index=False)
print(f"Dataset created and saved as {file_name}")
