#Evaluate

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the directory for the processed dataset
PROCESSED_DATASET_DIR = "../processed_dataset"
IMG_SIZE = 64  # Image size
BATCH_SIZE = 32

# Load the trained model
model = load_model("../models/sign_language_model.h5")

# Prepare the image data generator for validation/testing
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(PROCESSED_DATASET_DIR,
                                                  target_size=(IMG_SIZE, IMG_SIZE),
                                                  batch_size=BATCH_SIZE,
                                                  class_mode='categorical', shuffle=False)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
