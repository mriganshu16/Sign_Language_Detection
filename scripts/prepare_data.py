import os
import cv2
import numpy as np

# Set dataset directory
DATASET_DIR = "../dataset"
PROCESSED_DIR = "../processed_dataset"
IMG_SIZE = 64  # Resize images to 64x64

# Create processed dataset folder if it doesn't exist
if not os.path.exists(PROCESSED_DIR):
    os.makedirs(PROCESSED_DIR)

# Preprocess images
for label in os.listdir(DATASET_DIR):
    label_dir = os.path.join(DATASET_DIR, label)
    processed_label_dir = os.path.join(PROCESSED_DIR, label)

    if not os.path.exists(processed_label_dir):
        os.makedirs(processed_label_dir)

    if os.path.isdir(label_dir):
        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)
            img = cv2.imread(img_path)

            if img is not None:
                # Resize and normalize the image
                img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img_normalized = img_resized / 255.0

                # Save processed image
                processed_img_path = os.path.join(processed_label_dir, img_name)
                cv2.imwrite(processed_img_path, (img_normalized * 255).astype(np.uint8))

print("Preprocessing complete! Processed images are saved in:", PROCESSED_DIR)
