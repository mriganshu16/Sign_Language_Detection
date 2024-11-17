import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the pre-trained model
model = load_model('sign_language_model.h5')

# Labels for 0-9 and a-z (adjust if needed)
labels = [str(i) for i in range(10)] + [chr(i) for i in range(ord('a'), ord('z') + 1)]

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Image size (adjust according to your model's input size)
IMG_SIZE = 64

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Convert to grayscale (this is just for processing, not for the model input)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the image to match the model input
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

    # Convert grayscale image to 3 channels (RGB) by duplicating the grayscale data
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

    # Expand dimensions to match the model input shape (batch size, height, width, channels)
    img_array = np.array(resized_rgb)
    img_array = np.expand_dims(img_array, axis=0)  # Batch size of 1

    # Normalize the image
    img_array = img_array / 255.0

    # Predict the class
    prediction = model.predict(img_array)
    predicted_class = labels[np.argmax(prediction)]

    # Display the predicted class
    cv2.putText(frame, f"Prediction: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Sign Language Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
