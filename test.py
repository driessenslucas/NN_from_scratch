import cv2
import numpy as np
import pickle


from neuralnet import NeuralNetMLP, int_to_onehot

model = NeuralNetMLP(num_features=28*28, num_hidden=50, num_classes=10)


import cv2
import numpy as np

# Load model weights
with open('model_weights.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Set the weights and biases
model.weight_hidden = model_data['weight_hidden']
model.bias_hidden = model_data['bias_hidden']
model.weight_output = model_data['weight_output']
model.bias_output = model_data['bias_output']

def preprocess_digit(roi):
    """
    Preprocess the region of interest (ROI) to match model input format.
    Resizes to 28x28, flattens, and normalizes.
    """
    resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    normalized = ((resized / 255.0) - 0.5) * 2
    flattened = normalized.flatten().reshape(1, -1)
    return flattened

# Open the webcam
cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply binary thresholding to create a binary image
    _, binary = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Compute bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Ignore small contours that are unlikely to be digits
        if w > 10 and h > 10:
            # Extract the region of interest (ROI) and preprocess it
            roi = gray[y:y+h, x:x+w]
            processed_roi = preprocess_digit(roi)

            # Predict the digit using the model
            _, probas = model.forward(processed_roi)
            prediction = np.argmax(probas, axis=1)[0]

            # Draw bounding box and prediction on the original frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, str(prediction), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Show the webcam feed with bounding boxes and predictions
    cv2.imshow('Webcam - Digit Recognition', frame)

    # Quit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
