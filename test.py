import cv2
import numpy as np
import tensorflow as tf
from google.colab.patches import cv2_imshow

# Define the model architecture
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(50, 50, 3)),  # Use Input layer explicitly
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')  # 2 classes: cat, dog
    ])
    return model

# Load weights into the model
def load_model_with_weights(weights_path):
    model = create_model()  # Create the model with the architecture
    model.load_weights(weights_path)  # Load the saved weights
    return model

# Function to preprocess the frame from the video
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (50, 50))  # Resize to 50x50
    frame_normalized = frame_resized / 255.0     # Normalize the pixel values
    return frame_normalized

# Predict the class for a single frame
def predict_frame(model, frame):
    preprocessed_frame = preprocess_frame(frame)
    preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)  # Add batch dimension
    prediction = model.predict(preprocessed_frame)
    predicted_class = np.argmax(prediction, axis=1)[0]  # 0 for cat, 1 for dog
    return predicted_class

# Draw a bounding box and label on the frame
def draw_bounding_box(frame, label):
    height, width = frame.shape[:2]
    # Draw a rectangle around the entire frame (adjust as needed)
    cv2.rectangle(frame, (10, 10), (width - 10, height - 10), (0, 255, 0), 2)
    cv2.putText(frame, f'Detected: {label}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# Process the video, frame by frame
def process_video(video_path, model):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        # Predict the class (cat or dog) for the current frame
        predicted_class = predict_frame(model, frame)
        predicted_label = 'Cat' if predicted_class == 0 else 'Dog'

        # Draw bounding box and annotation on the frame
        draw_bounding_box(frame, predicted_label)

        # Instead of cv2.imshow(), use cv2_imshow for Colab
        cv2_imshow(frame)  # Display the frame

        # Press 'q' to quit (for local environments, this can be activated)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()

# Load the model with saved weights
weights_path = 'final.weights.h5'  # Update with the correct weight file
model = load_model_with_weights(weights_path)

# Path to the video file
video_path = 'test1.mp4'  # Replace with the actual video path

# Process the video and make predictions for each frame
process_video(video_path, model)
