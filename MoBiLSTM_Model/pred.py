import cv2
import numpy as np
from collections import deque
import tensorflow.keras.models as models




# Specify the height and width to which each video frame will be resized in our dataset.
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64

# Specify the number of frames of a video that will be fed to the model as one sequence.
SEQUENCE_LENGTH = 16

# Load the pre-trained model
#   # Your model
model = models.load_model('D:\Tutorial 7\Streaming and SaveFrames\my_h5_model.h5')
# Create a function to preprocess frames
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
    normalized_frame = resized_frame / 255
    return normalized_frame

# Function to predict violence in real-time
def predict_real_time(model):
    # Start the video stream
    video_stream = cv2.VideoCapture(0)  # Use the appropriate video source (0 for webcam)

    # Declare a queue to store video frames
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)

    while True:
        # Read a frame from the video stream
        ret, frame = video_stream.read()

        if not ret:
            break

        # Preprocess the frame
        normalized_frame = preprocess_frame(frame)

        # Append the preprocessed frame to the frames queue
        frames_queue.append(normalized_frame)

        # Perform prediction when we have enough frames in the queue
        if len(frames_queue) == SEQUENCE_LENGTH:
            # Convert the frames queue to a numpy array
            frames_array = np.array(frames_queue)

            # Reshape the frames array to match the model's input shape
            input_frames = np.expand_dims(frames_array, axis=0)

            # Perform prediction
            prediction = model.predict(input_frames)

            # Get the predicted class index
            predicted_class_index = np.argmax(prediction)
            CLASSES_LIST=['Violence','Non-Violence']
            # Get the predicted class name
            predicted_class_name = CLASSES_LIST[predicted_class_index]

            # Display the predicted class on the frame
            cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow('Video Stream', frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video stream and close the window
    video_stream.release()
    cv2.destroyAllWindows()

# Call the function to start real-time prediction
predict_real_time(model)
