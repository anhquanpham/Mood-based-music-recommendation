# Mood based music recommendation

## Introduction

This project uses Deep Learning and Computer Vision techniques to recognize and classify emotions in real-time video and make music recommendations based on the detected mood. The system consists of four main components: data collection, data training, inference, and a Streamlit-based user interface.

## Technical Overview

The system uses the Mediapipe holistic solution to detect and track landmarks on the face and hands in video frames. The landmark data is normalized by subtracting the x and y coordinates of a reference landmark from the x and y coordinates of each landmark. This has the effect of centering the landmarks relative to the reference landmark, which can improve the performance of the deep learning model by making it invariant to the position of the face and hands in the video frame. 

This solution uses computer vision techniques such as object detection and tracking to locate and track the landmarks in real-time.

The normalized landmark data is used to train a deep learning model to recognize and classify emotions. The model is implemented using the Keras library and consists of an input layer with a number of units equal to the number of features in the data (i.e., the number of landmarks), two hidden layers with 512 and 256 units respectively and ReLU activation functions, and an output layer with a number of units equal to the number of emotion categories and a softmax activation function. The model is trained using the RMSprop optimizer and categorical crossentropy loss function.

Once trained, the model can be used in real-time to make predictions on new video data. The `inference.py` script uses the OpenCV library to capture video from a webcam and process it with the Mediapipe holistic solution to extract landmark data. This data is then normalized and fed into the trained model to make a prediction on the mood being displayed.

This system has potential applications in areas such as music therapy, where music recommendations based on a person's emotional state could be used to improve their mood or reduce stress. The system could also be developed further by adding support for additional emotions or integrating with other music streaming services.

## How to Install

1. Clone the repository to your local machine.
2. Install the required dependencies by running `pip install -r requirements.txt` in the project directory.
3. Run the `data_collection.py` script to collect data for training the model.
4. Run the `data_training.py` script to train the model on the collected data.
5. Run the `main.py` script to start the Streamlit-based user interface.

## Usage

1. Open a terminal window and navigate to the project directory.
2. Start the Streamlit-based user interface by running `streamlit run main.py`.
3. Open your web browser and navigate to `http://localhost:8501` to access the user interface.
4. Enter your preferred language and artists in the text input fields. Press Enter after each field.
5. Once both fields are filled, there will be a camera video to detect your mood.
6. Once the mood is detected, click the "Song recommendation" button to open YouTube search results for songs in your specified language and by your specified singer that match the detected mood.
Detailed demonstration video: https://youtu.be/VKoV5dypCJY 
