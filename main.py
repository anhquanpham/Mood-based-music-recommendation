import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser

# Load trained model and label list from files
model = load_model("model.h5")
label = np.load("labels.npy")

# Initialize holistic and hands solutions
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()

# Initialize drawing utilities
drawing = mp.solutions.drawing_utils

# Create Streamlit header
st.header("Music Recommendation Based On Mood")

# Initialize run state variable if it doesn't exist
if "run" not in st.session_state:
    st.session_state["run"] = "true"

# Try to load mood from file, otherwise set it to empty string
try:
    mood = np.load("mood.npy")[0]
except:
    mood = ""

# Set run state variable based on whether mood is empty or not
if not mood:
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"


# Define MoodProcessor class for processing video frames with WebRTCStreamer
class MoodProcessor:
    def recv(self, frame):
        # Convert frame to numpy array with BGR color format
        frm = frame.to_ndarray(format="bgr24")

        # Flip frame to eliminate mirror effect
        frm = cv2.flip(frm, 1)

        # Process frame with holistic solution
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        # Initialize list to store landmark data
        lst = []

        # If face landmarks are detected
        if res.face_landmarks:
            # Append normalized x and y coordinates of each landmark to list
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

        # If left hand landmarks are detected
        if res.left_hand_landmarks:
            # Append normalized x and y coordinates of each landmark to list
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            # If no left hand landmarks are detected, append 0.0 to list for each landmark
            for i in range(42):
                lst.append(0.0)

        # If right hand landmarks are detected
        if res.right_hand_landmarks:
            # Append normalized x and y coordinates of each landmark to list
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            # If no right hand landmarks are detected, append 0.0 to list for each landmark
            for i in range(42):
                lst.append(0.0)

        # Convert list of landmark data to numpy array and reshape it to have shape (1, -1)
        lst = np.array(lst).reshape(1,-1)

        # Use trained model to make prediction on landmark data and get corresponding label from label list
        prediction = label[np.argmax(model.predict(lst))]

        # Print predicted label and display it on frame
        print(prediction)
        cv2.putText(frm, prediction, (50,50),cv2.FONT_ITALIC, 1, (255,0,0),2)

        # Save predicted label to file
        np.save("mood.npy", np.array([prediction]))

        # Draw face and hand landmarks on frame with specified drawing specifications
        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                               landmark_drawing_spec=drawing.DrawingSpec(color=(0,0,255), thickness=-1,
                                                                         circle_radius=1),
                               connection_drawing_spec=drawing.DrawingSpec(thickness=1))
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        # Return processed frame as VideoFrame object with BGR color format
        return av.VideoFrame.from_ndarray(frm, format="bgr24")


# Create Streamlit text inputs for language and artists
artists = st.text_input("Artists")
language = st.text_input("Language")

# If language and artists are specified and run state variable is not "false", start WebRTCStreamer with MoodProcessor
if language and artists and st.session_state["run"] != "false":
    webrtc_streamer(key="key", desired_playing_state=True, video_processor_factory=MoodProcessor)

# Create Streamlit button for song recommendation
btn = st.button("Song recommendations")

# If button is clicked, open YouTube search results for specified language, mood, and artists in web browser
if btn:
    webbrowser.open(f"https://www.youtube.com/results?search_query={language}+{mood}+song+{artists}")
    # Reset mood in file and set run state variable to "false"
    np.save("mood.npy", np.array([""]))
    st.session_state["run"] = "false"