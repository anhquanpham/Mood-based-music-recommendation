import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

# Load trained model and label list from files
model = load_model("model.h5")
label = np.load("labels.npy")

# Initialize holistic and hands solutions
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()

# Initialize drawing utilities
drawing = mp.solutions.drawing_utils

# Initialize video capture object
cap = cv2.VideoCapture(0)

while True:
    # Initialize list to store landmark data
    lst = []

    # Read frame from video capture object
    _, frm = cap.read()

    # Flip frame to eliminate mirror effect
    frm = cv2.flip(frm, 1)

    # Process frame with holistic solution
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

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
    pred = label[np.argmax(model.predict(lst))]

    # Print predicted label and display it on frame
    print(pred)
    cv2.putText(frm, pred, (50,50),cv2.FONT_ITALIC, 1, (255,0,0),2)

    # Draw face and hand landmarks on frame
    drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
    drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

    # Show frame in window
    cv2.imshow("window", frm)

    # If escape key is pressed, break loop
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
        break