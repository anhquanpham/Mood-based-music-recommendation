import cv2
import mediapipe as mp
import numpy as np

# Initialize video capture object
cap = cv2.VideoCapture(0)

# Get name of data from user
name = input("Enter the name of the data : ")

# Initialize holistic and hands solutions
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()

# Initialize drawing utilities
drawing = mp.solutions.drawing_utils

# Initialize data array and size counter
X = []
data_size = 0

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

    # Append list of landmark data to data array and increment data size counter
    X.append(lst)
    data_size += 1

    # Draw face and hand landmarks on frame
    drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
    drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

    # Display data size on frame
    cv2.putText(frm, str(data_size), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

    # Show frame in window
    cv2.imshow("window", frm)

    # If escape key is pressed or data size exceeds 99, break loop
    if cv2.waitKey(1) == 27 or data_size > 99:
        cv2.destroyAllWindows()
        cap.release()
        break

# Save data array to file with specified name
np.save(f"{name}.npy", np.array(X))
print(np.array(X).shape)
