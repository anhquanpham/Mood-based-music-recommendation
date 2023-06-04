import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense
from keras.models import Model

# Initialize variables
is_init = False
size = -1
label = []
dictionary = {}
c = 0  # Counter variable for assigning integer values to labels
X = 0  # 2D numpy array for storing data from .npy files
Y = 0  # Unused variable
y = 0  # 2D numpy array for storing labels for data points in X array

# Loop through files in current directory
for i in os.listdir():
    # If file is a .npy file and not the labels file
    if i.split(".")[-1] == "npy" and not (i.split(".")[0] == "labels"):
        # If this is the first .npy file encountered
        if not is_init:
            # Load data from file into X array and set size variable
            is_init = True
            X = np.load(i)
            size = X.shape[0]
            # Create y array with label for each data point in X array
            y = np.array([i.split('.')[0]] * size).reshape(-1, 1)
        else:
            # Concatenate data from file with existing X array and update y array with labels for new data points
            X = np.concatenate((X, np.load(i)))
            y = np.concatenate((y, np.array([i.split('.')[0]] * size).reshape(-1, 1)))

        # Add label to label list and update dictionary with label and corresponding integer value
        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c
        c += 1

# Convert labels in y array to integer values using dictionary
for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")

# Convert integer labels in y array to one-hot encoded categorical labels
y = to_categorical(y)

# Create new arrays for shuffled data and labels
X_new = X.copy()
y_new = y.copy()
counter = 0

# Create array of indices and shuffle them randomly
cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)

# Use shuffled indices to shuffle data and labels in new arrays
for i in cnt:
    X_new[counter] = X[i]
    y_new[counter] = y[i]
    counter += 1

# Create input layer for model with shape of data points in X array
ip = Input(shape=(X.shape[1]))

# Create hidden layers with ReLU activation function
m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)

# Create output layer with softmax activation function and number of units equal to number of categories in y array
op = Dense(y.shape[1], activation="softmax")(m)

# Create model with specified input and output layers
model = Model(inputs=ip, outputs=op)

# Compile model with RMSprop optimizer and categorical crossentropy loss function
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

# Fit model to data for 50 epochs
model.fit(X, y, epochs=50)

# Save trained model and label list to files
model.save("model.h5")
np.save("labels.npy", np.array(label))
