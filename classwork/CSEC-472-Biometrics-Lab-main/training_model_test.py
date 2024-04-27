import os
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Step 1: Load and preprocess the data
def load_data(directory):
    data = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            img = load_img(os.path.join(directory, filename), target_size=(64, 64))
            img_array = img_to_array(img)
            data.append(img_array)
            labels.append(int(filename.startswith("match")))  # Assuming file names indicate matches or mismatches
    return np.array(data), np.array(labels)

train_data, train_labels = load_data("train")
test_data, test_labels = load_data("test")

# Step 2: Define and train the model
model = Sequential([
    Flatten(input_shape=(64, 64, 3)),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=10, batch_size=32)

# Step 3: Evaluate the model
loss, accuracy = model.evaluate(test_data, test_labels)
print("Test Accuracy:", accuracy)

# Step 4: Save the model to a pickle file
with open("my_neuralnetwork.pkl", "wb") as f:
    pickle.dump(model, f)
