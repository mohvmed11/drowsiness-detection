import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

# Set labels
labels = os.listdir("mrl-eye-dataset/mrleyedataset")

# Function to get data for closed and open eyes
def get_data(dir_path="mrl-eye-dataset/mrleyedataset", eye_cas="haarcascade.xml", num_samples_per_class=8000):
    labels = ['Close-Eyes', 'Open-Eyes']
    IMG_SIZE = 145
    data = []
    for label in labels:
        path = os.path.join(dir_path, label)
        class_num = labels.index(label)
        print(class_num)
        count = 0
        for img in os.listdir(path):
            if count >= num_samples_per_class:
                break
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append([resized_array, class_num])
                count += 1
            except Exception as e:
                print(e)
    return data

# Get new data 
new_data = get_data(num_samples_per_class=8000)

# Separate labels and features
X = []
y = []
for feature, label in new_data:
    X.append(feature)
    y.append(label)

# Reshape the array
X = np.array(X)
X = X.reshape(-1, 145, 145, 3)

# Train-test split
seed = 42
test_size = 0.30
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test_size)

# Data Augmentation
train_datagen = ImageDataGenerator(rescale=1/255, zoom_range=0.2, horizontal_flip=True, rotation_range=20)
test_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow(X_train, y_train, shuffle=False)
test_generator = test_datagen.flow(X_test, y_test, shuffle=False)

# CNN Model
model = Sequential()
model.add(Conv2D(256, (3, 3), activation="relu", input_shape=(145, 145, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dense(1, activation="sigmoid"))  # Output layer for binary classification
model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")

# Train the model
history = model.fit(train_generator, epochs=5, validation_data=test_generator, shuffle=True, validation_steps=len(test_generator))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)

# Save the model
model.save("drowsiness_binary_cnn.h5")

# Predictions and evaluation
predictions = model.predict(test_generator)
prediction = (predictions > 0.5).astype(int)
labels_new = ["Close-Eyes", "Open-Eyes"]

# Flatten y_test if it's a 2D list
y_test_flat = np.array(y_test).flatten()

# Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test_flat, prediction, target_names=labels_new))



