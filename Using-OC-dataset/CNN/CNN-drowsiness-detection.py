import numpy as np
import os
import cv2
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.metrics import classification_report

# Set labels
labels = os.listdir("oc-dataset/train")

# Function to get data for closed and open eye
def get_data(dir_path="oc-dataset/train", eye_cas="haarcascade.xml"):
    labels = ['Closed', 'Open']
    IMG_SIZE = 145
    data = []
    for label in labels:
        path = os.path.join(dir_path, label)
        class_num = labels.index(label)
        print(class_num)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append([resized_array, class_num])
            except Exception as e:
                print(e)
    return data

# Get new data
new_data = get_data()

# Separate labels and features
X = []
y = []
for feature, label in new_data:
    X.append(feature)
    y.append(label)

# Reshape the array
X = np.array(X)
X = X.reshape(-1, 145, 145, 3)

# LabelBinarizer
label_bin = LabelBinarizer()
y = label_bin.fit_transform(y)

# Convert y to array
y = np.array(y)

# Train-test split
seed = 42
test_size = 0.30
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test_size)

# Data Augmentation
train_generator = ImageDataGenerator(rescale=1/255, zoom_range=0.2, horizontal_flip=True, rotation_range=20)
test_generator = ImageDataGenerator(rescale=1/255)

train_generator = train_generator.flow(X_train, y_train, shuffle=False)
test_generator = test_generator.flow(X_test, y_test, shuffle=False)

# Model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=X_train.shape[1:]))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))  # Binary classification, so 1 neuron with sigmoid activation
model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")
model.summary()

# Train the model
history = model.fit(train_generator, epochs=5, validation_data=test_generator, shuffle=True, validation_steps=len(test_generator))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)

# Save the model
model.save("drowsiness_binary.h5")

# Predictions and evaluation
predictions = model.predict(test_generator)
prediction = (predictions > 0.5).astype(int)
labels_new = ["Closed", "Open"]

# Classification report
print(classification_report(y_test, prediction, target_names=labels_new))


