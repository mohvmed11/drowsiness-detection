import numpy as np
import pandas as pd
import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import save_model
from keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

# Set labels
labels = os.listdir("dataset/train")

# Read an image
a = plt.imread("dataset/train/yawn/10.jpg")
print(a.shape)

# Function to extract faces for yawn and not_yawn
def face_for_yawn(direc="dataset/train", face_cas_path="haarcascade_frontalface_default.xml"):
    yaw_no = []
    IMG_SIZE = 145
    categories = ["yawn", "no_yawn"]
    for category in categories:
        path_link = os.path.join(direc, category)
        class_num1 = categories.index(category)
        print(class_num1)
        for image in os.listdir(path_link):
            image_array = cv2.imread(os.path.join(path_link, image), cv2.IMREAD_COLOR)
            face_cascade = cv2.CascadeClassifier(face_cas_path)
            faces = face_cascade.detectMultiScale(image_array, 1.3, 5)
            for (x, y, w, h) in faces:
                img = cv2.rectangle(image_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
                roi_color = img[y:y+h, x:x+w]
                resized_array = cv2.resize(roi_color, (IMG_SIZE, IMG_SIZE))
                yaw_no.append([resized_array, class_num1])
    return yaw_no

# Function to get data for closed and open eye
def get_data(dir_path="dataset/train/", face_cas="haarcascade_frontalface_default.xml", eye_cas="haarcascade.xml"):
    labels = ['Closed', 'Open']
    IMG_SIZE = 145
    data = []
    for label in labels:
        path = os.path.join(dir_path, label)
        class_num = labels.index(label)
        class_num += 2
        print(class_num)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append([resized_array, class_num])
            except Exception as e:
                print(e)
    return data

# Function to append data
def append_data():
    yaw_no = face_for_yawn()
    data = get_data()
    yaw_no.extend(data)
    return np.array(yaw_no)

# Get new data
new_data = append_data()

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
train_generator = ImageDataGenerator(rescale=1/255, zoom_range=0.2, horizontal_flip=True, rotation_range=30)
test_generator = ImageDataGenerator(rescale=1/255)

train_generator = train_generator.flow(np.array(X_train), y_train, shuffle=False)
test_generator = test_generator.flow(np.array(X_test), y_test, shuffle=False)

# Load the pre-trained VGG16 model (excluding the top layer)
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(145, 145, 3))

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Create a new model by adding custom layers on top of the pre-trained model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=50, validation_data=test_generator, shuffle=True, validation_steps=len(test_generator))

# Evaluate the model on the test set
accuracy = model.evaluate(test_generator)

# Print the test accuracy
print("Test Accuracy:", accuracy[1])

# Save the model
model.save("drowsiness_transfer_learning.h5")

# Prediction function
labels_new = ["yawn", "no_yawn", "Closed", "Open"]
IMG_SIZE = 145
def prepare(filepath, face_cas="haarcascade_frontalface_default.xml"):
    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    img_array = img_array / 255
    resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return resized_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

# Load the model
model = tf.keras.models.load_model("drowsiness_transfer_learning.h5")

# Classification report
from sklearn.metrics import classification_report

# Predictions on the entire test set
predictions = model.predict(test_generator)

# Choose the class with the highest probability as the predicted class
predicted_classes = np.argmax(predictions, axis=1)

# Print the classification report
print(classification_report(np.argmax(y_test, axis=1), predicted_classes, target_names=labels_new))

