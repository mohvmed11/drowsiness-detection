import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
import seaborn as sns
import time
import random

folder_path = f"output"
os.makedirs(folder_path, exist_ok=True)

# Path dataset
dataset_dir = r"dataset/train"

class_names = []
features = []
labels = []
image_size = (227, 227)  # we reduce image size to improve memory consumption

total_samples_limit = 6000  # Set the desired limit

# Get all files in the dataset directory
all_files = []
for path, subdirs, files in os.walk(dataset_dir):
    for name in tqdm(files):
        all_files.append(os.path.join(path, name))

# Shuffle the files to get a random half of the dataset
random.shuffle(all_files)

for img_path in tqdm(all_files[:total_samples_limit]):
    if img_path.endswith(".jpg"): #type of files
        image_read = cv2.imread(img_path)
        image_resized = cv2.resize(image_read, image_size) 
        image_normalized = image_resized / 255.0  

        path_parts = img_path.split('/')
        label = path_parts[-2]  # Use the second-to-last part as the label

        if label not in class_names:
            class_names.append(label)

        features.append(image_normalized)
        index = class_names.index(label)
        labels.append(index)

features = np.asarray(features)
labels = np.asarray(labels)

# Train-test split
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.3, shuffle=True, random_state=42
)

# Hyperparameters
learning_rate = 0.001
epochs = 10
batch_size = 16

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense

# Define the model
model = Sequential()

# Add layers similar to the first code
model.add(Conv2D(256, (3, 3), activation="relu", input_shape=(640, 480, 3)))
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
model.add(Dense(2, activation="softmax"))  # 2 classes: drowsy and not drowsy

model.compile(optimizer=Adam(learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

model_checkpoint = ModelCheckpoint(os.path.join(folder_path, f"best_model.h5"), monitor='val_loss', save_best_only=True)
csv_logger = CSVLogger(os.path.join(folder_path, f"log.csv"), separator=',', append=False)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Hitung waktu training
start_time = time.time()

# Training
history = model.fit(
    features_train,
    labels_train,
    epochs=epochs,
    validation_data=(features_test, labels_test),
    callbacks=[model_checkpoint, csv_logger, early_stopping],
    batch_size=batch_size,
)

# Hitung waktu training
end_time = time.time()

print(f"Training Time : {end_time - start_time}")

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(folder_path, 'loss.png'))
plt.show()

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(folder_path, 'accuracy.png'))
plt.show()

y_pred = model.predict(features_test)

y_pred = np.argmax(y_pred, axis=1)

classification_rep = classification_report(labels_test, y_pred, target_names=class_names, digits=4)
print("Classification Report:\n", classification_rep)

with open(os.path.join(folder_path, 'classification_report.txt'), 'w') as file:
    file.write(classification_rep)

cm = confusion_matrix(labels_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.savefig(os.path.join(folder_path, 'confusion matrix.png'))
plt.show()

