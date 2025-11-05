import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

data_dir = "dataset/"
labels = os.listdir(data_dir)
images, targets = [], []

for i, label in enumerate(labels):
    for img_name in os.listdir(f"{data_dir}/{label}"):
        img_path = os.path.join(data_dir, label, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (100, 100))
        images.append(img)
        targets.append(i)

X = np.array(images) / 255.0
y = to_categorical(np.array(targets), num_classes=len(labels))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(100,100,3)),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
model.save("trained_model/sign_model.keras")


# Save labels
with open("trained_model/labels.txt", "w") as f:
    f.write('\n'.join(labels))
