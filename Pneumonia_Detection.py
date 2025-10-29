# Pneumonia Detection using CNN
# Author: Tejan

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Define paths
train_dir = "chest_xray/train"
val_dir = "chest_xray/val"
test_dir = "chest_xray/test"

# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(train_dir,
                                              target_size=(150, 150),
                                              batch_size=32,
                                              class_mode='binary')

val_set = test_datagen.flow_from_directory(val_dir,
                                           target_size=(150, 150),
                                           batch_size=32,
                                           class_mode='binary')

test_set = test_datagen.flow_from_directory(test_dir,
                                            target_size=(150, 150),
                                            batch_size=32,
                                            class_mode='binary')

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Train the model
history = model.fit(train_set,
                    validation_data=val_set,
                    epochs=10,
                    batch_size=32)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_set)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Save model
model.save('pneumonia_detection_model.h5')

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
