# Import required libraries

# tensorflow, numpy , matplotlib
# 3.13.0   i recommend python 3.12.3 or 3.11.9  , you need to downgrade

# pip install scipy

# File extension    .py        .ipynb

# For interacting with the operating system (like reading directory contents)
import os
import numpy as np  # type: ignore # For numerical operations

# TensorFlow for building and training the deep learning model
import tensorflow as tf

# High-level TensorFlow API for neural networks
from tensorflow import keras  # type: ignore

# For data augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore

# Linear stack of neural network layers
from tensorflow.keras.models import Sequential  # type: ignore

# Layers used in CNN
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # type: ignore

# Optimizer for training
from tensorflow.keras.optimizers import Adam  # type: ignore

# Training callbacks
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping  # type: ignore
import matplotlib.pyplot as plt  # type: ignore # For plotting training history
from tensorflow.keras.layers import Input  # type: ignore

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define the constant value
IMAGE_SIZE = (256, 256)  # Input image size
BATCH_SIZE = 32  # Number of images process in a batch
EPOCHS = 20  # Number of full passes throught the dataset
NUM_CLASSES = 2  # Number of output classes for crops (diseases, healthy)
ANIMAL_CLASSES = 3  # Number of animal class (cat, dog and human)

# Define the dataset directory and the model save  paths
DATASET_DIR = (
    "Machine Learning"  # Directory containing training and validation datasets
)
MODEL_PATH = "jassim_model.h5"  # File path to save a trained model


# Create a Convolutional Neural Network (CNN) model


def create_model(input_shape, num_classes):
    model = Sequential(
        [
            Input(shape=input_shape),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(256, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),
            Dropout(0.5),
            Dense(512, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )
    return model


if __name__ == "__main__":
    # Choose input shape and number of classes
    input_shape = (256, 256, 3)  # 3 for RGB images
    num_classes = NUM_CLASSES  # or ANIMAL_CLASSES if you want 3 classes

    # Create the model
    model = create_model(input_shape, num_classes)

    # Compile the model
    model.compile(
        optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Prepare data generators
    train_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
    )
    val_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
    )

    # Set up callbacks (optional)
    callbacks = [
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_loss"),
        # EarlyStopping(patience=3, restore_best_weights=True),
    ]

    # Train the model
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
    )

    # Plot training history (optional)
    plt.plot(history.history["accuracy"], label="train acc")
    plt.plot(history.history["val_accuracy"], label="val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
