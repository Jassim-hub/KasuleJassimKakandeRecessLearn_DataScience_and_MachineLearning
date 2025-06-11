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
DATASET_DIR = "LEARN_ML"  # Directory containing training
MODEL_PATH = "jeff_model.h5"  # File path to save a trained model


# Create a Convolutional Neural Network (CNN) model

def create_model(input_shape, num_classes):
    model = Sequential([

        Conv2D(32, (3, 3), activation='relu',
               input_shape=input_shape),  # filters, 3x3 kernel
        MaxPooling2D(2, 2),  # Downsample features maps by 2
        Conv2D(64, (3, 3), activation='relu'),  # increase to 64 the filter
        MaxPooling2D(2, 2),  # <-- Added missing comma
        Conv2D(128, (3, 3), activation='relu'),  # increase to 128 the filter
        MaxPooling2D(2, 2),  # <-- Added missing comma
        Conv2D(256, (3, 3), activation='relu'),  # increase to 256 the filter
        MaxPooling2D(2, 2),

        Flatten(),  # Flattern features map to 1D vector
        Dropout(0.5),  # Droppout to 50% of the neurons to prevent overfitting
        # Fully connected layers of neurons with 512
        Dense(512, activation='relu'),
        Dense(num_classes, activation='softmax')

    ])
