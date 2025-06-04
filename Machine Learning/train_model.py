# import required libraries
# tensorflow,numpy,matplotlib
import os
import numpy as np
import tensorflow as tf  # For building and training deep learning model
from tensorflow import keras  # -High level API for neural networks
from tensorflow.keras.processing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.models import sequential  # type: ignore #Linear Stack of neural Networks
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flattern, Dense, Dropout  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore #Optimizer for training
from tensorflow.keras.callbacks import ModelCheckpoint  # type: ignore #Training call backs
import matplotlib as plt  # type: ignore

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
# Define the constant value
IMAGE_SIZE = (256, 256)  # Input Image size
BATCH_SIZE = 32  # Number of images processed in a batch
EPOCHS = 28
NUM_CLASSES = 2  # Numbber of output classes for crops(diseases,healthy)
ANIMAL_CLASSES = 3  # Number of animal classes(cat,dog,human)

# Define the dataset directory and the model save paths
DATASET_DIR = "Machine Learning"  # Directory bcontaing training data
MODEL_PATH = "jassim_model.h5"
