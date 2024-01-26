import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras
from keras import layers

def myencoder(width, image_size, image_channels):
    return keras.Sequential(
        [
            keras.Input(shape=(image_size, image_size, image_channels)),
            layers.Conv2D(width, kernel_size = 3, strides = 2 , activation = "relu"),
            layers.Flatten(),
            layers.Dense(width, activation = "relu"),
        ],
        name = "base.encoder"
)



