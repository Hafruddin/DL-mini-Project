"""
Convolutional Neural Network model definition for waste classification.
Implements the 3-block Conv2D architecture with MaxPooling, Dropout, and Dense classification head.
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers

from config import (
    IMG_HEIGHT,
    IMG_WIDTH,
    CHANNELS,
    INPUT_SHAPE,
    TARGET_CLASSES,
    LEARNING_RATE
)


def build_waste_cnn(num_classes: int = len(TARGET_CLASSES)) -> keras.Model:
    """
    Builds the baseline 3-block Convolutional Neural Network:
    - Input: 128 x 128 x 3 RGB image (normalized [0, 1])
    - Block 1: Conv2D(32, (3, 3), relu) + MaxPooling2D((2, 2))
    - Block 2: Conv2D(64, (3, 3), relu) + MaxPooling2D((2, 2))
    - Block 3: Conv2D(128, (3, 3), relu) + MaxPooling2D((2, 2))
    - Classification Head:
        - Flatten
        - Dense(128, relu)
        - Dropout(0.5)
        - Dense(6, softmax)
    """
    inputs = keras.Input(shape=INPUT_SHAPE, name="waste_image_input")

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1")(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)

    # Classification Head
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(128, activation="relu", name="dense_features")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="waste_probabilities")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="WasteClassificationCNN")

    # Compile with Adam optimizer and Sparse Categorical Crossentropy
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def get_data_augmentation_layer() -> keras.Sequential:
    """
    Data augmentation pipeline applied strictly during training only:
    - Random modest rotations
    - Random horizontal flips
    - Modest zoom variations
    - Small translation variations
    """
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal", name="aug_flip"),
            layers.RandomRotation(0.10, name="aug_rotation"),
            layers.RandomZoom(0.10, name="aug_zoom"),
            layers.RandomTranslation(height_factor=0.08, width_factor=0.08, name="aug_translation"),
        ],
        name="training_data_augmentation",
    )


if __name__ == "__main__":
    model = build_waste_cnn()
    model.summary()
