# models.py - Definições dos modelos MLP, CNN e Híbrido (Keras)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_mlp_model(input_shape, num_classes):
    """
    Creates a Multi-Layer Perceptron (MLP) model for galaxy classification.

    Architecture (Listing 10 do enunciado):
        Input → Dense(256) → BN → Dropout(0.3)
              → Dense(128) → BN → Dropout(0.3)
              → Dense(64)  → BN → Dropout(0.2)
              → Dense(num_classes, softmax)

    Args:
        input_shape (int): Number of input features (e.g. 15).
        num_classes (int): Number of output classes (e.g. 3).

    Returns:
        keras.Model: Compiled Keras MLP model.
    """
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model
