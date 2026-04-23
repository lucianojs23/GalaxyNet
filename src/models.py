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
        layers.BatchNormalization(momentum=0.9),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(momentum=0.9),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(momentum=0.9),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model


def create_cnn_model(input_shape=(64, 64, 3), num_classes=3):
    """
    Creates a Convolutional Neural Network (CNN) for galaxy image classification.

    Architecture (Listing 11 do enunciado):
        4 blocos convolucionais (Conv2D → BN → MaxPool → Dropout)
        com filtros crescentes: 32 → 64 → 128 → 256.
        Seguidos de Flatten → Dense(512) → BN → Dropout(0.5) → Softmax.

    Args:
        input_shape (tuple): Shape of input images (height, width, channels).
        num_classes   (int): Number of output classes.

    Returns:
        keras.Model: Compiled Keras CNN model.
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(momentum=0.9),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(momentum=0.9),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(momentum=0.9),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(momentum=0.9),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Classifier head
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(momentum=0.9),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model


def create_hybrid_model(tabular_shape, image_shape=(64, 64, 3), num_classes=3):
    """
    Creates a Hybrid model combining CNN (images) + MLP (tabular) branches.

    Architecture (Listing 12 do enunciado):
        CNN branch: 4 blocos Conv2D(3x3,same)/BN/MaxPool/Dropout(0.25)
                    filtros 32→64→128→256, Flatten → Dense(256)/BN/Dropout(0.5)
        MLP branch: Dense(128)/BN/Dropout(0.3) → Dense(64)/BN/Dropout(0.3)
        Fusão:      Concatenate → Dense(128)/BN/Dropout(0.5) → Softmax

    Args:
        tabular_shape (int): Number of tabular input features (e.g. 15).
        image_shape (tuple): Shape of input images (height, width, channels).
        num_classes  (int): Number of output classes.

    Returns:
        keras.Model: Compiled Keras Functional API model with two inputs
                     (named 'tabular_input' and 'image_input').
    """
    # ── CNN branch (images) ──────────────────────────────────────────────────
    image_input = layers.Input(shape=image_shape, name='image_input')

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    cnn_out = layers.Dropout(0.5)(x)

    # ── MLP branch (tabular features) ───────────────────────────────────────
    tabular_input = layers.Input(shape=(tabular_shape,), name='tabular_input')

    t = layers.Dense(128, activation='relu')(tabular_input)
    t = layers.BatchNormalization(momentum=0.9)(t)
    t = layers.Dropout(0.3)(t)

    t = layers.Dense(64, activation='relu')(t)
    t = layers.BatchNormalization(momentum=0.9)(t)
    mlp_out = layers.Dropout(0.3)(t)

    # ── Fusion ───────────────────────────────────────────────────────────────
    merged = layers.Concatenate()([cnn_out, mlp_out])

    f = layers.Dense(128, activation='relu')(merged)
    f = layers.BatchNormalization(momentum=0.9)(f)
    f = layers.Dropout(0.5)(f)

    output = layers.Dense(num_classes, activation='softmax', name='output')(f)

    model = keras.Model(inputs=[tabular_input, image_input], outputs=output,
                        name='hybrid_galaxy_classifier')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model
