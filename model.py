# model.py
# Build a transfer-learning model using a lightweight base (MobileNetV2)
# and add a classifier head. Returns compiled model.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(num_classes, img_size=(224,224,3), base_trainable=False):
    base_model = keras.applications.MobileNetV2(
        input_shape=img_size,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = base_trainable

    inputs = keras.Input(shape=img_size)
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
