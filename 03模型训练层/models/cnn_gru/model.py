"""TensorFlow/Keras小型CNN-GRU模型。"""

from __future__ import annotations

from typing import Dict


def build_cnn_gru(input_shape, params: Dict, huber_delta: float):
    import tensorflow as tf

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(
        filters=int(params["cnn_filters"]),
        kernel_size=int(params["kernel_size"]),
        activation="relu",
        padding="same",
    )(inputs)
    x = tf.keras.layers.Dropout(float(params["dropout"]))(x)
    x = tf.keras.layers.GRU(int(params["gru_hidden_size"]))(x)
    x = tf.keras.layers.Dropout(float(params["dropout"]))(x)
    outputs = tf.keras.layers.Dense(1, activation="linear", dtype="float32")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(params["learning_rate"])),
        loss=tf.keras.losses.Huber(delta=float(huber_delta)),
    )
    return model
