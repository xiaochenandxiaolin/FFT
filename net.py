import tensorflow as tf
from tensorflow.keras import layers


def create_model():
    model = tf.keras.Sequential([
        layers.Conv1D(64, 3, activation="relu", input_shape=(8192, 1)),
        # layers.Conv1D(64, 1, activation="relu", ),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(32, 3, activation="relu"),
        # layers.Conv1D(32, 1, activation="relu"),
        layers.Conv1D(16, 3, activation="relu"),
        layers.Dropout(0.4),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        # layers.Conv1D(16, 1, activation="relu"),
        layers.Conv1D(8, 3, activation="relu"),
        # layers.GlobalMaxPool1D(),
        # layers.Conv1D(8, 1, activation="relu"),
        layers.BatchNormalization(),
        layers.GlobalMaxPool1D(),
        # tf.keras.layers.Flatten(),
        layers.Dense(8, activation="relu"),
        layers.Dense(4, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        layers.Dense(2,  activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    ])
    # Dense(2, kernel_initializer='he_normal', activation='softmax', kernel_regularizer=l2(0.0001))
    model.summary()
    return model