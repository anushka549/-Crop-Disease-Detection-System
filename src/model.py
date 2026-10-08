"""MobileNetV2 transfer-learning model."""
import tensorflow as tf


def build_model(num_classes: int, input_shape=(224, 224, 3), learning_rate: float = 1e-3, weights="imagenet", verbose: bool = True):
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights=weights)
    base.trainable = False
    inputs = tf.keras.Input(shape=input_shape, name="leaf_image")
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="feature_dense")(x)
    x = tf.keras.layers.Dropout(0.35, name="classifier_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
    model = tf.keras.Model(inputs, outputs, name="crop_disease_mobilenetv2")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    if verbose:
        model.summary()
    return model
