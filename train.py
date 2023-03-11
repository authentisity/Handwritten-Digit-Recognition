import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# base model

# model = tf.keras.Sequential([
#   tf.keras.layers.Rescaling(1./255, input_shape=(28, 28, 1)),
#   tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L1(0.01)),
#   tf.keras.layers.Dense(10)
# ])

model = tf.keras.models.load_model("model.h5")

training = tf.keras.utils.image_dataset_from_directory(
    directory='data/',
    validation_split=0.2,
    subset="training",
    labels='inferred',
    label_mode='categorical',
    color_mode='grayscale',
    image_size=(28, 28),
    seed=123
)

validation = tf.keras.utils.image_dataset_from_directory(
    directory='data/',
    validation_split=0.2,
    subset="training",
    labels='inferred',
    label_mode='categorical',
    color_mode='grayscale',
    image_size=(28, 28),
    seed=123
)

training = training.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

try:
    model.fit(training, validation_data=validation, epochs=50)
except KeyboardInterrupt:
    model.save("model.h5")
else:
    model.save("model.h5")