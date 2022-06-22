"""# **Aula 13 - Redes Neurais com Tensorflow + Keras**"""

# tensorflow XOR
# TensorFlow e tf.keras
import tensorflow as tf
from tensorflow import keras

# Libraries auxiliares
import numpy as np

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(4,)),
    keras.layers.Dense(256, activation='sigmoid'),
    keras.layers.Dense(256, activation='sigmoid'),
    keras.layers.Dense(4, activation='relu')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
model.summary()
model.fit(np.array([[0,1,1,0]]), np.array([[0,1,1,0]]), epochs=20)

predictions = model.predict(np.array([[0,1,1,0]]))

print(predictions)