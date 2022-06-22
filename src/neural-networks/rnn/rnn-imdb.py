import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import imdb 

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))
number_of_words = 20000
max_len = 300

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=number_of_words)

X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_len)
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_len)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=number_of_words, output_dim=128, input_shape=(X_train.shape[1],)))
model.add(tf.keras.layers.LSTM(units=128, activation='tanh'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
rmsprop = tf.keras.optimizers.RMSprop(learning_rate=0.0009)
# Compilando o modelo
model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

epochs = 5
batch_size= 50
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1, shuffle=False)

# plot history
from matplotlib import pyplot
pyplot.clf()
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

test_loss, test_acurracy = model.evaluate(X_test, y_test)
print("Test accuracy: {}".format(test_acurracy))

predictions = model.predict(X_test)
print(predictions[0:9])

print(y_test[0:9])